"""Whether the console sends HDR, as an Elgato HD60 X capture card sees it (Experimental).

Windows' camera driver describes the card's 10-bit P010 video the same way whether the console sends HDR10 or SDR
(BT.709, no transfer function), so the picture alone can't say how to read it: SDR read as HDR10 comes out too bright
and oversaturated, HDR10 read as SDR washed out. The card itself knows. It has a small vendor HID interface next to its
video and sound, through which Elgato's own apps read the HDMI input's state, and that's read here, the same way
Elgato's MIT-licensed sample code does (github.com/elgatosf/capture-device-support, ElgatoUVCDevice.cpp): an I2C read
of the card's microcontroller (address 0x55) through one HID output report and one input report. Nothing is written.

Which register: the sample reads 0x09, the HDMI HDR infoframe. On a real HD60 X with a PS5 that one keeps the last HDR
infoframe it got after the console leaves HDR (the PS5 stops sending it rather than sending one that says SDR), so it
says "PQ" with SDR video coming in. Register 0x00, the input signal, changes with every switch: width and height
(16-bit, little-endian, from byte 1 and 3), frame rate (byte 5), and byte 8, 1 while the console sends HDR and 0 in
SDR; all zero for a moment while the console switches (no signal). Found by watching every register while a PS5 turned
its HDR on and off (HD60 X Rev. 2, firmware 25.07.16). Anything that doesn't look like that is taken as "can't tell".

The card's own HDR tone mapping (register 0x0A, Elgato's SetHDRTonemappingEnabled; on by default, and Elgato's apps
switch it): while it's on, HDR coming in leaves the card as SDR, BT.709, in every format, P010 too. So a frame is HDR10
only with HDR coming in and tone mapping off. For HDR output SmoovPlay turns it off while it plays (HdrWatch.keep_hdr)
and puts it back as it was when it closes: with it off, the card's P010 matched a PS5's Remote Play HDR stream of the
same screen to about 2 codes in 1023; with it on, read as HDR10 it was a tenth as bright and oversaturated.

    python -m framegen.elgato        # what the card says, once a second until Ctrl+C
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

ELGATO = 0x0FD9
HD60_X = {0x0082: "HD60 X", 0x008A: "HD60 X Rev. 2"}  # the "new chip" HID protocol (Elgato's IsNewDeviceType)
MCU = 0x55
INPUT_STATUS = 0x00
TONE_MAPPING = 0x0A  # 1: HDR in, SDR out (the card's own tone mapping); 0: HDR10 out
POLL_S = 0.5


@dataclass(frozen=True)
class InputStatus:
    """The card's HDMI input: what the console sends. w == 0 means no signal (the console switching modes)."""
    w: int
    h: int
    fps: int
    hdr: bool

    @property
    def signal(self) -> bool:
        return self.w > 0


NO_SIGNAL = InputStatus(0, 0, 0, False)


def parse_input_status(data: bytes) -> InputStatus | None:
    """Register 0x00 as read: an InputStatus, NO_SIGNAL (all zero), or None when it doesn't look like the layout
    above (another card or firmware): then the card can't say."""
    if len(data) < 9:
        return None
    if not any(data[:9]):
        return NO_SIGNAL
    w, h, fps, flag = int.from_bytes(data[1:3], "little"), int.from_bytes(data[3:5], "little"), data[5], data[8]
    if not (320 <= w <= 7680 and 200 <= h <= 4320 and 20 <= fps <= 250 and flag in (0, 1)):
        return None
    return InputStatus(w, h, fps, flag == 1)


class Hd60x:
    """An Elgato HD60 X's vendor HID interface, for reading its microcontroller's registers. open() -> None when
    there's no HD60 X (or more than one, as which goes with which video device can't be told)."""

    def __init__(self, handle, name: str, out_len: int) -> None:
        self._h, self.name, self._out_len = handle, name, out_len
        self._lock = threading.Lock()

    @classmethod
    def open(cls) -> "Hd60x | None":
        if sys.platform != "win32":
            return None
        import ctypes
        from ctypes import wintypes as wt
        hid, cfg = ctypes.WinDLL("hid"), ctypes.WinDLL("cfgmgr32")
        k32 = ctypes.WinDLL("kernel32")
        k32.CreateFileW.restype = wt.HANDLE
        k32.CreateFileW.argtypes = (wt.LPCWSTR, wt.DWORD, wt.DWORD, ctypes.c_void_p, wt.DWORD, wt.DWORD, wt.HANDLE)
        k32.CloseHandle.argtypes = (wt.HANDLE,)

        class Guid(ctypes.Structure):
            _fields_ = [("d1", wt.DWORD), ("d2", wt.WORD), ("d3", wt.WORD), ("d4", ctypes.c_ubyte * 8)]

        class Attributes(ctypes.Structure):
            _fields_ = [("Size", wt.ULONG), ("VendorID", wt.USHORT), ("ProductID", wt.USHORT),
                        ("Version", wt.USHORT)]

        guid = Guid()
        hid.HidD_GetHidGuid(ctypes.byref(guid))
        size = wt.ULONG()
        if cfg.CM_Get_Device_Interface_List_SizeW(ctypes.byref(size), ctypes.byref(guid), None, 0) != 0:
            return None
        buf = ctypes.create_unicode_buffer(max(1, size.value))
        if cfg.CM_Get_Device_Interface_ListW(ctypes.byref(guid), None, buf, size.value, 0) != 0:
            return None
        found = []
        for path in (p for p in buf[:size.value].split("\0") if f"vid_{ELGATO:04x}" in p.lower()):
            h = k32.CreateFileW(path, 0xC0000000, 3, None, 3, 0, None)  # read/write, shared, OPEN_EXISTING
            if not h or h == wt.HANDLE(-1).value:
                continue
            a = Attributes(Size=ctypes.sizeof(Attributes))
            if hid.HidD_GetAttributes(h, ctypes.byref(a)) and a.ProductID in HD60_X:
                pre = ctypes.c_void_p()
                caps = (ctypes.c_ushort * 32)()  # HIDP_CAPS: usage, usage page, input/output report lengths, ...
                if hid.HidD_GetPreparsedData(h, ctypes.byref(pre)):
                    hid.HidP_GetCaps(pre, caps)
                    hid.HidD_FreePreparsedData(pre)
                if caps[3] >= 8:
                    found.append((h, HD60_X[a.ProductID], caps[3]))
                    continue
            k32.CloseHandle(h)
        if len(found) != 1:
            for h, _, _ in found:
                k32.CloseHandle(h)
            return None
        card = cls(*found[0])
        card._hid, card._k32 = hid, k32
        return card

    def read(self, register: int, n: int = 32) -> bytes | None:
        """n bytes of an MCU register, or None if the card didn't answer. Report 6 out: [length, 7 = I2C read,
        address, 1 byte written (the register), the register, bytes to read]; report 5 in, its length carrying the
        report case (7) in its high byte, as Elgato's code sends it."""
        import ctypes
        with self._lock:
            if self._h is None:
                return None
            out = (ctypes.c_ubyte * self._out_len)(6, 6, 7, MCU, 1, register, n)
            if not self._hid.HidD_SetOutputReport(self._h, out, self._out_len):
                return None
            size = 0xFF | (7 << 8)
            inp = (ctypes.c_ubyte * size)(5)
            if not self._hid.HidD_GetInputReport(self._h, inp, size):
                return None
            return bytes(inp[1:1 + n])

    def write(self, register: int, value: int) -> bool:
        """One byte into an MCU register. Report 6 out: [length, 6 = I2C write, address, 2 bytes written (the
        register and the value), the register, the value], as Elgato's WriteI2cData sends it."""
        import ctypes
        with self._lock:
            if self._h is None:
                return False
            out = (ctypes.c_ubyte * self._out_len)(6, 6, 6, MCU, 2, register, value)
            return bool(self._hid.HidD_SetOutputReport(self._h, out, self._out_len))

    def input_status(self) -> InputStatus | None:
        data = self.read(INPUT_STATUS)
        return parse_input_status(data) if data is not None else None

    def tone_mapping(self) -> bool | None:
        """Whether the card turns HDR into SDR itself; None if it didn't say."""
        data = self.read(TONE_MAPPING, 4)
        return bool(data[0]) if data is not None and data[0] in (0, 1) else None

    def set_tone_mapping(self, on: bool) -> bool:
        """Turns the card's tone mapping on or off; True once it reads back so."""
        if not self.write(TONE_MAPPING, 1 if on else 0):
            return False
        for _ in range(10):
            if self.tone_mapping() == on:
                return True
            time.sleep(0.05)
        return False

    def close(self) -> None:
        with self._lock:
            if self._h is not None:
                self._k32.CloseHandle(self._h)
                self._h = None


# What keep_hdr changed, until it's put back: a SmoovPlay that ends without closing the card (killed, or crashed) would
# otherwise leave the card's own tone mapping off, and other apps would get HDR from it as washed-out SDR.
RESTORE_NOTE = Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()) / "SmoovPlay" / "elgato-tone-mapping.txt"


def _remember_restore(on: bool) -> None:
    try:
        RESTORE_NOTE.parent.mkdir(parents=True, exist_ok=True)
        RESTORE_NOTE.write_text("1" if on else "0", encoding="ascii")
    except OSError:
        pass


def _restore_pending() -> bool | None:
    try:
        text = RESTORE_NOTE.read_text(encoding="ascii").strip()
    except OSError:
        return None
    return {"1": True, "0": False}.get(text)


def _forget_restore() -> None:
    try:
        RESTORE_NOTE.unlink()
    except OSError:
        pass


class HdrWatch:
    """Whether the card's frames are HDR10, kept up to date from the card on a thread (every POLL_S): HDR coming in
    and the card's tone mapping off. Callable: watch() -> bool. status: the input as the card last said it with a
    signal (a switch's moment without one changes nothing); tone_mapping: the card's own HDR-to-SDR, on or off.
    keep_hdr() turns that off for HDR output, and close() puts it back. open(video device name) -> None unless that's
    an Elgato HD60 X whose input it can read."""

    def __init__(self, card, first: InputStatus, tone_mapping: bool = False, poll_s: float = POLL_S,
                 log=print) -> None:
        self.card, self.poll_s, self.log = card, poll_s, log
        self.status, self.tone_mapping = first, tone_mapping
        self.hdr = first.hdr and not tone_mapping
        self._restore: bool | None = None  # the tone mapping to put back at close (keep_hdr changed it)
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True, name="elgato-hdr").start()

    @classmethod
    def open(cls, video_device: str, log=print) -> "HdrWatch | None":
        if "hd60 x" not in video_device.lower():
            return None
        try:
            card = Hd60x.open()
        except OSError:
            return None
        if card is None:
            return None
        pending = _restore_pending()
        if pending is not None:  # a SmoovPlay that ended without closing the card left its tone mapping changed
            if card.set_tone_mapping(pending):
                _forget_restore()
                log("[capture] put the card's own HDR tone mapping back on, as an earlier SmoovPlay left it off")
        first, tone = card.input_status(), card.tone_mapping()
        if first is None or tone is None:  # not the layout this was written for: can't tell
            card.close()
            return None
        return cls(card, first, tone, log=log)

    def __call__(self) -> bool:
        return self.hdr

    def describe(self) -> str:
        s = self.status
        sent = (f"the console sends {s.w}x{s.h} at {s.fps} Hz in {'HDR' if s.hdr else 'SDR'}" if s.signal else
                "no signal from the console yet")
        return sent + (", and the card's own HDR tone mapping is on (HDR comes out as SDR)" if self.tone_mapping
                       else "")

    def keep_hdr(self) -> bool:
        """For HDR output: the card's tone mapping off, so HDR stays HDR10, until close() puts it back as it was.
        True when it's off."""
        if not self.tone_mapping:
            return True
        _remember_restore(True)  # first, so a SmoovPlay that ends without closing still gets it put back
        if not self.card.set_tone_mapping(False):
            _forget_restore()
            self.log("[capture] couldn't turn off the card's own HDR tone mapping: HDR from the console comes as SDR")
            return False
        self._restore, self.tone_mapping = True, False
        import atexit
        atexit.register(self._at_exit)  # also when SmoovPlay ends without closing the source
        self.hdr = self.status.hdr
        self.log("[capture] the card's own HDR tone mapping is off while SmoovPlay plays, so the console's HDR stays "
                 "HDR10 (back on when it closes)")
        return True

    def _run(self) -> None:
        while not self._stop.wait(self.poll_s):
            try:
                s, tone = self.card.input_status(), self.card.tone_mapping()
            except Exception:  # noqa: BLE001 - the card went away: keep the last answer
                continue
            if tone is not None and tone != self.tone_mapping:  # (Elgato's app switched it: frames follow)
                self.tone_mapping = tone
                self.log(f"[capture] the card's own HDR tone mapping was turned {'on' if tone else 'off'}")
            if s is not None and s.signal:
                if s.hdr != self.status.hdr:
                    self.log(f"[capture] the console switched to {'HDR' if s.hdr else 'SDR'}")
                self.status = s
            hdr = self.status.hdr and not self.tone_mapping
            if hdr != self.hdr:
                self.hdr = hdr
                self.log(f"[capture] the picture is now read as {'HDR10' if hdr else 'SDR'}")

    def _put_back(self, tries: int = 6, wait_s: float = 0.5) -> bool:
        """The tone mapping back as keep_hdr found it, checked, a few times if the card doesn't take it at once (a
        USB device can be busy for a moment as its video stops). If it still won't, the note on disk has the next
        open() put it back."""
        if self._restore is None:
            return True
        for i in range(tries):
            if i:
                time.sleep(wait_s)
            if self.card.set_tone_mapping(self._restore):
                self._restore = None
                _forget_restore()
                self.log("[capture] the card's own HDR tone mapping is back on")
                return True
        self.log("[capture] couldn't turn the card's own HDR tone mapping back on: SmoovPlay does it the next time it "
                 "opens the card, or turn it on in Elgato's app")
        return False

    def _at_exit(self) -> None:
        if not self._stop.is_set():  # (close() has had its go, and the note covers the rest)
            self._put_back()

    def close(self) -> None:
        self._stop.set()
        self._put_back()
        self.card.close()


def main() -> int:
    card = Hd60x.open()
    if card is None:
        print("no Elgato HD60 X found (or more than one)")
        return 1
    print(f"Elgato {card.name}")
    try:
        while True:
            data = card.read(INPUT_STATUS)
            s = parse_input_status(data) if data is not None else None
            what = ("no answer" if data is None else "can't tell: " + data[:9].hex() if s is None else
                    "no signal" if not s.signal else f"{s.w}x{s.h} {s.fps} Hz {'HDR' if s.hdr else 'SDR'}")
            print(time.strftime("%H:%M:%S"), what, flush=True)
            time.sleep(1.0)
    except KeyboardInterrupt:
        return 0
    finally:
        card.close()


if __name__ == "__main__":
    sys.exit(main())

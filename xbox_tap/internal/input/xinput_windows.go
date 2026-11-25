//go:build windows

package input

import (
	"syscall"
	"unsafe"
)

// Win32 XInput button masks (XINPUT_GAMEPAD_*)
const (
	xinputDPadUp        uint16 = 0x0001
	xinputDPadDown      uint16 = 0x0002
	xinputDPadLeft      uint16 = 0x0004
	xinputDPadRight     uint16 = 0x0008
	xinputStart         uint16 = 0x0010
	xinputBack          uint16 = 0x0020
	xinputLeftThumb     uint16 = 0x0040
	xinputRightThumb    uint16 = 0x0080
	xinputLeftShoulder  uint16 = 0x0100
	xinputRightShoulder uint16 = 0x0200
	xinputGuide         uint16 = 0x0400 // exposed by undocumented XInputGetStateEx (ordinal 100)
	xinputA             uint16 = 0x1000
	xinputB             uint16 = 0x2000
	xinputX             uint16 = 0x4000
	xinputY             uint16 = 0x8000
)

type xinputGamepad struct {
	Buttons      uint16
	LeftTrigger  uint8
	RightTrigger uint8
	ThumbLX      int16
	ThumbLY      int16
	ThumbRX      int16
	ThumbRY      int16
}

type xinputState struct {
	PacketNumber uint32
	Gamepad      xinputGamepad
}

type xinputVibration struct {
	LeftMotorSpeed  uint16
	RightMotorSpeed uint16
}

var (
	xinputDLL      *syscall.LazyDLL
	addrGetStateEx uintptr
	procGetState   *syscall.LazyProc
	procSetState   *syscall.LazyProc
)

func init() {
	kernel32 := syscall.NewLazyDLL("kernel32.dll")
	procGetProcAddress := kernel32.NewProc("GetProcAddress")

	for _, name := range []string{"xinput1_4.dll", "xinput1_3.dll", "xinput9_1_0.dll"} {
		dll := syscall.NewLazyDLL(name)
		if err := dll.Load(); err == nil {
			xinputDLL = dll
			procGetState = dll.NewProc("XInputGetState")
			procSetState = dll.NewProc("XInputSetState")
			// Ordinal 100 is XInputGetStateEx which also reports the Xbox Guide button (0x0400).
			// Win32 GetProcAddress resolves ordinals when passed as an integer uintptr(100)
			// (MAKEINTRESOURCEA(100)), not as a Go string.
			if h, err2 := syscall.LoadLibrary(name); err2 == nil {
				if addr, _, _ := procGetProcAddress.Call(uintptr(h), uintptr(100)); addr != 0 {
					addrGetStateEx = addr
				} else {
					_ = syscall.FreeLibrary(h)
				}
			}
			break
		}
	}
}

// PollGamepad reads controller `index` (0..3) via XInput and returns its state in Xbox wire format.
func PollGamepad(index uint8) GamepadState {
	if (addrGetStateEx == 0 && procGetState == nil) || index > 3 {
		return GamepadState{Index: index}
	}
	var st xinputState
	var ret uintptr
	if addrGetStateEx != 0 {
		ret, _, _ = syscall.SyscallN(addrGetStateEx, uintptr(index), uintptr(unsafe.Pointer(&st)))
	} else {
		ret, _, _ = procGetState.Call(uintptr(index), uintptr(unsafe.Pointer(&st)))
	}
	if ret != 0 {
		return GamepadState{Index: index, Connected: false}
	}
	gp := st.Gamepad
	return GamepadState{
		Index:       index,
		Connected:   true,
		Buttons:     mapXInputButtons(gp.Buttons),
		LeftThumbX:  gp.ThumbLX,
		LeftThumbY:  invertAxis(gp.ThumbLY),
		RightThumbX: gp.ThumbRX,
		RightThumbY: invertAxis(gp.ThumbRY),
		// Scale 0..255 -> 0..65535
		LeftTrigger:  uint16(gp.LeftTrigger) * 257,
		RightTrigger: uint16(gp.RightTrigger) * 257,
	}
}

// SetRumble applies left/right motor speeds (0.0..1.0) to XInput controller `index`.
func SetRumble(index uint8, left, right float32) {
	if procSetState == nil || index > 3 {
		return
	}
	clamp := func(v float32) uint16 {
		if v <= 0 {
			return 0
		}
		if v >= 1 {
			return 65535
		}
		return uint16(v * 65535)
	}
	vib := xinputVibration{
		LeftMotorSpeed:  clamp(left),
		RightMotorSpeed: clamp(right),
	}
	_, _, _ = procSetState.Call(uintptr(index), uintptr(unsafe.Pointer(&vib)))
}

func invertAxis(v int16) int16 {
	if v == -32768 {
		return 32767
	}
	return -v
}

func mapXInputButtons(b uint16) uint16 {
	var out uint16
	if b&xinputGuide != 0 {
		out |= BtnNexus
	}
	if b&xinputStart != 0 {
		out |= BtnMenu
	}
	if b&xinputBack != 0 {
		out |= BtnView
	}
	if b&xinputA != 0 {
		out |= BtnA
	}
	if b&xinputB != 0 {
		out |= BtnB
	}
	if b&xinputX != 0 {
		out |= BtnX
	}
	if b&xinputY != 0 {
		out |= BtnY
	}
	if b&xinputDPadUp != 0 {
		out |= BtnDPadUp
	}
	if b&xinputDPadDown != 0 {
		out |= BtnDPadDown
	}
	if b&xinputDPadLeft != 0 {
		out |= BtnDPadLeft
	}
	if b&xinputDPadRight != 0 {
		out |= BtnDPadRight
	}
	if b&xinputLeftShoulder != 0 {
		out |= BtnLeftShoulder
	}
	if b&xinputRightShoulder != 0 {
		out |= BtnRightShoulder
	}
	if b&xinputLeftThumb != 0 {
		out |= BtnLeftThumb
	}
	if b&xinputRightThumb != 0 {
		out |= BtnRightThumb
	}
	return out
}

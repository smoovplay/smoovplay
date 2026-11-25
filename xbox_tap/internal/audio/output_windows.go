//go:build windows

package audio

import (
	"fmt"
	"sync"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

const (
	waveMapper     = 0xFFFFFFFF
	waveFormatPCM  = 1
	callbackEvent  = 0x00050000
	whdrDone       = 0x00000001
	whdrPrepared   = 0x00000002
	numWaveBuffers = 8
)

var (
	winmm                 = syscall.NewLazyDLL("winmm.dll")
	procWaveOutOpen       = winmm.NewProc("waveOutOpen")
	procWaveOutPrepareHdr = winmm.NewProc("waveOutPrepareHeader")
	procWaveOutUnprepHdr  = winmm.NewProc("waveOutUnprepareHeader")
	procWaveOutWrite      = winmm.NewProc("waveOutWrite")
	procWaveOutReset      = winmm.NewProc("waveOutReset")
	procWaveOutClose      = winmm.NewProc("waveOutClose")
)

type waveFormatEx struct {
	FormatTag      uint16
	Channels       uint16
	SamplesPerSec  uint32
	AvgBytesPerSec uint32
	BlockAlign     uint16
	BitsPerSample  uint16
	Size           uint16
}

type waveHdr struct {
	Data          uintptr
	BufferLength  uint32
	BytesRecorded uint32
	User          uintptr
	Flags         uint32
	Loops         uint32
	Next          uintptr
	Reserved      uintptr
}

type waveOutSink struct {
	mu     sync.Mutex
	handle uintptr
	event  windows.Handle
	hdrs   [numWaveBuffers]waveHdr
	bufs   [numWaveBuffers][]byte
	next   int
	closed bool
}

// OpenDefaultSink opens Windows' default audio playback device via winmm waveOut.
func OpenDefaultSink(sampleRate, channels int) (Sink, error) {
	ev, err := windows.CreateEvent(nil, 0, 0, nil)
	if err != nil {
		return nil, fmt.Errorf("CreateEvent: %w", err)
	}
	blockAlign := uint16(channels * 2)
	wf := waveFormatEx{
		FormatTag:      waveFormatPCM,
		Channels:       uint16(channels),
		SamplesPerSec:  uint32(sampleRate),
		AvgBytesPerSec: uint32(sampleRate) * uint32(blockAlign),
		BlockAlign:     blockAlign,
		BitsPerSample:  16,
		Size:           0,
	}
	var h uintptr
	r, _, _ := procWaveOutOpen.Call(
		uintptr(unsafe.Pointer(&h)),
		waveMapper,
		uintptr(unsafe.Pointer(&wf)),
		uintptr(ev),
		0,
		callbackEvent,
	)
	if r != 0 {
		_ = windows.CloseHandle(ev)
		return nil, fmt.Errorf("waveOutOpen failed (mmresult=%d)", r)
	}
	s := &waveOutSink{handle: h, event: ev}
	for i := range s.hdrs {
		s.hdrs[i].Flags = whdrDone // mark free initially
	}
	return s, nil
}

func (s *waveOutSink) WritePCM(pcm []byte) error {
	if len(pcm) == 0 {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}

	idx := s.next
	s.next = (s.next + 1) % numWaveBuffers
	hdr := &s.hdrs[idx]

	// Wait briefly if this slot is still playing (keeps max latency bounded to numWaveBuffers chunks).
	for hdr.Flags&whdrDone == 0 {
		_, _ = windows.WaitForSingleObject(s.event, 50)
		if s.closed {
			return nil
		}
	}
	if hdr.Flags&whdrPrepared != 0 {
		_, _, _ = procWaveOutUnprepHdr.Call(s.handle, uintptr(unsafe.Pointer(hdr)), unsafe.Sizeof(*hdr))
	}

	if cap(s.bufs[idx]) < len(pcm) {
		s.bufs[idx] = make([]byte, len(pcm))
	}
	s.bufs[idx] = s.bufs[idx][:len(pcm)]
	copy(s.bufs[idx], pcm)

	*hdr = waveHdr{
		Data:         uintptr(unsafe.Pointer(&s.bufs[idx][0])),
		BufferLength: uint32(len(pcm)),
	}
	if r, _, _ := procWaveOutPrepareHdr.Call(s.handle, uintptr(unsafe.Pointer(hdr)), unsafe.Sizeof(*hdr)); r != 0 {
		return fmt.Errorf("waveOutPrepareHeader failed (mmresult=%d)", r)
	}
	if r, _, _ := procWaveOutWrite.Call(s.handle, uintptr(unsafe.Pointer(hdr)), unsafe.Sizeof(*hdr)); r != 0 {
		return fmt.Errorf("waveOutWrite failed (mmresult=%d)", r)
	}
	return nil
}

func (s *waveOutSink) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	if s.handle != 0 {
		_, _, _ = procWaveOutReset.Call(s.handle)
		for i := range s.hdrs {
			if s.hdrs[i].Flags&whdrPrepared != 0 {
				_, _, _ = procWaveOutUnprepHdr.Call(s.handle, uintptr(unsafe.Pointer(&s.hdrs[i])), unsafe.Sizeof(s.hdrs[i]))
			}
		}
		_, _, _ = procWaveOutClose.Call(s.handle)
		s.handle = 0
	}
	if s.event != 0 {
		_ = windows.CloseHandle(s.event)
		s.event = 0
	}
	return nil
}

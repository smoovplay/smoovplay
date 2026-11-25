//go:build windows

package frametap

import (
	"fmt"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

var (
	kernel32                 = syscall.NewLazyDLL("kernel32.dll")
	procQueryPerformanceFreq = kernel32.NewProc("QueryPerformanceFrequency")
	procQueryPerformanceCnt  = kernel32.NewProc("QueryPerformanceCounter")
)

type platformState struct {
	mapHandle      windows.Handle
	audioMapHandle windows.Handle
	eventHandle    windows.Handle
	viewAddr       uintptr
	audioViewAddr  uintptr
}

func queryQPCFreq() uint64 {
	var freq int64
	_, _, _ = procQueryPerformanceFreq.Call(uintptr(unsafe.Pointer(&freq)))
	if freq <= 0 {
		return 10_000_000
	}
	return uint64(freq)
}

func queryQPC() uint64 {
	var cnt int64
	_, _, _ = procQueryPerformanceCnt.Call(uintptr(unsafe.Pointer(&cnt)))
	return uint64(cnt)
}

func u32ptr(b []byte, off int) *uint32 {
	return (*uint32)(unsafe.Pointer(&b[off]))
}

func u64ptr(b []byte, off int) *uint64 {
	return (*uint64)(unsafe.Pointer(&b[off]))
}

func (w *Writer) openPlatform() error {
	w.qpcFreq = queryQPCFreq()

	namePtr, err := windows.UTF16PtrFromString(w.name)
	if err != nil {
		return err
	}
	hMap, err := windows.CreateFileMapping(
		windows.InvalidHandle, nil, windows.PAGE_READWRITE,
		uint32(uint64(TotalSize)>>32), uint32(TotalSize), namePtr,
	)
	if err != nil && err != windows.ERROR_ALREADY_EXISTS {
		return fmt.Errorf("CreateFileMapping(%s): %w", w.name, err)
	}
	addr, err := windows.MapViewOfFile(hMap, windows.FILE_MAP_READ | windows.FILE_MAP_WRITE, 0, 0, TotalSize)
	if err != nil {
		_ = windows.CloseHandle(hMap)
		return fmt.Errorf("MapViewOfFile(%s): %w", w.name, err)
	}

	evPtr, err := windows.UTF16PtrFromString(w.eventName)
	if err != nil {
		_ = windows.UnmapViewOfFile(addr)
		_ = windows.CloseHandle(hMap)
		return err
	}
	hEvent, err := windows.CreateEvent(nil, 0, 0, evPtr)
	if err != nil && err != windows.ERROR_ALREADY_EXISTS {
		_ = windows.UnmapViewOfFile(addr)
		_ = windows.CloseHandle(hMap)
		return fmt.Errorf("CreateEvent(%s): %w", w.eventName, err)
	}

	// Companion audio ring (optional — if it fails, video still works).
	var hAudioMap windows.Handle
	var audioAddr uintptr
	if aPtr, aErr := windows.UTF16PtrFromString(w.name + AudioSuffix); aErr == nil {
		if hm, hmErr := windows.CreateFileMapping(
			windows.InvalidHandle, nil, windows.PAGE_READWRITE,
			0, AudioTotalSize, aPtr,
		); (hmErr == nil || hmErr == windows.ERROR_ALREADY_EXISTS) && hm != 0 {
			if aa, aaErr := windows.MapViewOfFile(hm, windows.FILE_MAP_READ | windows.FILE_MAP_WRITE, 0, 0, AudioTotalSize); aaErr == nil {
				hAudioMap = hm
				audioAddr = aa
				w.audioMem = unsafe.Slice((*byte)(unsafe.Pointer(aa)), AudioTotalSize)
			} else {
				_ = windows.CloseHandle(hm)
			}
		}
	}

	w.platform = platformState{
		mapHandle:      hMap,
		audioMapHandle: hAudioMap,
		eventHandle:    hEvent,
		viewAddr:       addr,
		audioViewAddr:  audioAddr,
	}
	w.mem = unsafe.Slice((*byte)(unsafe.Pointer(addr)), TotalSize)
	return nil
}

func (w *Writer) signalFrame() {
	if w.platform.eventHandle != 0 {
		_ = windows.SetEvent(w.platform.eventHandle)
	}
}

func (w *Writer) closePlatform() {
	if w.platform.viewAddr != 0 {
		_ = windows.UnmapViewOfFile(w.platform.viewAddr)
		w.platform.viewAddr = 0
	}
	if w.platform.mapHandle != 0 {
		_ = windows.CloseHandle(w.platform.mapHandle)
		w.platform.mapHandle = 0
	}
	if w.platform.audioViewAddr != 0 {
		_ = windows.UnmapViewOfFile(w.platform.audioViewAddr)
		w.platform.audioViewAddr = 0
	}
	if w.platform.audioMapHandle != 0 {
		_ = windows.CloseHandle(w.platform.audioMapHandle)
		w.platform.audioMapHandle = 0
	}
	if w.platform.eventHandle != 0 {
		_ = windows.CloseHandle(w.platform.eventHandle)
		w.platform.eventHandle = 0
	}
	w.mem = nil
	w.audioMem = nil
}

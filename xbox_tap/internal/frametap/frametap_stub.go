//go:build !windows

package frametap

import (
	"time"
	"unsafe"
)

type platformState struct {
	signaled int
}

var startTime = time.Now()

func queryQPCFreq() uint64 { return 1_000_000_000 }

func queryQPC() uint64 {
	return uint64(time.Since(startTime).Nanoseconds())
}

func u32ptr(b []byte, off int) *uint32 {
	return (*uint32)(unsafe.Pointer(&b[off]))
}

func u64ptr(b []byte, off int) *uint64 {
	return (*uint64)(unsafe.Pointer(&b[off]))
}

func (w *Writer) openPlatform() error {
	w.qpcFreq = queryQPCFreq()
	w.mem = make([]byte, TotalSize)
	w.audioMem = make([]byte, AudioTotalSize)
	return nil
}

func (w *Writer) signalFrame() {
	w.platform.signaled++
}

func (w *Writer) closePlatform() {
	w.mem = nil
	w.audioMem = nil
}

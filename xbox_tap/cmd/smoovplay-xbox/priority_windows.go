//go:build windows

package main

import (
	"log"
	"syscall"
)

const highPriorityClass = 0x00000080 // HIGH_PRIORITY_CLASS

func setHighPriority() {
	kernel32 := syscall.NewLazyDLL("kernel32.dll")
	getCurrentProcess := kernel32.NewProc("GetCurrentProcess")
	setPriorityClass := kernel32.NewProc("SetPriorityClass")

	h, _, _ := getCurrentProcess.Call()
	r, _, err := setPriorityClass.Call(h, highPriorityClass)
	if r == 0 {
		log.Printf("[xbox-tap] warning: SetPriorityClass(HIGH_PRIORITY_CLASS) failed: %v", err)
	}
}

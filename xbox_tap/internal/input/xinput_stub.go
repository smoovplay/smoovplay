//go:build !windows

package input

// PollGamepad is a no-op stub on non-Windows platforms.
func PollGamepad(index uint8) GamepadState {
	return GamepadState{Index: index, Connected: false}
}

// SetRumble is a no-op stub on non-Windows platforms.
func SetRumble(index uint8, left, right float32) {}

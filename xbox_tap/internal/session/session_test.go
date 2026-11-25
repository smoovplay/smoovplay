package session

import (
	"sync"
	"testing"
	"time"

	"github.com/noveblue53/smoovplay/xbox_tap/internal/input"
)

func TestSyntheticPressLifecycle(t *testing.T) {
	var s syntheticPress
	t0 := time.Unix(1000, 0)

	// Zero button mask is a no-op
	if s.trigger(t0, 0, 100) {
		t.Fatal("expected trigger(0) = false")
	}
	if btns, rel := s.step(t0); btns != 0 || rel {
		t.Fatalf("idle step = (0x%x, %v), want (0, false)", btns, rel)
	}

	// Trigger Menu for 100 ms
	if !s.trigger(t0, input.BtnMenu, 100) {
		t.Fatal("expected trigger(BtnMenu) = true")
	}
	if btns, rel := s.step(t0); btns != input.BtnMenu || rel {
		t.Fatalf("at t0: step = (0x%x, %v), want (BtnMenu, false)", btns, rel)
	}
	if btns, rel := s.step(t0.Add(50 * time.Millisecond)); btns != input.BtnMenu || rel {
		t.Fatalf("at t0+50ms: step = (0x%x, %v), want (BtnMenu, false)", btns, rel)
	}
	// At t0+100ms: expires and signals release once
	if btns, rel := s.step(t0.Add(100 * time.Millisecond)); btns != 0 || !rel {
		t.Fatalf("at t0+100ms: step = (0x%x, %v), want (0, true)", btns, rel)
	}
	// Subsequent tick: idle again
	if btns, rel := s.step(t0.Add(108 * time.Millisecond)); btns != 0 || rel {
		t.Fatalf("at t0+108ms: step = (0x%x, %v), want (0, false)", btns, rel)
	}
}

type rumbleCall struct {
	idx   uint8
	left  float32
	right float32
}

func TestRumbleTimerSuperseding(t *testing.T) {
	var mu sync.Mutex
	var calls []rumbleCall
	record := func(idx uint8, left, right float32) {
		mu.Lock()
		calls = append(calls, rumbleCall{idx: idx, left: left, right: right})
		mu.Unlock()
	}
	snapshot := func() []rumbleCall {
		mu.Lock()
		defer mu.Unlock()
		out := make([]rumbleCall, len(calls))
		copy(out, calls)
		return out
	}

	rc := newRumbleController(record)
	defer rc.stopAll()

	// First vibration on pad 0: 40 ms
	rc.apply(input.VibrationReport{GamepadIndex: 0, LeftMotor: 0.8, RightMotor: 0.5, DurationMs: 40})
	// Independent vibration on pad 1: 30 ms (must not be cancelled by pad 0)
	rc.apply(input.VibrationReport{GamepadIndex: 1, LeftMotor: 0.4, RightMotor: 0.4, DurationMs: 30})

	// After 15 ms, supersede pad 0 with a longer 100 ms vibration
	time.Sleep(15 * time.Millisecond)
	rc.apply(input.VibrationReport{GamepadIndex: 0, LeftMotor: 1.0, RightMotor: 0.2, DurationMs: 100})

	// At t = 65 ms (past pad 0's original 40 ms and pad 1's 30 ms, well before pad 0's new 115 ms):
	// pad 1 should have zeroed, pad 0 must NOT have zeroed.
	time.Sleep(50 * time.Millisecond)
	got := snapshot()
	for _, c := range got {
		if c.idx == 0 && c.left == 0 && c.right == 0 {
			t.Fatalf("pad 0 zeroed prematurely after superseding packet: %+v", got)
		}
	}
	pad1Zeroed := false
	for _, c := range got {
		if c.idx == 1 && c.left == 0 && c.right == 0 {
			pad1Zeroed = true
		}
	}
	if !pad1Zeroed {
		t.Fatalf("expected pad 1 to zero after 30 ms, calls=%+v", got)
	}

	// Wait past pad 0's superseded 100 ms timer and verify it zeroes once.
	time.Sleep(85 * time.Millisecond)
	got = snapshot()
	pad0ZeroCount := 0
	for _, c := range got {
		if c.idx == 0 && c.left == 0 && c.right == 0 {
			pad0ZeroCount++
		}
	}
	if pad0ZeroCount != 1 {
		t.Fatalf("expected pad 0 to zero exactly once, got %d (calls=%+v)", pad0ZeroCount, got)
	}

	// Verify a DurationMs == 0 packet cancels a pending zeroing timer.
	rc.apply(input.VibrationReport{GamepadIndex: 0, LeftMotor: 0.6, RightMotor: 0.6, DurationMs: 30})
	time.Sleep(10 * time.Millisecond)
	rc.apply(input.VibrationReport{GamepadIndex: 0, LeftMotor: 0.9, RightMotor: 0.9, DurationMs: 0})
	time.Sleep(45 * time.Millisecond)
	got = snapshot()
	last := got[len(got)-1]
	if last.idx != 0 || last.left != 0.9 || last.right != 0.9 {
		t.Fatalf("expected DurationMs=0 packet to cancel pending zeroing timer, last call=%+v", last)
	}
}

func TestInputStateChangeDispatch(t *testing.T) {
	t0 := time.Unix(2000, 0)
	pad0 := input.GamepadState{Index: 0, Connected: true}

	var sent [][]byte
	var pressSentCount int
	var pendingChiaki uint32
	var pendingHold uint32

	disp := &inputDispatcher{
		start:        t0,
		seq:          2,
		wasConnected: [4]bool{true, false, false, false},
		pollGamepad: func(index uint8) input.GamepadState {
			if index == 0 {
				return pad0
			}
			return input.GamepadState{Index: index}
		},
		takePressRequest: func() (uint32, uint32) {
			b, h := pendingChiaki, pendingHold
			pendingChiaki, pendingHold = 0, 0
			return b, h
		},
		onPressSent: func() {
			pressSentCount++
		},
		sendPacket: func(pkt []byte) error {
			sent = append(sent, pkt)
			return nil
		},
	}

	// t0: first poll dispatches immediately
	if !disp.step(t0) || len(sent) != 1 {
		t.Fatalf("t0: expected initial send, sent=%d", len(sent))
	}

	// t0+4ms: unchanged state -> skipped (not yet 8 ms heartbeat)
	if disp.step(t0.Add(4*time.Millisecond)) || len(sent) != 1 {
		t.Fatalf("t0+4ms idle: expected skip, sent=%d", len(sent))
	}

	// t0+8ms: unchanged state -> 125 Hz idle heartbeat dispatches
	if !disp.step(t0.Add(8*time.Millisecond)) || len(sent) != 2 {
		t.Fatalf("t0+8ms idle: expected 125 Hz heartbeat, sent=%d", len(sent))
	}

	// t0+12ms (only 4 ms after heartbeat): button press -> dispatches immediately
	pad0.Buttons = input.BtnA
	if !disp.step(t0.Add(12*time.Millisecond)) || len(sent) != 3 {
		t.Fatalf("t0+12ms press: expected immediate send on state change, sent=%d", len(sent))
	}

	// t0+16ms: button still held, unchanged -> skipped
	if disp.step(t0.Add(16*time.Millisecond)) || len(sent) != 3 {
		t.Fatalf("t0+16ms held: expected skip, sent=%d", len(sent))
	}

	// t0+20ms: button released -> dispatches immediately
	pad0.Buttons = 0
	if !disp.step(t0.Add(20*time.Millisecond)) || len(sent) != 4 {
		t.Fatalf("t0+20ms release: expected immediate send on release, sent=%d", len(sent))
	}

	// t0+24ms: synthetic FEATURE_PRESS request -> dispatches immediately & calls onPressSent
	pendingChiaki = 1 << 12 // BUTTON_OPTIONS -> BtnMenu
	pendingHold = 40
	if !disp.step(t0.Add(24*time.Millisecond)) || len(sent) != 5 {
		t.Fatalf("t0+24ms synth press: expected immediate send, sent=%d", len(sent))
	}
	if pressSentCount != 1 {
		t.Fatalf("expected onPressSent count=1, got %d", pressSentCount)
	}

	// t0+28ms: synthetic still held, unchanged -> skipped
	if disp.step(t0.Add(28*time.Millisecond)) || len(sent) != 5 {
		t.Fatalf("t0+28ms synth held: expected skip, sent=%d", len(sent))
	}

	// t0+64ms: synthetic expires -> dispatches release immediately
	if !disp.step(t0.Add(64*time.Millisecond)) || len(sent) != 6 {
		t.Fatalf("t0+64ms synth release: expected immediate send on release, sent=%d", len(sent))
	}
}

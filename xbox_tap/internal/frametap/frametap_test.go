package frametap

import (
	"bytes"
	"encoding/binary"
	"testing"
)

func TestFrameTapHeaderAndSlotLayout(t *testing.T) {
	w, err := Open("test-tap")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer w.Close()

	feats := FeatureAudioDelay | FeatureAudioTap
	w.Begin(CodecH264, 2560, 1440, 60, feats)

	if got := binary.LittleEndian.Uint32(w.mem[0:4]); got != Magic {
		t.Fatalf("magic = 0x%x, want 0x%x", got, Magic)
	}
	if got := binary.LittleEndian.Uint32(w.mem[4:8]); got != Version {
		t.Fatalf("version = %d, want %d", got, Version)
	}
	if got := binary.LittleEndian.Uint32(w.mem[8:12]); got != SlotCount {
		t.Fatalf("slot_count = %d, want %d", got, SlotCount)
	}
	if got := binary.LittleEndian.Uint32(w.mem[12:16]); got != SlotSize {
		t.Fatalf("slot_size = %d, want %d", got, SlotSize)
	}
	if got := binary.LittleEndian.Uint32(w.mem[20:24]); got != 2560 {
		t.Fatalf("width = %d, want 2560", got)
	}
	if got := binary.LittleEndian.Uint32(w.mem[24:28]); got != 1440 {
		t.Fatalf("height = %d, want 1440", got)
	}
	if got := binary.LittleEndian.Uint32(w.mem[64:68]); got != feats {
		t.Fatalf("features = %d, want %d", got, feats)
	}

	payload := []byte{0x00, 0x00, 0x00, 0x01, 0x65, 0xAA, 0xBB, 0xCC}
	w.Push(payload, 2, true)

	if got := binary.LittleEndian.Uint64(w.mem[32:40]); got != 1 {
		t.Fatalf("write_seq = %d, want 1", got)
	}
	off := HeaderSize
	if got := binary.LittleEndian.Uint64(w.mem[off : off+8]); got != 0 {
		t.Fatalf("slot seq = %d, want 0", got)
	}
	if got := binary.LittleEndian.Uint32(w.mem[off+16 : off+20]); got != uint32(len(payload)) {
		t.Fatalf("slot size = %d, want %d", got, len(payload))
	}
	if got := int32(binary.LittleEndian.Uint32(w.mem[off+20 : off+24])); got != 2 {
		t.Fatalf("frames_lost = %d, want 2", got)
	}
	if got := binary.LittleEndian.Uint32(w.mem[off+24 : off+28]); got != 1 {
		t.Fatalf("recovered = %d, want 1", got)
	}
	if !bytes.Equal(w.mem[off+SlotHeader:off+SlotHeader+len(payload)], payload) {
		t.Fatalf("slot payload mismatch")
	}

	// IDR backchannel
	binary.LittleEndian.PutUint32(w.mem[40:44], 1)
	if !w.TakeIDRRequest() {
		t.Fatal("expected TakeIDRRequest() = true")
	}
	if w.TakeIDRRequest() {
		t.Fatal("expected second TakeIDRRequest() = false")
	}

	// Audio delay backchannel (requires fresh heartbeat)
	binary.LittleEndian.PutUint64(w.mem[56:64], queryQPC())
	binary.LittleEndian.PutUint32(w.mem[88:92], 42)
	if got := w.AudioDelayMs(); got != 42 {
		t.Fatalf("AudioDelayMs = %d, want 42", got)
	}

	// Companion audio ring
	w.BeginAudio(48000, 2)
	if got := binary.LittleEndian.Uint32(w.audioMem[0:4]); got != AudioMagic {
		t.Fatalf("audio magic = 0x%x, want 0x%x", got, AudioMagic)
	}
	pcm := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	w.PushAudio(pcm, 48000, 2)
	if got := binary.LittleEndian.Uint64(w.audioMem[24:32]); got != 1 {
		t.Fatalf("audio write_seq = %d, want 1", got)
	}
	if !bytes.Equal(w.audioMem[AudioHeaderSize+AudioSlotHeader:AudioHeaderSize+AudioSlotHeader+len(pcm)], pcm) {
		t.Fatalf("audio payload mismatch")
	}
}

func TestPressBackchannel(t *testing.T) {
	w, err := Open("test-tap-press")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer w.Close()

	// Pre-populate stale press request; Begin must clear offset 68.
	binary.LittleEndian.PutUint32(w.mem[68:72], 1<<12)
	w.Begin(CodecH264, 1920, 1080, 60, FeaturePress|FeatureAudioDelay)
	if btns, hold := w.TakePressRequest(); btns != 0 || hold != 0 {
		t.Fatalf("expected stale press cleared by Begin, got (%d, %d)", btns, hold)
	}

	// Normal request (Options = 1<<12, hold 100 ms)
	binary.LittleEndian.PutUint32(w.mem[72:76], 100)
	binary.LittleEndian.PutUint32(w.mem[68:72], 1<<12)
	btns, hold := w.TakePressRequest()
	if btns != (1<<12) || hold != 100 {
		t.Fatalf("TakePressRequest = (0x%x, %d), want (0x1000, 100)", btns, hold)
	}
	// Atomic clear: second call returns 0
	if b2, h2 := w.TakePressRequest(); b2 != 0 || h2 != 0 {
		t.Fatalf("second TakePressRequest = (%d, %d), want (0, 0)", b2, h2)
	}

	// Hold clamping: <20 -> 20, >2000 -> 2000
	binary.LittleEndian.PutUint32(w.mem[72:76], 5)
	binary.LittleEndian.PutUint32(w.mem[68:72], 1<<12)
	if _, h := w.TakePressRequest(); h != 20 {
		t.Fatalf("clamped min hold = %d, want 20", h)
	}
	binary.LittleEndian.PutUint32(w.mem[72:76], 9999)
	binary.LittleEndian.PutUint32(w.mem[68:72], 1<<12)
	if _, h := w.TakePressRequest(); h != 2000 {
		t.Fatalf("clamped max hold = %d, want 2000", h)
	}

	// PressSent stamps QPC at offset 80 and increments press_count at offset 76
	beforeCount := binary.LittleEndian.Uint32(w.mem[76:80])
	w.PressSent()
	afterCount := binary.LittleEndian.Uint32(w.mem[76:80])
	qpcVal := binary.LittleEndian.Uint64(w.mem[80:88])
	if afterCount != beforeCount+1 {
		t.Fatalf("press_count = %d, want %d", afterCount, beforeCount+1)
	}
	if qpcVal == 0 {
		t.Fatalf("expected non-zero QPC timestamp at offset 80")
	}
}

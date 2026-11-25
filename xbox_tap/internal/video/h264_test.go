package video

import (
	"bytes"
	"testing"

	"github.com/pion/rtp"
)

func TestParseSPSDimensions1080pAnd1440p(t *testing.T) {
	// Real H.264 High Profile 1920x1080 SPS (with bottom crop=4 -> 1088-8=1080)
	sps1080 := []byte{
		0x67, 0x64, 0x00, 0x2a, 0xac, 0xd9, 0x40, 0x78,
		0x02, 0x27, 0xe5, 0xc0, 0x44, 0x00, 0x00, 0x03,
		0x00, 0x04, 0x00, 0x00, 0x03, 0x00, 0xf0, 0x3c,
		0x60, 0xc6, 0x58,
	}
	w, h, ok := ParseSPSDimensions(sps1080)
	if !ok || w != 1920 || h != 1080 {
		t.Fatalf("1080p SPS = (%d, %d, %v), want (1920, 1080, true)", w, h, ok)
	}

	// Real H.264 Baseline 1280x720 SPS
	sps720 := []byte{
		0x67, 0x42, 0xe0, 0x1f, 0xda, 0x01, 0x40, 0x16,
		0xec, 0x04, 0x40, 0x00, 0x00, 0x03, 0x00, 0x40,
		0x00, 0x00, 0x0f, 0x03, 0xc6, 0x0c, 0x65, 0x80,
	}
	w, h, ok = ParseSPSDimensions(sps720)
	if !ok || w != 1280 || h != 720 {
		t.Fatalf("720p SPS = (%d, %d, %v), want (1280, 720, true)", w, h, ok)
	}
}

func TestDepacketizerFUAAndLossDetection(t *testing.T) {
	d := NewDepacketizer(60)
	// FU-A fragmented IDR frame (nalType = 5, 2 packets)
	p1 := &rtp.Packet{
		Header:  rtp.Header{SequenceNumber: 100, Timestamp: 90000, Marker: false},
		Payload: []byte{0x7C, 0x85, 0xAA, 0xBB}, // FU-A start, IDR
	}
	p2 := &rtp.Packet{
		Header:  rtp.Header{SequenceNumber: 101, Timestamp: 90000, Marker: true},
		Payload: []byte{0x7C, 0x45, 0xCC, 0xDD}, // FU-A end, IDR
	}
	if f := d.Push(p1); f != nil {
		t.Fatalf("expected nil before marker, got frame")
	}
	f := d.Push(p2)
	if f == nil || !f.Keyframe || f.FramesLost != 0 || f.Corrupted {
		t.Fatalf("unexpected IDR frame: %+v", f)
	}

	// Next frame after a 2-frame RTP timestamp gap (4500 ticks at 60 fps = 2 lost frames)
	p3 := &rtp.Packet{
		Header:  rtp.Header{SequenceNumber: 105, Timestamp: 94500, Marker: true},
		Payload: []byte{0x41, 0x11, 0x22},
	}
	f2 := d.Push(p3)
	if f2 == nil || f2.Keyframe || f2.FramesLost != 2 || !f2.Corrupted {
		t.Fatalf("unexpected post-gap frame: %+v", f2)
	}
}

func TestDepacketizerTruncatedFUAResetsOnNewTimestamp(t *testing.T) {
	t.Run("with_prior_nalu_in_buffer", func(t *testing.T) {
		d := NewDepacketizer(60)

		// Frame 1: a complete PPS NALU followed by a truncated FU-A IDR start (end lost).
		if f := d.Push(&rtp.Packet{
			Header:  rtp.Header{SequenceNumber: 10, Timestamp: 90000, Marker: false},
			Payload: []byte{0x68, 0xEE, 0x3C, 0x80}, // PPS
		}); f != nil {
			t.Fatalf("expected nil after PPS, got %+v", f)
		}
		if f := d.Push(&rtp.Packet{
			Header:  rtp.Header{SequenceNumber: 11, Timestamp: 90000, Marker: false},
			Payload: []byte{0x7C, 0x85, 0xDE, 0xAD}, // FU-A start, IDR (never completed)
		}); f != nil {
			t.Fatalf("expected nil after truncated FU-A start, got %+v", f)
		}

		// Frame 2 (new timestamp): FU-A non-IDR slice. Pushing start flushes Frame 1 as corrupted.
		flushed := d.Push(&rtp.Packet{
			Header:  rtp.Header{SequenceNumber: 13, Timestamp: 91500, Marker: false},
			Payload: []byte{0x5C, 0x81, 0x11, 0x22}, // FU-A start, non-IDR (nal=0x41)
		})
		if flushed == nil || !flushed.Corrupted || flushed.Timestamp != 90000 {
			t.Fatalf("expected corrupted flush of frame 1, got %+v", flushed)
		}

		// Complete Frame 2 with FU-A end + Marker.
		f2 := d.Push(&rtp.Packet{
			Header:  rtp.Header{SequenceNumber: 14, Timestamp: 91500, Marker: true},
			Payload: []byte{0x5C, 0x41, 0x33, 0x44}, // FU-A end, non-IDR
		})
		if f2 == nil {
			t.Fatalf("expected frame 2 on marker, got nil")
		}
		want := []byte{0x00, 0x00, 0x00, 0x01, 0x41, 0x11, 0x22, 0x33, 0x44}
		if !bytes.Equal(f2.Data, want) {
			t.Fatalf("frame 2 data = %x, want %x (stale FU-A leaked)", f2.Data, want)
		}
		if f2.Keyframe {
			t.Fatalf("frame 2 unexpectedly marked Keyframe")
		}
	})

	t.Run("only_truncated_fua_in_prev_frame", func(t *testing.T) {
		d := NewDepacketizer(60)

		// Frame 1: only a truncated FU-A start (buf stays empty, fragment buffered inside d.pkt).
		if f := d.Push(&rtp.Packet{
			Header:  rtp.Header{SequenceNumber: 20, Timestamp: 90000, Marker: false},
			Payload: []byte{0x7C, 0x85, 0xDE, 0xAD}, // FU-A start, IDR
		}); f != nil {
			t.Fatalf("expected nil after truncated FU-A start, got %+v", f)
		}

		// Frame 2 (new timestamp): FU-A non-IDR slice.
		if f := d.Push(&rtp.Packet{
			Header:  rtp.Header{SequenceNumber: 22, Timestamp: 91500, Marker: false},
			Payload: []byte{0x5C, 0x81, 0xAA, 0xBB}, // FU-A start, non-IDR (nal=0x41)
		}); f != nil {
			t.Fatalf("expected nil before marker, got %+v", f)
		}
		f2 := d.Push(&rtp.Packet{
			Header:  rtp.Header{SequenceNumber: 23, Timestamp: 91500, Marker: true},
			Payload: []byte{0x5C, 0x41, 0xCC, 0xDD}, // FU-A end, non-IDR
		})
		if f2 == nil {
			t.Fatalf("expected frame 2 on marker, got nil")
		}
		want := []byte{0x00, 0x00, 0x00, 0x01, 0x41, 0xAA, 0xBB, 0xCC, 0xDD}
		if !bytes.Equal(f2.Data, want) {
			t.Fatalf("frame 2 data = %x, want %x (stale FU-A leaked)", f2.Data, want)
		}
		if f2.Keyframe {
			t.Fatalf("frame 2 unexpectedly marked Keyframe")
		}
	})
}

func TestDepacketizerDuplicateSequenceIgnored(t *testing.T) {
	d := NewDepacketizer(60)

	p1 := &rtp.Packet{
		Header:  rtp.Header{SequenceNumber: 200, Timestamp: 90000, Marker: false},
		Payload: []byte{0x7C, 0x85, 0xAA, 0xBB}, // FU-A start, IDR
	}
	p2 := &rtp.Packet{
		Header:  rtp.Header{SequenceNumber: 201, Timestamp: 90000, Marker: true},
		Payload: []byte{0x7C, 0x45, 0xCC, 0xDD}, // FU-A end, IDR
	}

	if f := d.Push(p1); f != nil {
		t.Fatalf("expected nil after p1, got %+v", f)
	}
	// Duplicate of p1 should be dropped without corrupting the frame.
	if f := d.Push(p1); f != nil {
		t.Fatalf("expected nil on duplicate p1, got %+v", f)
	}

	f1 := d.Push(p2)
	if f1 == nil {
		t.Fatalf("expected frame 1 on p2, got nil")
	}
	if f1.Corrupted || f1.FramesLost != 0 || !f1.Keyframe {
		t.Fatalf("duplicate p1 caused bad frame 1 flags: %+v", f1)
	}
	want1 := []byte{0x00, 0x00, 0x00, 0x01, 0x65, 0xAA, 0xBB, 0xCC, 0xDD}
	if !bytes.Equal(f1.Data, want1) {
		t.Fatalf("frame 1 data = %x, want %x", f1.Data, want1)
	}

	// Duplicate of marker packet p2 should also be dropped cleanly.
	if f := d.Push(p2); f != nil {
		t.Fatalf("expected nil on duplicate p2, got %+v", f)
	}

	// Next consecutive frame should be completely clean.
	p3 := &rtp.Packet{
		Header:  rtp.Header{SequenceNumber: 202, Timestamp: 91500, Marker: true},
		Payload: []byte{0x41, 0x11, 0x22},
	}
	f2 := d.Push(p3)
	if f2 == nil {
		t.Fatalf("expected frame 2 on p3, got nil")
	}
	if f2.Corrupted || f2.FramesLost != 0 {
		t.Fatalf("duplicate p2 caused bad frame 2 flags: %+v", f2)
	}
	want2 := []byte{0x00, 0x00, 0x00, 0x01, 0x41, 0x11, 0x22}
	if !bytes.Equal(f2.Data, want2) {
		t.Fatalf("frame 2 data = %x, want %x", f2.Data, want2)
	}
}

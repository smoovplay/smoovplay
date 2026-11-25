// Package frametap implements the writer side of SmoovPlay's v1 compressed frame tap
// and companion audio ring (exact binary match with chiaki_tap/frametap.h and framegen/chiaki_tap.py).
package frametap

import (
	"encoding/binary"
	"math"
	"sync"
	"sync/atomic"
)

const (
	DefaultName = `Local\smoovplay-tap`

	Magic      uint32 = 0x50544643 // 'CFTP'
	Version    uint32 = 1
	HeaderSize        = 4096
	SlotHeader        = 64
	SlotCount         = 32
	SlotSize          = 2 * 1024 * 1024
	TotalSize         = HeaderSize + SlotCount*SlotSize

	CodecH264    uint32 = 0
	CodecH265    uint32 = 1
	CodecH265HDR uint32 = 2

	FeaturePress      uint32 = 1
	FeatureAudioDelay uint32 = 2
	FeatureAudioTap   uint32 = 4

	FlagTruncated uint32 = 1

	AudioSuffix     = "-audio"
	AudioMagic      = uint32(0x41544643) // 'CFTA'
	AudioHeaderSize = 64
	AudioSlotHeader = 32
	AudioSlotCount  = 512
	AudioSlotSize   = 8192
	AudioTotalSize  = AudioHeaderSize + AudioSlotCount*AudioSlotSize
)

// Writer writes compressed video frames and optional 16-bit PCM audio chunks into
// SmoovPlay's shared-memory rings. All methods are safe for concurrent use.
type Writer struct {
	name      string
	eventName string

	mu       sync.Mutex
	mem      []byte
	audioMem []byte
	qpcFreq  uint64
	session  uint32
	writeSeq uint64
	audioSeq uint64
	live     bool

	// platform handles (Windows HANDLEs or nil in stub)
	platform platformState
}

// Open creates or opens the named shared-memory mappings and frame event.
func Open(name string) (*Writer, error) {
	if name == "" {
		name = DefaultName
	}
	w := &Writer{
		name:      name,
		eventName: name + "-frame",
	}
	if err := w.openPlatform(); err != nil {
		return nil, err
	}
	return w, nil
}

// Begin starts a new compressed video stream (bumps session counter, resets write_seq).
func (w *Writer) Begin(codec, width, height, maxFPS, features uint32) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if len(w.mem) < TotalSize {
		return
	}
	// Mark not-live while updating header fields.
	binary.LittleEndian.PutUint32(w.mem[0:4], 0)
	if prev := binary.LittleEndian.Uint32(w.mem[44:48]); prev >= w.session {
		w.session = prev
	}
	w.session++
	w.writeSeq = 0

	binary.LittleEndian.PutUint32(w.mem[4:8], Version)
	binary.LittleEndian.PutUint32(w.mem[8:12], SlotCount)
	binary.LittleEndian.PutUint32(w.mem[12:16], SlotSize)
	binary.LittleEndian.PutUint32(w.mem[16:20], codec)
	binary.LittleEndian.PutUint32(w.mem[20:24], width)
	binary.LittleEndian.PutUint32(w.mem[24:28], height)
	binary.LittleEndian.PutUint32(w.mem[28:32], maxFPS)
	binary.LittleEndian.PutUint64(w.mem[32:40], 0) // write_seq
	binary.LittleEndian.PutUint32(w.mem[40:44], 0) // idr_request
	binary.LittleEndian.PutUint32(w.mem[44:48], w.session)
	binary.LittleEndian.PutUint64(w.mem[48:56], w.qpcFreq)
	binary.LittleEndian.PutUint32(w.mem[64:68], features)
	binary.LittleEndian.PutUint32(w.mem[68:72], 0) // clear any stale press_buttons

	// Publish magic last so readers see a consistent header.
	atomic.StoreUint32(u32ptr(w.mem, 0), Magic)
	w.live = true
	w.signalFrame()
}

// Push writes one compressed Annex-B frame into the next ring slot and signals the reader.
func (w *Writer) Push(payload []byte, framesLost int32, recovered bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if !w.live || len(w.mem) < TotalSize || len(payload) == 0 {
		return
	}
	seq := w.writeSeq
	off := HeaderSize + int(seq%SlotCount)*SlotSize
	maxData := SlotSize - SlotHeader
	flags := uint32(0)
	n := len(payload)
	if n > maxData {
		n = maxData
		flags |= FlagTruncated
	}
	// Invalidate slot seq while writing.
	atomic.StoreUint64(u64ptr(w.mem, off), math.MaxUint64)
	binary.LittleEndian.PutUint64(w.mem[off+8:off+16], queryQPC())
	binary.LittleEndian.PutUint32(w.mem[off+16:off+20], uint32(n))
	binary.LittleEndian.PutUint32(w.mem[off+20:off+24], uint32(framesLost))
	var rec uint32
	if recovered {
		rec = 1
	}
	binary.LittleEndian.PutUint32(w.mem[off+24:off+28], rec)
	binary.LittleEndian.PutUint32(w.mem[off+28:off+32], flags)
	copy(w.mem[off+SlotHeader:off+SlotHeader+n], payload[:n])

	// Publish slot seq, then header write_seq, then wake reader.
	atomic.StoreUint64(u64ptr(w.mem, off), seq)
	w.writeSeq = seq + 1
	atomic.StoreUint64(u64ptr(w.mem, 32), w.writeSeq)
	w.signalFrame()
}

// TakeIDRRequest atomically checks and clears SmoovPlay's keyframe request flag.
func (w *Writer) TakeIDRRequest() bool {
	if len(w.mem) < HeaderSize {
		return false
	}
	return atomic.SwapUint32(u32ptr(w.mem, 40), 0) != 0
}

// TakePressRequest atomically checks and clears SmoovPlay's delay-test button press request.
// Returns (buttons, holdMs) where buttons is a ChiakiControllerButton bitmask (0 if none)
// and holdMs is clamped to [20, 2000] ms (matching chiaki_tap/frametap.c).
func (w *Writer) TakePressRequest() (buttons uint32, holdMs uint32) {
	if len(w.mem) < HeaderSize {
		return 0, 0
	}
	buttons = atomic.SwapUint32(u32ptr(w.mem, 68), 0)
	if buttons == 0 {
		return 0, 0
	}
	holdMs = atomic.LoadUint32(u32ptr(w.mem, 72))
	if holdMs < 20 {
		holdMs = 20
	} else if holdMs > 2000 {
		holdMs = 2000
	}
	return buttons, holdMs
}

// PressSent stamps the QPC time (offset 80) when a requested button press was sent to the
// console and increments press_count (offset 76) so SmoovPlay's Delay Test can time it.
func (w *Writer) PressSent() {
	if len(w.mem) < HeaderSize {
		return
	}
	atomic.StoreUint64(u64ptr(w.mem, 80), queryQPC())
	atomic.AddUint32(u32ptr(w.mem, 76), 1)
}

// AudioDelayMs returns SmoovPlay's requested audio holdback (0–500 ms), or 0 if
// SmoovPlay hasn't sent a heartbeat in the last 2 seconds.
func (w *Writer) AudioDelayMs() uint32 {
	if len(w.mem) < HeaderSize {
		return 0
	}
	hb := atomic.LoadUint64(u64ptr(w.mem, 56))
	if hb == 0 || w.qpcFreq == 0 {
		return 0
	}
	now := queryQPC()
	if now > hb && now-hb > w.qpcFreq*2 {
		return 0
	}
	ms := atomic.LoadUint32(u32ptr(w.mem, 88))
	if ms > 500 {
		ms = 500
	}
	return ms
}

// BeginAudio initializes the companion PCM16 audio ring (for SmoovPlay Instant Replay).
func (w *Writer) BeginAudio(rate, channels uint32) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if len(w.audioMem) < AudioTotalSize {
		return
	}
	atomic.StoreUint32(u32ptr(w.audioMem, 0), 0)
	w.audioSeq = 0
	binary.LittleEndian.PutUint32(w.audioMem[4:8], Version)
	binary.LittleEndian.PutUint32(w.audioMem[8:12], AudioSlotCount)
	binary.LittleEndian.PutUint32(w.audioMem[12:16], AudioSlotSize)
	binary.LittleEndian.PutUint32(w.audioMem[16:20], rate)
	binary.LittleEndian.PutUint32(w.audioMem[20:24], channels)
	binary.LittleEndian.PutUint64(w.audioMem[24:32], 0)
	binary.LittleEndian.PutUint64(w.audioMem[32:40], w.qpcFreq)
	atomic.StoreUint32(u32ptr(w.audioMem, 0), AudioMagic)
}

// PushAudio writes an interleaved signed 16-bit PCM chunk into the audio ring.
func (w *Writer) PushAudio(pcm []byte, rate, channels uint32) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if len(w.audioMem) < AudioTotalSize || len(pcm) == 0 {
		return
	}
	maxData := AudioSlotSize - AudioSlotHeader
	for len(pcm) > 0 {
		n := len(pcm)
		if n > maxData {
			n = maxData
		}
		seq := w.audioSeq
		off := AudioHeaderSize + int(seq%AudioSlotCount)*AudioSlotSize
		atomic.StoreUint64(u64ptr(w.audioMem, off), math.MaxUint64)
		binary.LittleEndian.PutUint64(w.audioMem[off+8:off+16], queryQPC())
		binary.LittleEndian.PutUint32(w.audioMem[off+16:off+20], uint32(n))
		binary.LittleEndian.PutUint32(w.audioMem[off+20:off+24], rate)
		binary.LittleEndian.PutUint32(w.audioMem[off+24:off+28], channels)
		copy(w.audioMem[off+AudioSlotHeader:off+AudioSlotHeader+n], pcm[:n])
		atomic.StoreUint64(u64ptr(w.audioMem, off), seq)
		w.audioSeq = seq + 1
		atomic.StoreUint64(u64ptr(w.audioMem, 24), w.audioSeq)
		pcm = pcm[n:]
	}
}

// End marks the video and audio rings as inactive.
func (w *Writer) End() {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.live = false
	if len(w.mem) >= 4 {
		atomic.StoreUint32(u32ptr(w.mem, 0), 0)
	}
	if len(w.audioMem) >= 4 {
		atomic.StoreUint32(u32ptr(w.audioMem, 0), 0)
	}
	w.signalFrame()
}

// Close ends the stream and releases OS shared-memory handles.
func (w *Writer) Close() {
	w.End()
	w.closePlatform()
}

// Package audio decodes Xbox's WebRTC Opus stream in pure Go, feeds SmoovPlay's
// companion AudioTap ring (for Instant Replay), applies SmoovPlay's audio_delay_ms
// holdback so sound stays synced with interpolated frames, and outputs to Windows waveOut.
package audio

import "sync"

const (
	SampleRate = 48000
	Channels   = 2
	MaxDelayMs = 500
)

// DelayLine holds stereo int16 PCM samples in a fixed circular buffer to apply
// dynamic audio holdback (0..500 ms) requested by SmoovPlay's FrameTap.
type DelayLine struct {
	mu      sync.Mutex
	buf     []int16
	write   int
	delayS  int // delay in interleaved int16 samples (delayMs * SampleRate / 1000 * Channels)
}

// NewDelayLine creates a delay buffer supporting up to MaxDelayMs of stereo 48 kHz PCM.
func NewDelayLine() *DelayLine {
	// +20% headroom beyond MaxDelayMs so writes never wrap onto reads.
	capSamples := ((MaxDelayMs + 100) * SampleRate / 1000) * Channels
	return &DelayLine{buf: make([]int16, capSamples)}
}

// SetDelayMs updates the active holdback in milliseconds (clamped to 0..500).
func (d *DelayLine) SetDelayMs(ms uint32) {
	if ms > MaxDelayMs {
		ms = MaxDelayMs
	}
	d.mu.Lock()
	d.delayS = int(ms) * (SampleRate / 1000) * Channels
	d.mu.Unlock()
}

// Process feeds `in` into the delay line and writes the delayed samples into `out`
// (len(out) must equal len(in)). When delay is 0, copies directly without buffering.
func (d *DelayLine) Process(in, out []int16) {
	d.mu.Lock()
	defer d.mu.Unlock()
	n := len(in)
	if d.delayS == 0 {
		copy(out, in)
		// Still advance write pointer with current samples so switching to >0 ms is glitch-free.
		for i := 0; i < n; i++ {
			d.buf[d.write] = in[i]
			d.write++
			if d.write == len(d.buf) {
				d.write = 0
			}
		}
		return
	}
	size := len(d.buf)
	for i := 0; i < n; i++ {
		d.buf[d.write] = in[i]
		read := d.write - d.delayS
		if read < 0 {
			read += size
		}
		out[i] = d.buf[read]
		d.write++
		if d.write == size {
			d.write = 0
		}
	}
}

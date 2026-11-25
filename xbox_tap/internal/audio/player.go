package audio

import (
	"encoding/binary"
	"fmt"

	"github.com/tphakala/go-opus/opus"
)

// Sink plays interleaved 16-bit little-endian stereo PCM at 48 kHz.
type Sink interface {
	WritePCM(pcm []byte) error
	Close() error
}

// Player decodes Opus packets to 48 kHz stereo PCM16, pushes undelayed PCM to an optional
// tap callback (for SmoovPlay's Instant Replay AudioTap), applies DelayLine holdback, and
// writes to the OS audio Sink.
type Player struct {
	dec      *opus.Decoder
	delay    *DelayLine
	sink     Sink
	onRawPCM func([]byte)
	pcmIn    []int16
	pcmOut   []int16
	rawBytes []byte
	outBytes []byte
}

// NewPlayer constructs a 48 kHz stereo Opus player. If sink is nil, OpenDefaultSink is used.
func NewPlayer(sink Sink, onRawPCM func([]byte)) (*Player, error) {
	dec, err := opus.NewDecoder(SampleRate, Channels)
	if err != nil {
		return nil, fmt.Errorf("opus decoder: %w", err)
	}
	if sink == nil {
		sink, err = OpenDefaultSink(SampleRate, Channels)
		if err != nil {
			return nil, fmt.Errorf("audio output: %w", err)
		}
	}
	// Max Opus frame at 48 kHz is 120 ms = 5760 samples/channel.
	maxSamples := 5760 * Channels
	return &Player{
		dec:      dec,
		delay:    NewDelayLine(),
		sink:     sink,
		onRawPCM: onRawPCM,
		pcmIn:    make([]int16, maxSamples),
		pcmOut:   make([]int16, maxSamples),
		rawBytes: make([]byte, maxSamples*2),
		outBytes: make([]byte, maxSamples*2),
	}, nil
}

// SetDelayMs updates SmoovPlay's audio holdback (0..500 ms).
func (p *Player) SetDelayMs(ms uint32) {
	p.delay.SetDelayMs(ms)
}

// PushOpus decodes one Opus RTP payload and plays it.
func (p *Player) PushOpus(payload []byte) error {
	if len(payload) == 0 {
		return nil
	}
	nPerCh, err := p.dec.Decode(payload, p.pcmIn)
	if err != nil || nPerCh <= 0 {
		return err
	}
	total := nPerCh * Channels
	inSlice := p.pcmIn[:total]
	outSlice := p.pcmOut[:total]

	if p.onRawPCM != nil {
		int16ToLE(inSlice, p.rawBytes[:total*2])
		p.onRawPCM(p.rawBytes[:total*2])
	}

	p.delay.Process(inSlice, outSlice)
	int16ToLE(outSlice, p.outBytes[:total*2])
	return p.sink.WritePCM(p.outBytes[:total*2])
}

// Close closes the underlying audio output sink.
func (p *Player) Close() error {
	if p.sink != nil {
		return p.sink.Close()
	}
	return nil
}

func int16ToLE(src []int16, dst []byte) {
	for i, s := range src {
		binary.LittleEndian.PutUint16(dst[i*2:i*2+2], uint16(s))
	}
}

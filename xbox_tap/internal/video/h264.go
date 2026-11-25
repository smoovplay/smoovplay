// Package video implements a zero-jitter-buffer RFC 6184 H.264 RTP depacketizer
// and an H.264 SPS dimension parser so exact stream width/height (e.g. 1080p, 1440p)
// are read directly from the bitstream with zero CPU decoding.
package video

import (
	"bytes"

	"github.com/pion/rtp"
	"github.com/pion/rtp/codecs"
)

var annexBStart = []byte{0x00, 0x00, 0x00, 0x01}

// Frame is one complete Annex-B H.264 access unit emitted immediately when RTP M=1 arrives.
type Frame struct {
	Data       []byte
	Timestamp  uint32
	Keyframe   bool
	FramesLost int32
	Corrupted  bool
	Width      int // >0 if an SPS was parsed in this access unit
	Height     int
}

// Depacketizer reassembles H.264 RTP packets into Annex-B frames with zero buffering delay.
type Depacketizer struct {
	pkt         codecs.H264Packet
	buf         bytes.Buffer
	out         []byte // reused backing slice for emitted Frame.Data
	curTS       uint32
	hasTS       bool
	hasSeq      bool
	lastSeq     uint16
	lastEmitTS  uint32
	hasEmitTS   bool
	corrupted   bool
	hasKeyframe bool
	spsWidth    int
	spsHeight   int
	lastSPS     []byte
	fps         int
}

// NewDepacketizer creates a zero-jitter-buffer H.264 depacketizer.
func NewDepacketizer(fps int) *Depacketizer {
	if fps <= 0 {
		fps = 60
	}
	return &Depacketizer{fps: fps}
}

// Push feeds one RTP packet. Returns a completed Frame as soon as the RTP marker bit
// (or a timestamp change) completes an access unit, or nil if more fragments are coming.
// The returned Frame.Data aliases an internal buffer reused across frames; callers must
// consume or copy it before calling Push again.
func (d *Depacketizer) Push(p *rtp.Packet) *Frame {
	if p == nil || len(p.Payload) == 0 {
		return nil
	}

	// Drop exact duplicate RTP packets before they cause false corruption or duplicate NALU bytes.
	if d.hasSeq && p.SequenceNumber == d.lastSeq {
		return nil
	}

	var emitted *Frame
	// If timestamp changed before we saw a marker bit (lost last packet of previous frame),
	// flush the previous access unit marked corrupted and clear any stale incomplete FU-A state.
	if d.hasTS && p.Timestamp != d.curTS {
		if d.buf.Len() > 0 {
			d.corrupted = true
			emitted = d.flush(d.curTS)
		} else {
			d.pkt = codecs.H264Packet{}
		}
	}

	if d.hasSeq {
		expected := d.lastSeq + 1
		if p.SequenceNumber != expected {
			d.corrupted = true
		}
	}
	d.lastSeq = p.SequenceNumber
	d.hasSeq = true
	d.curTS = p.Timestamp
	d.hasTS = true

	nalu, err := d.pkt.Unmarshal(p.Payload)
	if err != nil {
		d.corrupted = true
	} else if len(nalu) > 0 {
		d.appendNALUs(nalu)
	}

	if p.Marker && d.buf.Len() > 0 {
		f := d.flush(p.Timestamp)
		if emitted == nil {
			emitted = f
		} else {
			// Rare: both flushed previous truncated frame and completed a 1-packet frame.
			// Prefer returning the newest complete frame and count 1 extra lost frame.
			f.FramesLost++
			emitted = f
		}
	}
	return emitted
}

func (d *Depacketizer) appendNALUs(data []byte) {
	// pion's H264Packet.Unmarshal already emits Annex-B start codes (00 00 00 01) for single NALUs,
	// STAP-A, and completed FU-A packets. Scan NAL units to detect keyframes and SPS dimensions.
	d.buf.Write(data)
	scanAnnexBNALUs(data, func(nal []byte) {
		if len(nal) == 0 {
			return
		}
		nalType := nal[0] & 0x1F
		if nalType == 5 {
			d.hasKeyframe = true
		} else if nalType == 7 {
			if !bytes.Equal(nal, d.lastSPS) {
				if w, h, ok := ParseSPSDimensions(nal); ok && w > 0 && h > 0 {
					d.spsWidth, d.spsHeight = w, h
					d.lastSPS = append(d.lastSPS[:0], nal...)
				}
			}
		}
	})
}

func (d *Depacketizer) flush(ts uint32) *Frame {
	d.out = append(d.out[:0], d.buf.Bytes()...)
	d.buf.Reset()
	d.pkt = codecs.H264Packet{}

	lost := d.inferLost(ts)
	if d.corrupted && lost == 0 {
		lost = 1
	}
	f := &Frame{
		Data:       d.out,
		Timestamp:  ts,
		Keyframe:   d.hasKeyframe,
		FramesLost: lost,
		Corrupted:  d.corrupted,
		Width:      d.spsWidth,
		Height:     d.spsHeight,
	}
	d.lastEmitTS = ts
	d.hasEmitTS = true
	d.corrupted = false
	d.hasKeyframe = false
	return f
}

func (d *Depacketizer) inferLost(ts uint32) int32 {
	if !d.hasEmitTS || ts == 0 || d.lastEmitTS == 0 || d.fps <= 0 {
		return 0
	}
	delta := ts - d.lastEmitTS // uint32 wrap-safe
	if delta == 0 || delta > 90000*5 {
		return 0
	}
	step := 90000.0 / float64(d.fps)
	if float64(delta) <= step*1.6 {
		return 0
	}
	missed := int32(float64(delta)/step+0.5) - 1
	if missed < 0 {
		return 0
	}
	return missed
}

func scanAnnexBNALUs(data []byte, fn func(nal []byte)) {
	n := len(data)
	i := 0
	for i < n {
		// Find 00 00 01 or 00 00 00 01
		start := -1
		scLen := 0
		for j := i; j+2 < n; j++ {
			if data[j] == 0 && data[j+1] == 0 {
				if data[j+2] == 1 {
					start = j
					scLen = 3
					break
				}
				if j+3 < n && data[j+2] == 0 && data[j+3] == 1 {
					start = j
					scLen = 4
					break
				}
			}
		}
		if start < 0 {
			return
		}
		nalStart := start + scLen
		// Find next start code
		nalEnd := n
		for j := nalStart; j+2 < n; j++ {
			if data[j] == 0 && data[j+1] == 0 && (data[j+2] == 1 || (j+3 < n && data[j+2] == 0 && data[j+3] == 1)) {
				nalEnd = j
				break
			}
		}
		if nalStart < nalEnd {
			fn(data[nalStart:nalEnd])
		}
		i = nalEnd
	}
}

// ParseSPSDimensions parses the coded (width, height) from a raw H.264 SPS NAL unit
// (starting at the NAL header byte 0x67/0x27, without Annex-B start code).
func ParseSPSDimensions(nal []byte) (int, int, bool) {
	if len(nal) < 4 || (nal[0]&0x1F) != 7 {
		return 0, 0, false
	}
	rbsp := removeEmulationPrevention(nal[1:])
	br := &bitReader{data: rbsp}

	profileIDC, ok := br.readBits(8)
	if !ok {
		return 0, 0, false
	}
	if _, ok = br.readBits(8); !ok { // constraint flags + reserved
		return 0, 0, false
	}
	if _, ok = br.readBits(8); !ok { // level_idc
		return 0, 0, false
	}
	if _, ok = br.readUE(); !ok { // seq_parameter_set_id
		return 0, 0, false
	}

	chromaFormatIDC := uint32(1)
	if isHighProfile(profileIDC) {
		chromaFormatIDC, ok = br.readUE()
		if !ok {
			return 0, 0, false
		}
		if chromaFormatIDC == 3 {
			if _, ok = br.readBits(1); !ok { // separate_colour_plane_flag
				return 0, 0, false
			}
		}
		if _, ok = br.readUE(); !ok { // bit_depth_luma_minus8
			return 0, 0, false
		}
		if _, ok = br.readUE(); !ok { // bit_depth_chroma_minus8
			return 0, 0, false
		}
		if _, ok = br.readBits(1); !ok { // qpprime_y_zero_transform_bypass_flag
			return 0, 0, false
		}
		seqScalingPresent, ok2 := br.readBits(1)
		if !ok2 {
			return 0, 0, false
		}
		if seqScalingPresent == 1 {
			count := 8
			if chromaFormatIDC == 3 {
				count = 12
			}
			for i := 0; i < count; i++ {
				present, ok3 := br.readBits(1)
				if !ok3 {
					return 0, 0, false
				}
				if present == 1 {
					size := 16
					if i >= 6 {
						size = 64
					}
					if !skipScalingList(br, size) {
						return 0, 0, false
					}
				}
			}
		}
	}

	if _, ok = br.readUE(); !ok { // log2_max_frame_num_minus4
		return 0, 0, false
	}
	picOrderCntType, ok := br.readUE()
	if !ok {
		return 0, 0, false
	}
	if picOrderCntType == 0 {
		if _, ok = br.readUE(); !ok { // log2_max_pic_order_cnt_lsb_minus4
			return 0, 0, false
		}
	} else if picOrderCntType == 1 {
		if _, ok = br.readBits(1); !ok {
			return 0, 0, false
		}
		if _, ok = br.readSE(); !ok {
			return 0, 0, false
		}
		if _, ok = br.readSE(); !ok {
			return 0, 0, false
		}
		nRef, ok2 := br.readUE()
		if !ok2 || nRef > 256 {
			return 0, 0, false
		}
		for i := uint32(0); i < nRef; i++ {
			if _, ok = br.readSE(); !ok {
				return 0, 0, false
			}
		}
	}

	if _, ok = br.readUE(); !ok { // max_num_ref_frames
		return 0, 0, false
	}
	if _, ok = br.readBits(1); !ok { // gaps_in_frame_num_value_allowed_flag
		return 0, 0, false
	}
	picWidthInMbsMinus1, ok := br.readUE()
	if !ok {
		return 0, 0, false
	}
	picHeightInMapUnitsMinus1, ok := br.readUE()
	if !ok {
		return 0, 0, false
	}
	frameMbsOnlyFlag, ok := br.readBits(1)
	if !ok {
		return 0, 0, false
	}
	if frameMbsOnlyFlag == 0 {
		if _, ok = br.readBits(1); !ok { // mb_adaptive_frame_field_flag
			return 0, 0, false
		}
	}
	if _, ok = br.readBits(1); !ok { // direct_8x8_inference_flag
		return 0, 0, false
	}
	frameCroppingFlag, ok := br.readBits(1)
	if !ok {
		return 0, 0, false
	}
	var cropLeft, cropRight, cropTop, cropBottom uint32
	if frameCroppingFlag == 1 {
		if cropLeft, ok = br.readUE(); !ok {
			return 0, 0, false
		}
		if cropRight, ok = br.readUE(); !ok {
			return 0, 0, false
		}
		if cropTop, ok = br.readUE(); !ok {
			return 0, 0, false
		}
		if cropBottom, ok = br.readUE(); !ok {
			return 0, 0, false
		}
	}

	cropUnitX := uint32(1)
	cropUnitY := 2 - frameMbsOnlyFlag
	if chromaFormatIDC == 1 || chromaFormatIDC == 2 {
		cropUnitX = 2
	}
	if chromaFormatIDC == 1 {
		cropUnitY = 2 * (2 - frameMbsOnlyFlag)
	}

	width := int((picWidthInMbsMinus1+1)*16) - int((cropLeft+cropRight)*cropUnitX)
	height := int((2-frameMbsOnlyFlag)*(picHeightInMapUnitsMinus1+1)*16) - int((cropTop+cropBottom)*cropUnitY)
	if width <= 0 || height <= 0 {
		return 0, 0, false
	}
	return width, height, true
}

func isHighProfile(p uint32) bool {
	switch p {
	case 100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135:
		return true
	default:
		return false
	}
}

func removeEmulationPrevention(in []byte) []byte {
	out := make([]byte, 0, len(in))
	zeros := 0
	for _, b := range in {
		if zeros == 2 && b == 0x03 {
			zeros = 0
			continue
		}
		out = append(out, b)
		if b == 0 {
			zeros++
		} else {
			zeros = 0
		}
	}
	return out
}

func skipScalingList(br *bitReader, size int) bool {
	lastScale, nextScale := int32(8), int32(8)
	for j := 0; j < size; j++ {
		if nextScale != 0 {
			delta, ok := br.readSE()
			if !ok {
				return false
			}
			nextScale = (lastScale + delta + 256) % 256
		}
		if nextScale != 0 {
			lastScale = nextScale
		}
	}
	return true
}

type bitReader struct {
	data []byte
	pos  int
}

func (b *bitReader) readBits(n int) (uint32, bool) {
	if b.pos+n > len(b.data)*8 {
		return 0, false
	}
	var v uint32
	for i := 0; i < n; i++ {
		byteIdx := b.pos >> 3
		bitIdx := 7 - (b.pos & 7)
		v = (v << 1) | uint32((b.data[byteIdx]>>bitIdx)&1)
		b.pos++
	}
	return v, true
}

func (b *bitReader) readUE() (uint32, bool) {
	zeros := 0
	for {
		bit, ok := b.readBits(1)
		if !ok {
			return 0, false
		}
		if bit == 1 {
			break
		}
		zeros++
		if zeros > 31 {
			return 0, false
		}
	}
	if zeros == 0 {
		return 0, true
	}
	rest, ok := b.readBits(zeros)
	if !ok {
		return 0, false
	}
	return ((1 << zeros) - 1) + rest, true
}

func (b *bitReader) readSE() (int32, bool) {
	u, ok := b.readUE()
	if !ok {
		return 0, false
	}
	if u&1 == 1 {
		return int32((u + 1) >> 1), true
	}
	return -int32(u >> 1), true
}

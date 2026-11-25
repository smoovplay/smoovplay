// Package input implements Xbox's binary WebRTC `input` DataChannel protocol
// (gamepad reports, client metadata, vibration parsing) and native Windows XInput polling.
package input

import (
	"encoding/binary"
	"math"
)

// ReportType bitmask (little-endian uint16 at packet offset 0).
const (
	ReportMetadata       uint16 = 1 << 0 // 1
	ReportGamepad        uint16 = 1 << 1 // 2
	ReportPointer        uint16 = 1 << 2 // 4
	ReportClientMetadata uint16 = 1 << 3 // 8
	ReportServerMetadata uint16 = 1 << 4 // 16
	ReportVibration      uint16 = 1 << 7 // 128
)

// Xbox binary input channel button bits (uint16 LE).
const (
	BtnNexus         uint16 = 1 << 0
	BtnMenu          uint16 = 1 << 1
	BtnView          uint16 = 1 << 2
	BtnA             uint16 = 1 << 3
	BtnB             uint16 = 1 << 4
	BtnX             uint16 = 1 << 5
	BtnY             uint16 = 1 << 6
	BtnDPadUp        uint16 = 1 << 7
	BtnDPadDown      uint16 = 1 << 8
	BtnDPadLeft      uint16 = 1 << 9
	BtnDPadRight     uint16 = 1 << 10
	BtnLeftShoulder  uint16 = 1 << 11
	BtnRightShoulder uint16 = 1 << 12
	BtnLeftThumb     uint16 = 1 << 13
	BtnRightThumb    uint16 = 1 << 14
)

// GamepadState represents a single controller's current inputs in Xbox wire format.
type GamepadState struct {
	Index        uint8
	Connected    bool
	Buttons      uint16
	LeftThumbX   int16
	LeftThumbY   int16 // Xbox wire convention: up is negative (inverted vs XInput)
	RightThumbX  int16
	RightThumbY  int16 // Xbox wire convention: up is negative (inverted vs XInput)
	LeftTrigger  uint16
	RightTrigger uint16
}

// VibrationReport is a rumble command received from the Xbox over the `input` channel.
type VibrationReport struct {
	GamepadIndex uint8
	LeftMotor    float32 // 0.0 - 1.0
	RightMotor   float32 // 0.0 - 1.0
	LeftTrigger  float32 // 0.0 - 1.0
	RightTrigger float32 // 0.0 - 1.0
	DurationMs   uint16
	DelayMs      uint16
	RepeatCount  uint8
}

const headerSize = 14

func writeHeader(buf []byte, reportType uint16, seq uint32, timestampMs float64) {
	binary.LittleEndian.PutUint16(buf[0:2], reportType)
	binary.LittleEndian.PutUint32(buf[2:6], seq)
	binary.LittleEndian.PutUint64(buf[6:14], math.Float64bits(timestampMs))
}

// EncodeClientMetadata builds the 15-byte initial ClientMetadata packet sent on `input` open.
func EncodeClientMetadata(seq uint32, timestampMs float64, maxTouchPoints uint8) []byte {
	buf := make([]byte, headerSize+1)
	writeHeader(buf, ReportClientMetadata, seq, timestampMs)
	buf[headerSize] = maxTouchPoints
	return buf
}

// MetadataFrame reports video frame arrival/render timing back to the Xbox on dcInput (ReportMetadata).
type MetadataFrame struct {
	ServerDataKey uint32 // RTP timestamp of the video frame
	ArrivalMs     uint32 // ms since session start
	SubmittedMs   uint32
	DecodedMs     uint32
	RenderedMs    uint32
	PacketTimeMs  uint32
	NowMs         uint32
}

// EncodeMetadataPacket builds a binary ReportMetadata packet (header + 1 + 28*N bytes).
func EncodeMetadataPacket(seq uint32, timestampMs float64, frames []MetadataFrame) []byte {
	n := len(frames)
	if n > 255 {
		n = 255
	}
	buf := make([]byte, headerSize+1+n*28)
	writeHeader(buf, ReportMetadata, seq, timestampMs)
	buf[headerSize] = uint8(n)
	off := headerSize + 1
	for i := 0; i < n; i++ {
		f := frames[i]
		binary.LittleEndian.PutUint32(buf[off:off+4], f.ServerDataKey)
		binary.LittleEndian.PutUint32(buf[off+4:off+8], f.ArrivalMs)
		binary.LittleEndian.PutUint32(buf[off+8:off+12], f.SubmittedMs)
		binary.LittleEndian.PutUint32(buf[off+12:off+16], f.DecodedMs)
		binary.LittleEndian.PutUint32(buf[off+16:off+20], f.RenderedMs)
		binary.LittleEndian.PutUint32(buf[off+20:off+24], f.PacketTimeMs)
		binary.LittleEndian.PutUint32(buf[off+24:off+28], f.NowMs)
		off += 28
	}
	return buf
}

// EncodeGamepadPacket builds a binary gamepad report packet (header + 1 + 23*N bytes).
func EncodeGamepadPacket(seq uint32, timestampMs float64, frames []GamepadState) []byte {
	if len(frames) == 0 {
		buf := make([]byte, headerSize)
		writeHeader(buf, 0, seq, timestampMs)
		return buf
	}
	n := len(frames)
	if n > 255 {
		n = 255
	}
	buf := make([]byte, headerSize+1+n*23)
	writeHeader(buf, ReportGamepad, seq, timestampMs)
	buf[headerSize] = uint8(n)
	off := headerSize + 1
	for i := 0; i < n; i++ {
		f := frames[i]
		buf[off] = f.Index
		off++
		binary.LittleEndian.PutUint16(buf[off:off+2], f.Buttons)
		off += 2
		binary.LittleEndian.PutUint16(buf[off:off+2], uint16(f.LeftThumbX))
		off += 2
		binary.LittleEndian.PutUint16(buf[off:off+2], uint16(f.LeftThumbY))
		off += 2
		binary.LittleEndian.PutUint16(buf[off:off+2], uint16(f.RightThumbX))
		off += 2
		binary.LittleEndian.PutUint16(buf[off:off+2], uint16(f.RightThumbY))
		off += 2
		binary.LittleEndian.PutUint16(buf[off:off+2], f.LeftTrigger)
		off += 2
		binary.LittleEndian.PutUint16(buf[off:off+2], f.RightTrigger)
		off += 2
		// PhysicalPhysicality (uint32 LE = 1) + VirtualPhysicality (uint32 BE = 1)
		binary.LittleEndian.PutUint32(buf[off:off+4], 1)
		off += 4
		binary.BigEndian.PutUint32(buf[off:off+4], 1)
		off += 4
	}
	return buf
}

// ParseVibration parses an inbound vibration packet from the Xbox.
func ParseVibration(data []byte) (VibrationReport, bool) {
	if len(data) < 13 || data[0] != byte(ReportVibration) {
		return VibrationReport{}, false
	}
	return VibrationReport{
		GamepadIndex: data[3],
		LeftMotor:    float32(data[4]) / 100.0,
		RightMotor:   float32(data[5]) / 100.0,
		LeftTrigger:  float32(data[6]) / 100.0,
		RightTrigger: float32(data[7]) / 100.0,
		DurationMs:   binary.LittleEndian.Uint16(data[8:10]),
		DelayMs:      binary.LittleEndian.Uint16(data[10:12]),
		RepeatCount:  data[12],
	}, true
}

// ParseServerMetadata parses video dimensions (width, height) from an inbound serverMetadata packet.
func ParseServerMetadata(data []byte) (width, height uint32, ok bool) {
	if len(data) < 10 || binary.LittleEndian.Uint16(data[0:2]) != ReportServerMetadata {
		return 0, 0, false
	}
	height = binary.LittleEndian.Uint32(data[2:6])
	width = binary.LittleEndian.Uint32(data[6:10])
	return width, height, width > 0 && height > 0
}

// MapChiakiButtons translates a SmoovPlay/ChiakiControllerButton bitmask (used by
// FEATURE_PRESS / Delay Test, e.g. BUTTON_OPTIONS = 1<<12 -> BtnMenu) into Xbox wire button bits.
func MapChiakiButtons(chiakiMask uint32) uint16 {
	var out uint16
	if chiakiMask&(1<<0) != 0 {
		out |= BtnA // Cross
	}
	if chiakiMask&(1<<1) != 0 {
		out |= BtnB // Circle
	}
	if chiakiMask&(1<<2) != 0 {
		out |= BtnX // Square
	}
	if chiakiMask&(1<<3) != 0 {
		out |= BtnY // Triangle
	}
	if chiakiMask&(1<<4) != 0 {
		out |= BtnDPadLeft
	}
	if chiakiMask&(1<<5) != 0 {
		out |= BtnDPadRight
	}
	if chiakiMask&(1<<6) != 0 {
		out |= BtnDPadUp
	}
	if chiakiMask&(1<<7) != 0 {
		out |= BtnDPadDown
	}
	if chiakiMask&(1<<8) != 0 {
		out |= BtnLeftShoulder // L1
	}
	if chiakiMask&(1<<9) != 0 {
		out |= BtnRightShoulder // R1
	}
	if chiakiMask&(1<<10) != 0 {
		out |= BtnLeftThumb // L3
	}
	if chiakiMask&(1<<11) != 0 {
		out |= BtnRightThumb // R3
	}
	if chiakiMask&(1<<12) != 0 {
		out |= BtnMenu // Options (SmoovPlay Delay Test default)
	}
	if chiakiMask&(1<<13) != 0 {
		out |= BtnView // Share
	}
	if chiakiMask&(1<<15) != 0 {
		out |= BtnNexus // PS / Guide
	}
	return out
}

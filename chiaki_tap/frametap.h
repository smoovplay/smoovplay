// Frame tap for chiaki-ng: copies every compressed video frame the PS5 sends into a shared-memory
// ring, so another process (SmoovPlay's ChiakiSource) can decode and interpolate it. A second, smaller ring
// carries the game's sound as it arrives (16-bit PCM, before chiaki-ng's own volume), for SmoovPlay's instant replay.
//
// Enabled only when the environment variable CHIAKI_FRAMETAP=1 is set; otherwise every call is a no-op.
// CHIAKI_FRAMETAP_NAME, if set, replaces the shared memory's name (tests run beside a real stream that way).
// Windows (named file mapping + named event) and Linux (POSIX shared memory /dev/shm/<name after the last
// backslash> + a futex on write_seq's low word); a no-op elsewhere.
//
// Layout (all little-endian, must match framegen/chiaki_tap.py):
//   header, 4096 bytes:
//     0  u32 magic 'CFTP'      4  u32 version          8  u32 slot_count     12 u32 slot_size
//     16 u32 codec (0 H.264, 1 H.265, 2 H.265 HDR)     20 u32 width          24 u32 height
//     28 u32 max_fps           32 u64 write_seq (frames written so far; bumped after the slot is complete)
//     40 u32 idr_request (reader sets 1, writer clears it and asks the PS5 for a keyframe)
//     44 u32 session (bumped at every stream start)    48 u64 qpc_freq
//     56 u64 reader_heartbeat_qpc (reader writes it; informational)
//     64 u32 features (writer; bit 0: it presses buttons when asked, for SmoovPlay's delay test)
//     68 u32 press_buttons (reader sets a ChiakiControllerButton mask; writer clears it and presses them)
//     72 u32 press_hold_ms (reader, before press_buttons)   76 u32 press_count (writer: presses sent so far)
//     80 u64 press_qpc (writer: when the latest press went to the PS5; written before press_count goes up)
//     88 u32 audio_delay_ms (reader: hold chiaki-ng's sound back this long, to meet SmoovPlay's picture; ignored
//        once the reader's heartbeat is over 2 s old, so sound comes back on time if SmoovPlay stops)
//   features bit 1: it honours audio_delay_ms; bit 2: it writes the sound ring below.
//   slot i at 4096 + i * slot_size:
//     0  u64 seq   8  u64 qpc_arrival   16 u32 size   20 i32 frames_lost   24 u32 recovered
//     28 u32 flags (bit 0: frame was larger than the slot and got truncated)   64 data[size]
//
// Sound ring: its own mapping, named like the frames' with "-audio" added, so a reader or writer that predates it
// is unaffected. Same scheme as the frames: a slot's seq is set to all ones while it's rewritten.
//   header, 64 bytes:
//     0  u32 magic 'CFTA' (0 while no stream)   4  u32 version   8  u32 slot_count   12 u32 slot_size
//     16 u32 rate (Hz)   20 u32 channels   24 u64 write_seq (chunks written so far)   32 u64 qpc_freq
//   slot i at 64 + i * slot_size:
//     0  u64 seq   8  u64 qpc_arrival (of the chunk's first sample)   16 u32 size (bytes)   20 u32 rate
//     24 u32 channels   32 data[size]: interleaved signed 16-bit samples
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FRAMETAP_NAME "Local\\chiaki-frametap"
#define FRAMETAP_EVENT_NAME "Local\\chiaki-frametap-frame"
#define FRAMETAP_MAGIC 0x50544643u /* 'CFTP' */
#define FRAMETAP_VERSION 1u
#define FRAMETAP_HEADER_SIZE 4096u
#define FRAMETAP_SLOT_HEADER 64u
#define FRAMETAP_SLOT_COUNT 32u
#define FRAMETAP_SLOT_SIZE (2u * 1024u * 1024u)
#define FRAMETAP_FEATURE_PRESS 1u
#define FRAMETAP_FEATURE_AUDIO_DELAY 2u
#define FRAMETAP_FEATURE_AUDIO_TAP 4u
#define FRAMETAP_AUDIO_SUFFIX "-audio"
#define FRAMETAP_AUDIO_MAGIC 0x41544643u /* 'CFTA' */
#define FRAMETAP_AUDIO_HEADER 64u
#define FRAMETAP_AUDIO_SLOT_HEADER 32u
#define FRAMETAP_AUDIO_SLOT_COUNT 512u
#define FRAMETAP_AUDIO_SLOT_SIZE 8192u

// True if CHIAKI_FRAMETAP=1 is set in the environment.
bool frametap_enabled(void);

// Call when a stream session starts. Creates the shared memory on first use. Returns false if disabled or on error.
bool frametap_begin(uint32_t codec, uint32_t width, uint32_t height, uint32_t max_fps);

// Call for every compressed video frame, before handing it to the decoder. Never blocks.
void frametap_push(const uint8_t *buf, size_t size, int32_t frames_lost, bool recovered);

// Returns true once per reader request for a keyframe (the caller then asks the PS5 for one).
bool frametap_take_idr_request(void);

// The buttons the reader asked to press (a ChiakiControllerButton mask), once per request, or 0; *hold_ms says how
// long to hold them. The caller presses them, sends the controller state, then calls frametap_press_sent().
uint32_t frametap_take_press_request(uint32_t *hold_ms);

// Call right after a press went to the PS5: the reader times the PS5's response from here.
void frametap_press_sent(void);

// How long to hold the sound back (0-500 ms), as the reader asked; 0 if it hasn't, or has stopped.
uint32_t frametap_audio_delay_ms(void);

// The sound's format, when chiaki-ng sets up its sound output (sample rate, channels of 16-bit samples).
void frametap_audio_format(uint32_t rate, uint32_t channels);

// Call for every chunk of decoded sound as it arrives (interleaved signed 16-bit). Never blocks.
void frametap_push_audio(const uint8_t *buf, size_t size);

// Call when the stream session ends.
void frametap_end(void);

#ifdef __cplusplus
}
#endif

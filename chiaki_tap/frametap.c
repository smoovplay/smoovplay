// See frametap.h.
#include "frametap.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32) || defined(__linux__)
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include "frametap_posix.h" // the Win32 calls below, on Linux
#endif

#define TOTAL_SIZE ((size_t)FRAMETAP_HEADER_SIZE + (size_t)FRAMETAP_SLOT_COUNT * FRAMETAP_SLOT_SIZE)
#define AUDIO_TOTAL_SIZE ((size_t)FRAMETAP_AUDIO_HEADER + (size_t)FRAMETAP_AUDIO_SLOT_COUNT * FRAMETAP_AUDIO_SLOT_SIZE)

static HANDLE g_map = NULL;
static HANDLE g_event = NULL;
static uint8_t *g_base = NULL;
static HANDLE g_amap = NULL;
static uint8_t *g_abase = NULL; // the sound ring; NULL if it couldn't be made (the frames still flow)
static uint32_t g_arate = 48000, g_achannels = 2;
static int g_enabled = -1;

#define H32(off) ((volatile uint32_t *)(g_base + (off)))
#define H64(off) ((volatile uint64_t *)(g_base + (off)))
#define A32(off) ((volatile uint32_t *)(g_abase + (off)))
#define A64(off) ((volatile uint64_t *)(g_abase + (off)))

bool frametap_enabled(void)
{
	if(g_enabled < 0)
	{
		const char *v = getenv("CHIAKI_FRAMETAP");
		g_enabled = (v && v[0] == '1') ? 1 : 0;
	}
	return g_enabled == 1;
}

static bool open_shared(void)
{
	if(g_base)
		return true;
	const char *name = getenv("CHIAKI_FRAMETAP_NAME");
	if(!name || !name[0] || strlen(name) > 200)
		name = FRAMETAP_NAME;
	char event_name[256];
	snprintf(event_name, sizeof(event_name), "%s-frame", name);
	// Either side may create the mapping first; CreateFileMapping opens the existing one if it's there.
	g_map = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE,
		(DWORD)((uint64_t)TOTAL_SIZE >> 32), (DWORD)(TOTAL_SIZE & 0xffffffffu), name);
	if(!g_map)
		return false;
	g_base = (uint8_t *)MapViewOfFile(g_map, FILE_MAP_ALL_ACCESS, 0, 0, TOTAL_SIZE);
	if(!g_base)
	{
		CloseHandle(g_map);
		g_map = NULL;
		return false;
	}
	g_event = CreateEventA(NULL, FALSE, FALSE, event_name); // auto-reset, one reader

	char audio_name[256];
	snprintf(audio_name, sizeof(audio_name), "%s" FRAMETAP_AUDIO_SUFFIX, name);
	g_amap = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, (DWORD)AUDIO_TOTAL_SIZE, audio_name);
	if(g_amap)
	{
		g_abase = (uint8_t *)MapViewOfFile(g_amap, FILE_MAP_ALL_ACCESS, 0, 0, AUDIO_TOTAL_SIZE);
		if(!g_abase)
		{
			CloseHandle(g_amap);
			g_amap = NULL;
		}
	}
	return true;
}

static void audio_header(uint64_t qpc_freq)
{
	if(!g_abase)
		return;
	InterlockedExchange((volatile LONG *)A32(0), 0);
	*A32(4) = 1;
	*A32(8) = FRAMETAP_AUDIO_SLOT_COUNT;
	*A32(12) = FRAMETAP_AUDIO_SLOT_SIZE;
	*A32(16) = g_arate;
	*A32(20) = g_achannels;
	*A64(32) = qpc_freq;
	InterlockedExchange64((volatile LONG64 *)A64(24), 0);
	InterlockedExchange((volatile LONG *)A32(0), (LONG)FRAMETAP_AUDIO_MAGIC);
}

bool frametap_begin(uint32_t codec, uint32_t width, uint32_t height, uint32_t max_fps)
{
	if(!frametap_enabled() || !open_shared())
		return false;
	InterlockedExchange((volatile LONG *)H32(0), 0); // header is being rewritten
	LARGE_INTEGER f;
	QueryPerformanceFrequency(&f);
	*H32(4) = FRAMETAP_VERSION;
	*H32(8) = FRAMETAP_SLOT_COUNT;
	*H32(12) = FRAMETAP_SLOT_SIZE;
	*H32(16) = codec;
	*H32(20) = width;
	*H32(24) = height;
	*H32(28) = max_fps;
	*H64(48) = (uint64_t)f.QuadPart;
	*H32(40) = 0;
	*H32(64) = FRAMETAP_FEATURE_PRESS | FRAMETAP_FEATURE_AUDIO_DELAY | (g_abase ? FRAMETAP_FEATURE_AUDIO_TAP : 0u);
	*H32(68) = 0;
	audio_header((uint64_t)f.QuadPart);
	// The reader treats a new session as a new stream: it resets its decoder and waits for a keyframe.
	InterlockedExchange64((volatile LONG64 *)H64(32), 0);
	InterlockedIncrement((volatile LONG *)H32(44));
	InterlockedExchange((volatile LONG *)H32(0), (LONG)FRAMETAP_MAGIC); // last: header is complete
	return true;
}

void frametap_push(const uint8_t *buf, size_t size, int32_t frames_lost, bool recovered)
{
	if(!g_base || *H32(0) != FRAMETAP_MAGIC)
		return;
	LARGE_INTEGER now;
	QueryPerformanceCounter(&now);
	uint64_t seq = *H64(32);
	uint8_t *slot = g_base + FRAMETAP_HEADER_SIZE + (size_t)(seq % FRAMETAP_SLOT_COUNT) * FRAMETAP_SLOT_SIZE;
	size_t cap = FRAMETAP_SLOT_SIZE - FRAMETAP_SLOT_HEADER;
	uint32_t flags = 0;
	if(size > cap)
	{
		size = cap;
		flags |= 1;
	}
	// Mark the slot as being rewritten first, so a reader copying it sees the sequence change and retries.
	*(volatile uint64_t *)(slot + 0) = UINT64_MAX;
	MemoryBarrier();
	*(volatile uint64_t *)(slot + 8) = (uint64_t)now.QuadPart;
	*(volatile uint32_t *)(slot + 16) = (uint32_t)size;
	*(volatile int32_t *)(slot + 20) = frames_lost;
	*(volatile uint32_t *)(slot + 24) = recovered ? 1u : 0u;
	*(volatile uint32_t *)(slot + 28) = flags;
	memcpy(slot + FRAMETAP_SLOT_HEADER, buf, size);
	MemoryBarrier();
	*(volatile uint64_t *)(slot + 0) = seq;
	InterlockedExchange64((volatile LONG64 *)H64(32), (LONG64)(seq + 1));
	if(g_event)
		SetEvent(g_event);
}

bool frametap_take_idr_request(void)
{
	if(!g_base)
		return false;
	return InterlockedCompareExchange((volatile LONG *)H32(40), 0, 1) == 1;
}

uint32_t frametap_take_press_request(uint32_t *hold_ms)
{
	if(!g_base)
		return 0;
	uint32_t buttons = (uint32_t)InterlockedExchange((volatile LONG *)H32(68), 0);
	if(buttons && hold_ms)
	{
		uint32_t h = *H32(72);
		*hold_ms = h < 20 ? 20 : h > 2000 ? 2000 : h;
	}
	return buttons;
}

void frametap_press_sent(void)
{
	if(!g_base)
		return;
	LARGE_INTEGER now;
	QueryPerformanceCounter(&now);
	InterlockedExchange64((volatile LONG64 *)H64(80), (LONG64)now.QuadPart);
	InterlockedIncrement((volatile LONG *)H32(76));
}

uint32_t frametap_audio_delay_ms(void)
{
	if(!g_base || *H32(0) != FRAMETAP_MAGIC)
		return 0;
	uint32_t ms = *H32(88);
	uint64_t freq = *H64(48);
	if(!ms || !freq)
		return 0;
	LARGE_INTEGER now;
	QueryPerformanceCounter(&now);
	if((int64_t)((uint64_t)now.QuadPart - *H64(56)) > (int64_t)(2 * freq)) // SmoovPlay stopped
		return 0;
	return ms > 500 ? 500 : ms;
}

void frametap_audio_format(uint32_t rate, uint32_t channels)
{
	if(!rate || !channels || channels > 8)
		return;
	g_arate = rate;
	g_achannels = channels;
	if(g_abase && *A32(0) == FRAMETAP_AUDIO_MAGIC)
	{
		*A32(16) = rate;
		*A32(20) = channels;
	}
}

void frametap_push_audio(const uint8_t *buf, size_t size)
{
	if(!g_abase || *A32(0) != FRAMETAP_AUDIO_MAGIC || !buf || !size)
		return;
	LARGE_INTEGER now;
	QueryPerformanceCounter(&now);
	uint64_t freq = *A64(32);
	size_t frame_bytes = 2u * g_achannels;
	size_t cap = (FRAMETAP_AUDIO_SLOT_SIZE - FRAMETAP_AUDIO_SLOT_HEADER) / frame_bytes * frame_bytes;
	size_t done = 0;
	while(done < size)
	{
		size_t n = size - done < cap ? size - done : cap;
		// A chunk too big for one slot goes in several; each slot's time is when its first sample arrived.
		uint64_t q = (uint64_t)now.QuadPart;
		if(done && freq)
			q += (uint64_t)((double)(done / frame_bytes) * (double)freq / (double)g_arate);
		uint64_t seq = *A64(24);
		uint8_t *slot = g_abase + FRAMETAP_AUDIO_HEADER + (size_t)(seq % FRAMETAP_AUDIO_SLOT_COUNT) * FRAMETAP_AUDIO_SLOT_SIZE;
		*(volatile uint64_t *)(slot + 0) = UINT64_MAX;
		MemoryBarrier();
		*(volatile uint64_t *)(slot + 8) = q;
		*(volatile uint32_t *)(slot + 16) = (uint32_t)n;
		*(volatile uint32_t *)(slot + 20) = g_arate;
		*(volatile uint32_t *)(slot + 24) = g_achannels;
		memcpy(slot + FRAMETAP_AUDIO_SLOT_HEADER, buf + done, n);
		MemoryBarrier();
		*(volatile uint64_t *)(slot + 0) = seq;
		InterlockedExchange64((volatile LONG64 *)A64(24), (LONG64)(seq + 1));
		done += n;
	}
}

void frametap_end(void)
{
	if(!g_base)
		return;
	// Keep the mapping open (the reader may still hold it); just mark the stream as stopped.
	InterlockedExchange((volatile LONG *)H32(0), 0);
	if(g_abase)
		InterlockedExchange((volatile LONG *)A32(0), 0);
	if(g_event)
		SetEvent(g_event);
}

#else // not Windows or Linux: compiled in, does nothing

bool frametap_enabled(void) { return false; }
bool frametap_begin(uint32_t codec, uint32_t width, uint32_t height, uint32_t max_fps)
{
	(void)codec; (void)width; (void)height; (void)max_fps;
	return false;
}
void frametap_push(const uint8_t *buf, size_t size, int32_t frames_lost, bool recovered)
{
	(void)buf; (void)size; (void)frames_lost; (void)recovered;
}
bool frametap_take_idr_request(void) { return false; }
uint32_t frametap_take_press_request(uint32_t *hold_ms) { (void)hold_ms; return 0; }
uint32_t frametap_audio_delay_ms(void) { return 0; }
void frametap_audio_format(uint32_t rate, uint32_t channels) { (void)rate; (void)channels; }
void frametap_push_audio(const uint8_t *buf, size_t size) { (void)buf; (void)size; }
void frametap_press_sent(void) {}
void frametap_end(void) {}

#endif

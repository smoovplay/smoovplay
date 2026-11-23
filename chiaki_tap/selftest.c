// Checks that frametap.c (the C writer inside chiaki-ng) and framegen/chiaki_tap.py (the Python reader) agree
// on the shared-memory layout. Run through chiaki_tap/selftest.py, which builds this with MSYS2's gcc: it writes
// 40 frames with known contents and flags and 21 chunks of sound (the last too big for one slot, so it's split), waits
// for the reader to ask for a keyframe, then answers a button press (as chiaki-ng does for the delay test) and reports
// the sound delay the reader set.
#include "frametap.h"

#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
static void Sleep(unsigned ms)
{
	struct timespec ts = {(time_t)(ms / 1000), (long)(ms % 1000) * 1000000L};
	nanosleep(&ts, NULL);
}
#define _putenv putenv
#endif

int main(void)
{
	_putenv("CHIAKI_FRAMETAP=1");
	if(!frametap_enabled() || !frametap_begin(1, 1920, 1080, 60))
	{
		fprintf(stderr, "frametap_begin failed\n");
		return 2;
	}
	static uint8_t buf[300000];
	// Sound first: 20 chunks of 10 ms (480 stereo samples), then one of 5000 samples (20000 bytes, three slots).
	frametap_audio_format(48000, 2);
	for(int k = 0; k < 21; k++)
	{
		size_t size = k < 20 ? 1920 : 20000;
		for(size_t j = 0; j < size; j++)
			buf[j] = (uint8_t)((k * 7 + j) & 0xff);
		frametap_push_audio(buf, size);
		Sleep(2);
	}
	for(int i = 0; i < 40; i++)
	{
		size_t size = (i % 2) ? 1000 + (size_t)i : 50000 + (size_t)i * 1000; // new/repeat-like sizes
		for(size_t j = 0; j < size; j++)
			buf[j] = (uint8_t)((i * 31 + j) & 0xff);
		frametap_push(buf, size, (i == 17) ? 2 : 0, i == 18);
		Sleep(16);
	}
	// The reader asks for a keyframe; chiaki-ng would then call RequestIDR().
	int t = 0;
	for(; t < 500 && !frametap_take_idr_request(); t++)
		Sleep(20);
	if(t == 500)
	{
		fprintf(stderr, "no idr request within 10 s\n");
		frametap_end();
		return 3;
	}
	// Then a button press (chiaki-ng presses, sends the state, and notes the time), and the sound delay.
	uint32_t buttons = 0, hold_ms = 0;
	for(t = 0; t < 500 && !(buttons = frametap_take_press_request(&hold_ms)); t++)
		Sleep(10);
	if(!buttons)
	{
		fprintf(stderr, "no press request within 5 s\n");
		frametap_end();
		return 4;
	}
	Sleep(30); // "sending"
	frametap_press_sent();
	uint32_t delay = 0;
	for(t = 0; t < 500 && !(delay = frametap_audio_delay_ms()); t++)
		Sleep(10);
	printf("idr request seen; press %u for %u ms sent; audio delay %u ms\n", buttons, hold_ms, delay);
	fflush(stdout);
	// Once the reader stops its heartbeat, the delay lapses on its own.
	for(t = 0; t < 400 && frametap_audio_delay_ms(); t++)
		Sleep(10);
	printf("audio delay lapsed: %s\n", frametap_audio_delay_ms() ? "no" : "yes");
	frametap_end();
	return 0;
}

// The handful of Win32 calls frametap.c uses, on Linux, so its Windows code builds there unchanged.
//
//   named file mapping       -> POSIX shared memory: "Local\chiaki-frametap" is /dev/shm/chiaki-frametap
//   QueryPerformanceCounter  -> CLOCK_MONOTONIC in nanoseconds (the clock time.perf_counter() reads on Linux)
//   Interlocked*, barriers   -> GCC/Clang __atomic builtins
//   the named auto-reset event -> a futex on write_seq's low 32 bits (header offset 32): SetEvent wakes the reader,
//                               which sleeps in FUTEX_WAIT there (framegen/tapsys.py). No name, nothing to open.
//
// Included by frametap.c only, in place of <windows.h>. CreateEventA is a macro naming frametap.c's g_base, which is
// in scope where it is used.
#ifndef FRAMETAP_POSIX_H
#define FRAMETAP_POSIX_H

#include <fcntl.h>
#include <linux/futex.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

typedef void *HANDLE;
typedef uint32_t DWORD;
typedef int32_t LONG;
typedef int64_t LONG64;
typedef union { int64_t QuadPart; } LARGE_INTEGER;

#define FALSE 0
#define INVALID_HANDLE_VALUE ((HANDLE)(intptr_t)-1)
#define PAGE_READWRITE 0x04
#define FILE_MAP_ALL_ACCESS 0xF001F

#define InterlockedExchange(p, v) __atomic_exchange_n((p), (v), __ATOMIC_SEQ_CST)
#define InterlockedExchange64(p, v) __atomic_exchange_n((p), (v), __ATOMIC_SEQ_CST)
#define InterlockedIncrement(p) __atomic_add_fetch((p), 1, __ATOMIC_SEQ_CST)
#define MemoryBarrier() __atomic_thread_fence(__ATOMIC_SEQ_CST)

static inline LONG frametap_cas32(volatile LONG *p, LONG exchange, LONG comparand)
{
	__atomic_compare_exchange_n(p, &comparand, exchange, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
	return comparand; // the value before, as InterlockedCompareExchange returns
}
#define InterlockedCompareExchange(p, exchange, comparand) frametap_cas32((p), (exchange), (comparand))

static inline int QueryPerformanceFrequency(LARGE_INTEGER *f)
{
	f->QuadPart = 1000000000;
	return 1;
}

static inline int QueryPerformanceCounter(LARGE_INTEGER *v)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	v->QuadPart = (int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
	return 1;
}

// A mapping handle is the shm file descriptor + 1, so NULL still means "failed".
static inline HANDLE CreateFileMappingA(HANDLE file, void *sa, DWORD protect, DWORD size_high, DWORD size_low,
	const char *name)
{
	(void)file; (void)sa; (void)protect;
	const char *leaf = strrchr(name, '\\');
	char path[1 + 256]; // "/" + the longest name frametap.c passes (its own 256-byte buffers)
	snprintf(path, sizeof(path), "/%s", leaf ? leaf + 1 : name);
	int fd = shm_open(path, O_RDWR | O_CREAT, 0600); // either side may create it first
	if(fd < 0)
		return NULL;
	off_t size = (off_t)(((uint64_t)size_high << 32) | size_low);
	struct stat st;
	if(fstat(fd, &st) != 0 || (st.st_size < size && ftruncate(fd, size) != 0))
	{
		close(fd);
		return NULL;
	}
	return (HANDLE)(intptr_t)(fd + 1);
}

static inline void *MapViewOfFile(HANDLE map, DWORD access, DWORD offset_high, DWORD offset_low, size_t size)
{
	(void)access; (void)offset_high; (void)offset_low;
	void *p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, (int)(intptr_t)map - 1, 0);
	return p == MAP_FAILED ? NULL : p;
}

static inline int CloseHandle(HANDLE h)
{
	return close((int)(intptr_t)h - 1) == 0;
}

// The event is write_seq's low word in the mapping; frametap.c creates it right after mapping g_base.
#define CreateEventA(sa, manual_reset, initial_state, name) ((void)(name), (HANDLE)(g_base + 32))

static inline int SetEvent(HANDLE event)
{
	// A shared futex (not FUTEX_PRIVATE): the reader is another process. Nothing waiting costs one syscall.
	syscall(SYS_futex, (uint32_t *)event, FUTEX_WAKE, 1, NULL, NULL, 0);
	return 1;
}

#endif

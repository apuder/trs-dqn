
#include <z80.c>
#include <unistd.h>
#include <strings.h>
#include <time.h>
#include <sys/time.h>
#include <errno.h>


// Breakpoint management
#define MAX_BREAKPOINTS 16

static int num_breakpoints = 0;
static ushort breakpoints[MAX_BREAKPOINTS];

int add_breakpoint(ushort addr)
{
    if (num_breakpoints >= MAX_BREAKPOINTS) {
        return -1;
    }
    breakpoints[num_breakpoints++] = addr;
    return 0;
}

int remove_breakpoint(ushort addr)
{
    for (int i = 0; i < num_breakpoints; i++) {
        if (breakpoints[i] == addr) {
            num_breakpoints--;
            for (int j = i; j < num_breakpoints; j++) {
                breakpoints[j] = breakpoints[j + 1];
            }
            return 0;
        }
    }
    return -1;
}

void clear_breakpoints()
{
    num_breakpoints = 0;
}

static int check_breakpoint(ushort addr)
{
    for (int i = 0; i < num_breakpoints; i++) {
        if (breakpoints[i] == addr) {
            return 1;
        }
    }
    return 0;
}


// Model I specs
#define TIMER_HZ_1 40
#define CLOCK_MHZ_1 1.77408
#define CYCLES_PER_TIMER ((unsigned int) (CLOCK_MHZ_1 * 1000000 / TIMER_HZ_1))

typedef unsigned char (*Z80_MEM_READ_FUNC)(unsigned long, ushort);
typedef void (*Z80_MEM_WRITE_FUNC)(int, ushort, byte);
typedef unsigned char (*Z80_IO_READ_FUNC)(unsigned long, ushort);
typedef void (*Z80_IO_WRITE_FUNC)(int, ushort, byte);

static Z80_MEM_READ_FUNC z80_mem_read_func = NULL;
static Z80_MEM_WRITE_FUNC z80_mem_write_func = NULL;
static Z80_IO_READ_FUNC z80_io_read_func = NULL;
static Z80_IO_WRITE_FUNC z80_io_write_func = NULL;

static volatile int z80_is_running = 1;

static Z80Context ctx;

volatile unsigned char ram[64 * 1024];

float screenshot[(2 * 64) * (3 * 16)];

void take_screenshot(int left, int top, int width, int height)
{
    bzero(screenshot, sizeof(screenshot));
    for (int x = left; x < (left + width); x++) {
        for (int y = top; y < (top + height); y++) {
            byte ch = ram[0x3c00 + y * 64 + x];
            if (ch < 0x80 || ch > 0xbf) continue;
            ch -= 0x80;
            int base = (y * 3 * 128) + (x * 2);
            for (int i = 0; i < 6; i++) {
                if (ch & 1) {
                    screenshot[base + (i & 1) + (i >> 1) * 128] = 1.0f;
                }
                ch = ch >> 1;
            }
        }
    }
}

static unsigned char z80_mem_read(unsigned long param, ushort address)
{
    if (z80_mem_read_func != NULL) {
        return (*z80_mem_read_func)(param, address);
    }
    return ram[address];
}

static void z80_mem_write(unsigned long param, ushort address, unsigned char data)
{
    if (z80_mem_write_func != NULL) {
        (*z80_mem_write_func)(param, address, data);
    }
    ram[address] = data;
}

static unsigned char z80_io_read(unsigned long param, ushort address)
{
    if (z80_io_read_func != NULL) {
        return (*z80_io_read_func)(param, address);
    }
    return 255;
}

static void z80_io_write(unsigned long param, ushort address, unsigned char data)
{
    if (z80_io_write_func != NULL) {
        (*z80_io_write_func)(param, address, data);
    }
}

void z80_init_callbacks(Z80_MEM_READ_FUNC mem_read_func,
                        Z80_MEM_WRITE_FUNC mem_write_func,
                        Z80_IO_READ_FUNC io_read_func,
                        Z80_IO_WRITE_FUNC io_write_func)
{
    z80_mem_read_func = mem_read_func;
    z80_mem_write_func = mem_write_func;
    z80_io_read_func = io_read_func;
    z80_io_write_func = io_write_func;
}

static int get_ticks()
{
    static struct timeval start_tv, now;
    static int init = 0;

    if (!init) {
       gettimeofday(&start_tv, NULL);
       init = 1;
    }

    gettimeofday(&now, NULL);
    return (now.tv_sec - start_tv.tv_sec) * 1000 +
                 (now.tv_usec - start_tv.tv_usec) / 1000;
}

static void delay(int ms)
{
    int was_error;
    struct timespec elapsed, tv;

    elapsed.tv_sec = ms / 1000;
    elapsed.tv_nsec = (ms % 1000) * 1000000;
    do {
        errno = 0;

        tv.tv_sec = elapsed.tv_sec;
        tv.tv_nsec = elapsed.tv_nsec;
        was_error = nanosleep(&tv, &elapsed);
    } while (was_error && (errno == EINTR));
}

static void sync_time_with_host()
{
	int curtime;
	int deltatime;
    static int lasttime = 0;

    deltatime = 1000 / TIMER_HZ_1;

	curtime = get_ticks();

	if (lasttime + deltatime > curtime) {
		delay(lasttime + deltatime - curtime);
    }
	curtime = get_ticks();

	lasttime += deltatime;
	if ((lasttime + deltatime) < curtime) {
		lasttime = curtime;
	}
}

void z80_reset(ushort entryAddr)
{
    bzero(&ctx, sizeof(Z80Context));
    Z80RESET(&ctx);
    ctx.PC = entryAddr;
    ctx.memRead = z80_mem_read;
    ctx.memWrite = z80_mem_write;
    ctx.ioRead = z80_io_read;
    ctx.ioWrite = z80_io_write;
}

void z80_run(ushort entryAddr)
{
    z80_reset(entryAddr);

    while (z80_is_running) {
        Z80Execute(&ctx);
        if (ctx.tstates >= CYCLES_PER_TIMER) {
		    sync_time_with_host();
		    ctx.tstates -=  CYCLES_PER_TIMER;
		}
    }
}

void z80_set_running(int is_running)
{
    z80_is_running = is_running;
}

int z80_run_for_tstates(int tstates, int original_speed)
{
    if (!original_speed) {
        ctx.tstates = 0;
    }
    int threshold_tstates = ctx.tstates + tstates;
    while (ctx.tstates <= threshold_tstates) {
        Z80Execute(&ctx);
        if (original_speed && (ctx.tstates >= CYCLES_PER_TIMER)) {
            sync_time_with_host();
            ctx.tstates -=  CYCLES_PER_TIMER;
            threshold_tstates -= CYCLES_PER_TIMER;
        }
    }
    return ctx.tstates - threshold_tstates;
}

ushort z80_resume(int original_speed)
{
  int current_tstates = ctx.tstates;
  do {
    Z80Execute(&ctx);
    if (original_speed && (ctx.tstates - current_tstates >= CYCLES_PER_TIMER)) {
        sync_time_with_host();
        current_tstates = ctx.tstates;
    }
  } while(!check_breakpoint(ctx.PC));
  return ctx.PC;
}

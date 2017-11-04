
#include <z80.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <errno.h>

// Model I specs
#define TIMER_HZ_1 40
#define CLOCK_MHZ_1 1.77408

typedef byte (*Z80_MEM_READ_FUNC)(int, ushort);
typedef void (*Z80_MEM_WRITE_FUNC)(int, ushort, byte);
typedef byte (*Z80_IO_READ_FUNC)(int, ushort);
typedef void (*Z80_IO_WRITE_FUNC)(int, ushort, byte);

static Z80_MEM_READ_FUNC z80_mem_read_func = NULL;
static Z80_MEM_WRITE_FUNC z80_mem_write_func = NULL;
static Z80_IO_READ_FUNC z80_io_read_func = NULL;
static Z80_IO_WRITE_FUNC z80_io_write_func = NULL;

static volatile int z80_is_running = 1;

volatile byte ram[64 * 1024];

static byte z80_mem_read(int param, ushort address)
{
    if (z80_mem_read_func != NULL) {
        return (*z80_mem_read_func)(param, address);
    }
    return ram[address];
}

static void z80_mem_write(int param, ushort address, byte data)
{
    if (z80_mem_write_func != NULL) {
        (*z80_mem_write_func)(param, address, data);
    }
    ram[address] = data;
}

static byte z80_io_read(int param, ushort address)
{
    if (z80_io_read_func != NULL) {
        return (*z80_io_read_func)(param, address);
    }
    return 0;
}

static void z80_io_write(int param, ushort address, byte data)
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

void z80_run(ushort entryAddr)
{
    Z80Context ctx;
    Z80RESET(&ctx);
    ctx.PC = entryAddr;
    ctx.memRead = z80_mem_read;
    ctx.memWrite = z80_mem_write;
    ctx.ioRead = z80_io_read;
    ctx.ioWrite = z80_io_write;

    unsigned int cycles_per_timer = CLOCK_MHZ_1 * 1000000 / TIMER_HZ_1;

    while (z80_is_running) {
        Z80Execute(&ctx);
        if (ctx.tstates >= cycles_per_timer) {
		    sync_time_with_host();
		    ctx.tstates -=  cycles_per_timer;
		}
    }
}

void z80_set_running(int is_running)
{
    z80_is_running = is_running;
}




import ctypes

wrapper = ctypes.CDLL('libtrs.so')

ram = None

def z80_mem_read(param, address):
    return 0

def z80_mem_write(param, address, data):
    pass

def z80_io_read(param, address):
    return 0

def z80_io_write(param, address, data):
    pass


class Z80():

    def __init__(self, memory):
        global ram, wrapper
        ram = memory

        Z80_MEM_READ_FUNC = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_ushort)
        Z80_MEM_WRITE_FUNC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int, ctypes.c_ushort, ctypes.c_byte)
        Z80_IO_READ_FUNC = ctypes.CFUNCTYPE(ctypes.c_byte, ctypes.c_int, ctypes.c_ushort)
        Z80_IO_WRITE_FUNC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int, ctypes.c_ushort, ctypes.c_byte)

        self.z80_mem_read_func = Z80_MEM_READ_FUNC(z80_mem_read)
        self.z80_mem_write_func = Z80_MEM_WRITE_FUNC(z80_mem_write)
        self.z80_io_read_func = Z80_IO_READ_FUNC(z80_io_read)
        self.z80_io_write_func = Z80_IO_WRITE_FUNC(z80_io_write)

        wrapper.z80_init_callbacks.argtypes = (Z80_MEM_READ_FUNC,
                                               Z80_MEM_WRITE_FUNC,
                                               Z80_IO_READ_FUNC,
                                               Z80_IO_WRITE_FUNC)
        #wrapper.z80_init_callbacks(self.z80_mem_read_func,
        #                           self.z80_mem_write_func,
        #                           self.z80_io_read_func,
        #                           self.z80_io_write_func)

        wrapper.z80_reset.argtypes = (ctypes.c_ushort,)
        wrapper.z80_run.argtypes = (ctypes.c_ushort,)
        wrapper.z80_run_for_tstates.argtypes = (ctypes.c_int, ctypes.c_int)
        wrapper.z80_run_for_tstates.restype = ctypes.c_int

    def reset(self, entry_addr):
        wrapper.z80_reset(entry_addr)

    def run_for_tstates(self, tstates, original_speed):
        return wrapper.z80_run_for_tstates(tstates, original_speed)

    def run(self, entry_addr):
        wrapper.z80_run(entry_addr)
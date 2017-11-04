
import ctypes

wrapper = ctypes.CDLL('libtrs.so')


class RAM():
    ram_size = 64 * 1024

    def __init__(self):
        self.ram = (ctypes.c_byte * RAM.ram_size).in_dll(wrapper, "ram")

    def peek(self, addr):
        if addr > RAM.ram_size:
            print 'abort'
        return self.ram[addr]

    def poke(self, addr, val):
        self.ram[addr] = val
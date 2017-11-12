
import ctypes

wrapper = ctypes.CDLL('libtrs.so')


class RAM():
    ram_size = 64 * 1024

    def __init__(self):
        self.ram = (ctypes.c_ubyte * RAM.ram_size).in_dll(wrapper, "ram")
        for i in range(RAM.ram_size):
            self.poke(i, 0)

    def backup(self):
        self.ramBackup = bytearray()
        for i in range(RAM.ram_size):
            self.ramBackup.append(self.peek(i))

    def restore(self):
        for i in range(RAM.ram_size):
            self.poke(i, self.ramBackup[i])

    def peek(self, addr):
        if addr > RAM.ram_size:
            print 'abort'
        return self.ram[addr]

    def poke(self, addr, val):
        self.ram[addr] = val
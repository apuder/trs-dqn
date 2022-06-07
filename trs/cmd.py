
from array import *
import os


class CMD():

    def __init__(self, ram):
        self.ram = ram

    def next(self):
        b = self.cmd[self.i]
        self.i += 1
        return b

    def load(self, cmd_file):
        self.cmd = array('B')
        statinfo = os.stat(cmd_file)
        f = open(cmd_file, 'rb')
        self.cmd.fromfile(f, statinfo.st_size)
        self.i = 0
        while True:
            b = self.next()
            len = self.next()
            if b == 1:
                if len < 3:
                    len += 256
                #len = 256 # XXX
                addr = self.next() | (self.next() << 8)
                for i in range(len - 2):
                    self.ram.poke(addr + i, self.next())
            elif b == 2:
                if len != 2:
                    raise "Entry address needs to be of len 2"
                return self.next() | (self.next() << 8)
            elif b == 5:
                for i in range(len):
                    self.next()
            else:
                raise "Bad CMD file format"


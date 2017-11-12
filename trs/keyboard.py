
from enum import Enum

class Key(Enum):
    A = 1
    B = 2
    C = 3
    D = 4
    E = 5
    F = 6
    G = 7
    H = 8
    I = 9
    J = 10
    K = 11
    L = 12
    M = 13
    N = 14
    O = 15
    P = 16
    Q = 17
    R = 18
    S = 19
    T = 20
    U = 21
    V = 22
    W = 23
    X = 24
    Y = 25
    Z = 26
    _0 = 27
    _1 = 28
    _2 = 29
    _3 = 30
    _4 = 31
    _5 = 32
    _6 = 33
    _7 = 34
    _8 = 35
    _9 = 36
    AT = 37
    COMMA = 38
    ENTER = 39
    CLEAR = 40
    BREAK = 41
    UP = 42
    DOWN = 43
    LEFT = 44
    RIGHT = 45
    SPACE = 46

keymap = {
    Key.AT: (1, 1),
    Key.A: (1, 2),
    Key.B: (1, 4),
    Key.C: (1, 8),
    Key.D: (1, 16),
    Key.E: (1, 32),
    Key.F: (1, 64),
    Key.G: (1, 128),
    Key.H: (2, 1),
    Key.I: (2, 2),
    Key.J: (2, 4),
    Key.K: (2, 8),
    Key.L: (2, 16),
    Key.M: (2, 32),
    Key.N: (2, 64),
    Key.O: (2, 128),
    Key.P: (4, 1),
    Key.Q: (4, 2),
    Key.R: (4, 4),
    Key.S: (4, 8),
    Key.T: (4, 16),
    Key.U: (4, 32),
    Key.V: (4, 64),
    Key.W: (4, 128),
    Key.X: (8, 1),
    Key.Y: (8, 2),
    Key.Z: (8, 4),
    Key.COMMA: (8, 8),
    Key._0: (16, 1),
    Key._1: (16, 2),
    Key._2: (16, 4),
    Key._3: (16, 8),
    Key._4: (16, 16),
    Key._5: (16, 32),
    Key._6: (16, 64),
    Key._7: (16, 128),
    Key._8: (32, 1),
    Key._9: (32, 2),
    Key.ENTER: (64, 1),
    Key.CLEAR: (64, 2),
    Key.BREAK: (64, 4),
    Key.UP: (64, 8),
    Key.DOWN: (64, 16),
    Key.LEFT: (64, 32),
    Key.RIGHT: (64, 64),
    Key.SPACE: (64, 128)
}

"""
Address   1     2     4     8     16     32     64     128    Hex Address
------- ----- ---,- ----- ----- -----  -----  -----   -----   -----------
14337     @     A     B     C      D      E      F      G        3801       1
14338     H     I     J     K      L      M      N      O        3802       2
14340     P     Q     R     S      T      U      V      W        3804       4
14344     X     Y     Z     ,      -      -      -      -        3808       8
14352     0     1     2     3      4      5      6      7        3810      16
14368     8     9    *:    +;     <,     =-     >.     ?/        3820      32
14400  enter  clear break  up    down   left  right  space       3840      64
14464  shift    -     -     -  control    -      -      -        3880
"""

class Keyboard():

    KEYB_BASE = 0x3800

    def __init__(self, ram):
        self.ram = ram

    def all_keys_up(self):
        for i in range(7):
            self.ram.poke(Keyboard.KEYB_BASE + (1 << i), 0)

    def key_down(self, key):
        (offset, mask) = keymap[key]
        addr = Keyboard.KEYB_BASE + offset
        b = self.ram.peek(addr)
        b |= mask
        self.ram.poke(addr, b)

    def key_up(self, key):
        (offset, mask) = keymap[key]
        addr = Keyboard.KEYB_BASE + offset
        b = self.ram.peek(addr)
        b &= ~mask
        self.ram.poke(addr, b)

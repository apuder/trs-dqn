
from pyglet.window import key

keymap = {
    key.AT: (1, 1),
    key.A: (1, 2),
    key.B: (1, 4),
    key.C: (1, 8),
    key.D: (1, 16),
    key.E: (1, 32),
    key.F: (1, 64),
    key.G: (1, 128),
    key.H: (2, 1),
    key.I: (2, 2),
    key.J: (2, 4),
    key.K: (2, 8),
    key.L: (2, 16),
    key.M: (2, 32),
    key.N: (2, 64),
    key.O: (2, 128),
    key.P: (4, 1),
    key.Q: (4, 2),
    key.R: (4, 4),
    key.S: (4, 8),
    key.T: (4, 16),
    key.U: (4, 32),
    key.V: (4, 64),
    key.W: (4, 128),
    key.X: (8, 1),
    key.Y: (8, 2),
    key.Z: (8, 4),
    key.COMMA: (8, 8),
    key._0: (16, 1),
    key._1: (16, 2),
    key._2: (16, 4),
    key._3: (16, 8),
    key._4: (16, 16),
    key._5: (16, 32),
    key._6: (16, 64),
    key._7: (16, 128),
    key._8: (32, 1),
    key._9: (32, 2),
    # Skipping some keys
    key.ENTER: (64, 1),
    key.F1: (64, 2), # Clear
    key.F2: (64, 4), # Break
    key.UP: (64, 8),
    key.DOWN: (64, 16),
    key.LEFT: (64, 32),
    key.RIGHT: (64, 64),
    key.SPACE: (64, 128)
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

    def key_down(self, symbol, modifier):
        if not symbol in keymap:
            return
        (offset, mask) = keymap[symbol]
        addr = Keyboard.KEYB_BASE + offset
        b = self.ram.peek(addr)
        b |= mask
        self.ram.poke(addr, b)

    def key_up(self, symbol, modifier):
        if not symbol in keymap:
            return
        (offset, mask) = keymap[symbol]
        addr = Keyboard.KEYB_BASE + offset
        b = self.ram.peek(addr)
        b &= ~mask
        self.ram.poke(addr, b)

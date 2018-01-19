
import pyglet, numpy
from PIL import ImageFont, ImageDraw, Image
from keyboard import Key
from pyglet.window import key


pyglet_keymap = {
    key.AT: Key.AT,
    key.A: Key.A,
    key.B: Key.B,
    key.C: Key.C,
    key.D: Key.D,
    key.E: Key.E,
    key.F: Key.F,
    key.G: Key.G,
    key.H: Key.H,
    key.I: Key.I,
    key.J: Key.J,
    key.K: Key.K,
    key.L: Key.L,
    key.M: Key.M,
    key.N: Key.N,
    key.O: Key.O,
    key.P: Key.P,
    key.Q: Key.Q,
    key.R: Key.R,
    key.S: Key.S,
    key.T: Key.T,
    key.U: Key.U,
    key.V: Key.V,
    key.W: Key.W,
    key.X: Key.X,
    key.Y: Key.Y,
    key.Z: Key.Z,
    key.COMMA: Key.COMMA,
    key._0: Key._0,
    key._1: Key._1,
    key._2: Key._2,
    key._3: Key._3,
    key._4: Key._4,
    key._5: Key._5,
    key._6: Key._6,
    key._7: Key._7,
    key._8: Key._8,
    key._9: Key._9,
    key.ENTER: Key.ENTER,
    key.F1: Key.CLEAR,
    key.F2: Key.BREAK,
    key.UP: Key.UP,
    key.DOWN: Key.DOWN,
    key.LEFT: Key.LEFT,
    key.RIGHT: Key.RIGHT,
    key.SPACE: Key.SPACE
}


class Video(pyglet.window.Window):

    screen_width = 64
    screen_height = 16
    font_size = 26
    char_width = 8
    char_height = 26

    trsTTF = ImageFont.truetype("var/AnotherMansTreasureMIII64C.ttf", font_size)

    def __init__(self, ram, keyboard, fps):
        super(Video, self).__init__(Video.screen_width * Video.char_width,
                                    Video.screen_height * Video.char_height, vsync = False)

        self.ram = ram
        self.keyboard = keyboard
        self.fps = fps

        self.font = []

        for i in range(256):
            image = Image.new("RGB", (Video.char_width, Video.char_height), "black")
            draw = ImageDraw.Draw(image)
            draw.text((0, -1), unichr(0xe000 + i), font=Video.trsTTF)
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            raw_image = image.tobytes(encoder_name="raw")
            pygimage = pyglet.image.ImageData(Video.char_width, Video.char_height, 'RGB', raw_image)
            self.font.append(pygimage)

        self.sprites = []
        self.batch = pyglet.graphics.Batch()
        height = (Video.screen_height - 1) * Video.char_height

        for i in range(Video.screen_width * Video.screen_height):
            x = i % Video.screen_width
            y = i / Video.screen_width
            x *= Video.char_width
            y *= Video.char_height
            y = height - y
            sprite = pyglet.sprite.Sprite(self.font[0], x=x, y=y, batch=self.batch)
            sprite.ch = 0
            self.sprites.append(sprite)

        pyglet.clock.schedule_interval(self.update, 1.0 / self.fps)
        pyglet.clock.set_fps_limit(self.fps)

    def update(self, dt):
        pass

    def begin(self):
        return 0x3c00

    def end(self):
        return 0x3c00 + 64 * 16

    def on_draw(self):
        pyglet.clock.tick()
        self.clear()
        for i in range(64 * 16):
            ch = self.ram.peek(0x3c00 + i)
            sprite = self.sprites[i]
            if sprite.ch != ch:
                sprite.image = self.font[ch]
                sprite.ch = ch
        self.batch.draw()

    def on_key_press(self, symbol, modifiers):
        if not symbol in pyglet_keymap:
            return
        self.keyboard.key_down(pyglet_keymap[symbol])

    def on_key_release(self, symbol, modifiers):
        if not symbol in pyglet_keymap:
            return
        self.keyboard.key_up(pyglet_keymap[symbol])

    def mainloop(self):
        pyglet.app.run()

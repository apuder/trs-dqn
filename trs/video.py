
import pyglet
from PIL import ImageFont, ImageDraw, Image

class Video(pyglet.window.Window):

    screen_width = 64
    screen_height = 16
    font_size = 26
    char_width = 8
    char_height = 26

    fps = 60.0

    trsTTF = ImageFont.truetype("var/AnotherMansTreasureMIII64C.ttf", font_size)

    def __init__(self, ram, keyboard):
        super(Video, self).__init__(Video.screen_width * Video.char_width,
                                    Video.screen_height * Video.char_height, vsync = False)

        self.ram = ram
        self.keyboard = keyboard

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

        pyglet.clock.schedule_interval(self.update, 1.0 / Video.fps)
        pyglet.clock.set_fps_limit(Video.fps)

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
        self.keyboard.key_down(symbol, modifiers)

    def on_key_release(self, symbol, modifiers):
        self.keyboard.key_up(symbol, modifiers)

    def mainloop(self):
        pyglet.app.run()

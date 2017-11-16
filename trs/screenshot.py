
import numpy
from PIL import ImageFont, ImageDraw, Image


class Screenshot():

    screen_width = 64
    screen_height = 16
    font_size = 26
    char_width = 8
    char_height = 26

    trsTTF = ImageFont.truetype("var/AnotherMansTreasureMIII64C.ttf", font_size)

    def __init__(self, ram, viewport):
        self.ram = ram
        self.viewport_x, self.viewport_y, self.viewport_w, self.viewport_h = viewport

    def screenshot(self):
        image = Image.new("RGB", (Screenshot.char_width * self.viewport_w, Screenshot.char_height * self.viewport_h), "black")
        draw = ImageDraw.Draw(image)
        for x in range(self.viewport_w):
            for y in range(self.viewport_h):
                ch = self.ram.peek(0x3c00 + Screenshot.screen_width * (y + self.viewport_y) + x + self.viewport_x)
                draw.text((x * Screenshot.char_width, y * Screenshot.char_height-1), unichr(0xe000 + ch),
                          font=Screenshot.trsTTF)
        return numpy.array(image)

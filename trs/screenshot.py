
import numpy
from PIL import ImageFont, ImageDraw, Image
import ctypes

wrapper = ctypes.CDLL('libtrs.so')


class Screenshot():

    screen_width = 64
    screen_height = 16
    font_size = 26
    char_width = 8
    char_height = 26
    resolution_x = 2 * 64
    resolution_y = 3 * 16
    screenshot_size = resolution_x * resolution_y

    trsTTF = ImageFont.truetype("var/AnotherMansTreasureMIII64C.ttf", font_size)

    def __init__(self, ram, viewport):
        global wrapper
        self.screenshot_mem = (ctypes.c_float * Screenshot.screenshot_size).in_dll(wrapper, "screenshot")
        wrapper.take_screenshot.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
        self.ram = ram
        self.viewport_x, self.viewport_y, self.viewport_w, self.viewport_h = viewport

    def screenshotx(self):
        image = Image.new("RGB", (Screenshot.char_width * self.viewport_w, Screenshot.char_height * self.viewport_h), "black")
        draw = ImageDraw.Draw(image)
        for x in range(self.viewport_w):
            for y in range(self.viewport_h):
                ch = self.ram.peek(0x3c00 + Screenshot.screen_width * (y + self.viewport_y) + x + self.viewport_x)
                draw.text((x * Screenshot.char_width, y * Screenshot.char_height-1), chr(0xe000 + ch),
                          font=Screenshot.trsTTF)
        return numpy.array(image)

    def screenshot(self):
        wrapper.take_screenshot(self.viewport_x, self.viewport_y, self.viewport_w, self.viewport_h)
        return numpy.array(self.screenshot_mem).reshape(Screenshot.resolution_y, Screenshot.resolution_x)
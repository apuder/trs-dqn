
from z80 import Z80
from screenshot import Screenshot
from keyboard import Keyboard
from keyboard import Key
from ram import RAM
from cmd import CMD

from threading import Thread
import types


class TRS():

    def __init__(self, config, original_speed, fps, no_ui):
        self.config = config
        self.original_speed = original_speed
        self.ram = RAM()
        self.keyboard = Keyboard(self.ram)
        self.z80 = Z80(self.ram)
        self.cmd = CMD(self.ram)
        self.entry_addr = self.cmd.load(config["cmd"])
        self.ram.backup()
        self.reset()
        if not no_ui:
            from video import Video
            self.video = Video(self.ram, self.keyboard, fps)
            self.video.mainloop()

    def cpu_thread(self):
        self.z80.run(self.entry_addr)

    def reset(self):
        self.z80.reset(self.entry_addr)
        self.ram.restore()

    def run_for_tstates(self, tstates):
        return self.z80.run_for_tstates(tstates, self.original_speed)

    def boot(self):
        self.reset()
        self.ram.restore()
        self.keyboard.all_keys_up()
        for action in self.config["boot"]:
            if type(action) is types.IntType:
                self.run_for_tstates(action)
            elif type(action) is Key:
                self.keyboard.key_down(action)
            else:
                print "Bad boot action:", action
        self.keyboard.all_keys_up()

    def run(self):
        self.cpu_thread()
        self.cpuThread = Thread(target = self.cpu_thread)
        self.cpuThread.start()

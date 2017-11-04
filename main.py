
from threading import Thread
from trs import *

ram = RAM()
keyboard = Keyboard(ram)
video = Video(ram, keyboard)
z80 = Z80(ram)
cmd = CMD(ram)
entry_addr = cmd.load("var/defense.cmd")

def cpu_thread():
    z80.run(entry_addr)

if __name__ == "__main__":
    cpuThread = Thread(target = cpu_thread)
    cpuThread.start()
    video.mainloop()

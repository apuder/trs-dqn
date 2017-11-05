
TRS-80 Deep-Q Network
=====================

Training a Deep-Q Network to play retro games for the TRS-80.
Work-in-progress.

Compiling
---------

Requires Python 2 and gcc installation.

```
sudo pip install pillow
sudo pip install pyglet
git clone --recursive git@github.com:apuder/trs-dqn.git
cd trs-dqn
export LD_LIBRARY_PATH=`pwd`/lib
make
python main.py
```

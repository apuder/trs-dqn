
TRS-80 Deep-Q Network
=====================

Training a Deep-Q Network to play retro games for the TRS-80.
Work-in-progress.

Compiling
---------

Requires Python 2 and gcc installation.

```
pip install pillow
pip install pyglet
pip install scikit-image
pip install keras
pip install tensorflow
pip install h5py
git clone --recursive https://github.com/apuder/trs-dqn.git
cd trs-dqn
export LD_LIBRARY_PATH=`pwd`
make
```

Running
-------

Let a human user play the game:

```
python main.py -m Play

```

Train the network. The optional parameter `--no-ui` disables the UI during training:

```
python main.py -m Train [--no-ui]

```

Play a game by using the trained network:

```
python main.py -m Run

```


Credits
-------

Based on <https://github.com/yanpanlau/Keras-FlappyBird>

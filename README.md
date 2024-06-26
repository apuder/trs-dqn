
# TRS-80 Deep-Q Network

Training a Deep-Q Network to play retro games for the TRS-80.
Work-in-progress.

*Forked from https://github.com/apuder/trs-dqn*

## Setup
> **NOTE**: This is working with Python 3.11 (and probably some earlier versions), but DOES NOT
            work with 3.12 since some dependencies are not supported yet.

Check this repository out recursively!
```
git clone --recursive <repo url>
```

Note: if you forgot to check it out recursively, you can run the following after the fact,
      without starting over:
```
git submodule init
git submodule update
```

Run the following inside the repo folder the first time after you've checked out
the repo and every time you check it out to ensure you have the latest
dependencies:
```
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt

export LD_LIBRARY_PATH=$(pwd)
make
```

## Running
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
python main.py -m Run --model=<model-file.h5>
```


## Credits

Based on <https://keras.io/examples/rl/deep_q_network_breakout/>


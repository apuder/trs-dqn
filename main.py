
from __future__ import print_function

from trs import TRS
from trs import Key
from trs import Screenshot

class RewardCosmicFighter():

    default_reward = (-0.2, False)

    def __init__(self, ram):
        self.ram = ram
        self.score = 0

    def reset(self):
        self.score = 0

    def compute(self):
        b = bytearray()
        for i in range(15):
            b.append(self.ram.peek(0x3c00 + i))

        if 0x20 not in b:
            # Score not shown right now
            #if 0xbf in b:
                # Evil eye appears to be attacking
                #return (-0.2, False)
            return RewardCosmicFighter.default_reward
        # Get score
        try:
            i = 0
            while b[i] != 0x20:
                i += 1
            j = i + 1
            while b[j] != 0x20:
                j += 1
            new_score = int(b[i:j])
        except (ValueError, IndexError):
            # Score was not fully rendered yet
            return RewardCosmicFighter.default_reward

        if b.count(b'\x5b') == 2:
            # Lost a ship. Game over
            return (-1.0, True)

        delta = new_score - self.score
        if delta != 0:
            # Score increased
            self.score = new_score
            if delta >= 100:
                # Killed an evil eye
                return (1.0, False)
            else:
                return (1.0, False)
        return RewardCosmicFighter.default_reward

config_cosmic = {
    "name": "cosmic",
    "cmd": "var/cosmic.cmd",
    "boot": [1000000, Key.CLEAR, Key._1, 1000000, 1000000, 1678000],
    "viewport": (0, 2, 64, 14),
    "step": 50000,
    "actions": [None, [Key.SPACE], [Key.LEFT], [Key.LEFT, Key.SPACE],
                [Key.RIGHT], [Key.RIGHT, Key.SPACE]
                #, [Key.D]
                ],
    "reward": RewardCosmicFighter
}


class RewardBreakdown():

    default_reward = (0.0, False)

    def __init__(self, ram):
        self.ram = ram
        self.score = 0

    def reset(self):
        self.score = 0

    def compute(self):
        b = bytearray()
        for i in range(5):
            b.append(self.ram.peek(0x3c00 + 6 + i))

        # Get score
        try:
            new_score = int(b[:])
        except (ValueError, IndexError):
            # Score was not fully rendered yet
            return RewardBreakdown.default_reward

        if self.ram.peek(0x3c00 + 668) == 80:
            return (-1.0, True)

        delta = new_score - self.score
        if delta != 0:
            # Score increased
            self.score = new_score
            return (1.0, False)
        return RewardBreakdown.default_reward

config = {
    "name": "breakdown",
    "cmd": "var/breakdown.cmd",
    "boot": [1000000, 1000000, 1000000, Key.SPACE, 1000000, 1000000, 1000000, 1000000, 1000000,
             1000000, 800000],
    "viewport": (0, 2, 64, 14),
    "step": 50000,
    "actions": [None, [Key.LEFT], [Key.RIGHT]],
    "reward": RewardBreakdown
}



class Game():

    def __init__(self, trs):
        self.trs = trs
        self.config = trs.config
        self.reward = self.config["reward"](trs.ram)
        self.step = self.config["step"]
        self.actions = self.config["actions"]
        viewport = self.config["viewport"]
        self.screenshot = Screenshot(trs.ram, viewport)
        self.delta_tstates = 0

    def frame_step(self, action):
        self.trs.keyboard.all_keys_up()
        i, = np.where(action == 1)
        keys = self.actions[i[0]]
        if keys != None:
            for key in keys:
                self.trs.keyboard.key_down(key)
        tstates = self.step - self.delta_tstates
        self.delta_tstates = self.trs.run_for_tstates(tstates)
        reward, terminal = self.reward.compute()
        screenshot = self.screenshot.screenshot()
        if terminal:
            self.reward.reset()
            self.trs.boot()
            self.delta_tstates = 0
        return (screenshot, reward, terminal)


#----------------------------------------------------------------------------
# The following is adopted from https://github.com/yanpanlau/Keras-FlappyBird
#----------------------------------------------------------------------------

import argparse
import skimage as skimage
from skimage import transform, color, exposure

import random
import numpy as np
from collections import deque

from threading import Thread

import sqlite3, pickle

import json
import os.path
from keras.models import Sequential
from keras.models import clone_model
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
import tensorflow as tf

CONFIG = 'nothreshold'
ACTIONS = len(config["actions"])  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVATION = 50000.  # timesteps to observe before training
EXPLORE = 1000000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.1  # final value of epsilon
INITIAL_EPSILON = 1  # starting value of epsilon
REPLAY_MEMORY = 500000  # number of previous transitions to remember
BATCH = 32  # size of minibatch
FRAME_PER_ACTION = 1
TARGET_MODEL_UPDATE = 10000
LEARNING_RATE = 0.0001

img_rows, img_cols = 80, 80
# Convert image into Black and white
img_channels = 4  # We stack 4 frames


class ReplayMemory():

    def __init__(self, name, size):
        self.size = size
        self.D = deque()

    def add_transition(self, transition):
        self.D.append(transition)
        if len(self.D) > self.size:
            self.D.popleft()

    def get_mini_batch(self, batch_size):
        return random.sample(self.D, batch_size)


class PersistentReplayMemory():

    def __init__(self, name, size):
        self.size = size
        self.conn = sqlite3.connect(name + '.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute("create table if not exists replay_mem(id integer primary key, transition blob)")
        row = self.cursor.execute("select min(id) from replay_mem").fetchone()[0]
        self.tail = 0 if row is None else row
        row = self.cursor.execute("select max(id) from replay_mem").fetchone()[0]
        self.head = 0 if row is None else row + 1

    def add_transition(self, transition):
        id = self.head
        self.head += 1
        self.cursor.execute("insert into replay_mem(id, transition) values (?, ?)", (id, sqlite3.Binary(pickle.dumps(transition, protocol=2))))
        if self.head - self.tail > self.size:
            id = self.tail
            self.cursor.execute("delete from replay_mem where id=?", (id,))
            self.tail += 1
        self.conn.commit()

    def get_mini_batch(self, batch_size):
        n = self.head - self.tail
        if not 0 <= batch_size <= n:
            raise ValueError, "sample larger than population"
        result = [None] * batch_size
        selected = set()
        for i in xrange(batch_size):
            j = random.randrange(n)
            while j in selected:
                j = random.randrange(n)
            selected.add(j)
            result[i] = self.get_transition(j)
        return result

    def get_transition(self, i):
        id = self.tail + i
        row = self.cursor.execute("select transition from replay_mem where id = ?", (id,)).fetchone()
        blob = row[0]
        return pickle.loads(str(blob))


def buildmodel():
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=(4, 4), padding='same',
                            input_shape=(img_rows, img_cols, img_channels)))  # 80*80*4
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(ACTIONS))

    optimizer = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse', optimizer=optimizer)
    return model


def trainNetwork(trs, model, args):
    # Target model
    target_model = clone_model(model)
    
    # open up a game state to communicate with emulator
    game_state = Game(trs)

    # store the previous observations in replay memory
    name = config["name"]
    D = PersistentReplayMemory(name, REPLAY_MEMORY)

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)

    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t, (80, 80))
    x_t = skimage.exposure.rescale_intensity(x_t, out_range=(0, 255))

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # In Keras, need to reshape
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 1*80*80*4

    t = 0
    OBSERVE = OBSERVATION
    epsilon = INITIAL_EPSILON

    # Try to load previous model
    modelName = name + ".h5"
    if os.path.isfile(modelName):
        model.load_weights(modelName)
    metaName = name + ".json"
    if os.path.isfile(metaName):
        with open(metaName, "r") as infile:
            meta = json.load(infile)
            OBSERVE = meta["timestep"]
            epsilon = meta["epsilon"]
        
    if args['mode'] == 'Run':
        OBSERVE = 999999999  # We keep observe, never train
        epsilon = -1

    while (True):
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        # choose an action epsilon greedy
        if random.random() <= epsilon:
            print("----------Random Action----------")
            action_index = random.randrange(ACTIONS)
        else:
            q = model.predict(s_t)  # input a stack of 4 images, get the prediction
            max_Q = np.argmax(q)
            action_index = max_Q
        a_t = np.zeros([ACTIONS])
        a_t[action_index] = 1
        # run the selected action and observed next state and reward
        for i in range(FRAME_PER_ACTION):
            x_t1_colored, r_t, terminal = game_state.frame_step(a_t)

        # We reduced the epsilon gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1, (80, 80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # 1x80x80x1
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        # store the transition in D
        D.add_transition((s_t, action_index, r_t, s_t1, terminal))

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = D.get_mini_batch(BATCH)

            # Now we do the experience replay
            state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
            state_t = np.concatenate(state_t)
            state_t1 = np.concatenate(state_t1)
            targets = target_model.predict(state_t)
            Q_sa = target_model.predict(state_t1)
            targets[range(BATCH), action_t] = reward_t + GAMMA * np.max(Q_sa, axis=1) * np.invert(terminal)
            loss += model.train_on_batch(state_t, targets)

        s_t = s_t1
        t = t + 1

        # Save progress and update training model
        if t % TARGET_MODEL_UPDATE == 0:
            name = config["name"]
            model.save_weights(name + ".h5", overwrite=True)
            meta = {"timestep": t, "epsilon": epsilon}
            with open(name + ".json", "w") as outfile:
                json.dump(meta, outfile)
            target_model = clone_model(model)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
              "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
              "/ Q_MAX ", np.max(Q_sa), "/ Loss ", loss)

    print("Episode finished!")
    print("************************")


def single_step():
    global config
    import sys
    trs = TRS(config, False, 20, False)
    #trs.boot()
    #trs.boot()
    #trs.boot()
    #trs.boot()
    def step_thread():
        trs.boot()
        game_state = Game(trs)
        while True:
            left_shoot = np.zeros(ACTIONS)
            left_shoot[0] = 1
            (screenshot, reward, terminal) = game_state.frame_step(left_shoot)
            print (reward, terminal)
            sys.stdin.readline()
    thread = Thread(target=step_thread)
    thread.start()
    trs.mainloop()
    
def main():
    global config
    parser = argparse.ArgumentParser(description='TRS DeepQ Network')
    parser.add_argument('-m', '--mode', help='Train/Run/Play', required=True)
    parser.add_argument('--no-ui', help='Do not show UI during training', action='store_true')
    args = vars(parser.parse_args())
    #single_step()
    #return
    fps = 20.0
    original_speed = 1
    if args["mode"] == "Play":
        trs = TRS(config, 1, fps, args["no_ui"])
        trs.run_cpu()
        trs.mainloop()
        return
    elif args["mode"] == "Train":
        original_speed = 0
        fps = 2.0
    trs = TRS(config, original_speed, fps, args["no_ui"])
    trs.boot()

    def training_thread():
        conf = tf.ConfigProto()
        conf.gpu_options.allow_growth = True
        sess = tf.Session(config=conf)
        from keras import backend as K
        K.set_session(sess)
        model = buildmodel()
        trainNetwork(trs, model, args)

    thread = Thread(target=training_thread)
    thread.start()
    trs.mainloop()


if __name__ == "__main__":
    main()

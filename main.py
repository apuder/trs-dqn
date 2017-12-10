
from __future__ import print_function

from trs import TRS
from trs import Key
from trs import Screenshot

class RewardCosmicFighter():

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
            return (0.1, False)
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
            return (0.1, False)

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
                return (0.9, False)
        return (0.1, False)

config = {
    "name": "cosmic",
    "cmd": "var/cosmic.cmd",
    "boot": [1000000, Key.CLEAR, Key._1, 1000000, 1000000, 1678000],
    "viewport": (0, 2, 64, 14),
    "step": 70000,
    "actions": [None, [Key.SPACE], [Key.LEFT], [Key.LEFT, Key.SPACE],
                [Key.RIGHT], [Key.RIGHT, Key.SPACE], [Key.D]],
    "reward": RewardCosmicFighter
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

import json
from keras.models import Sequential
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
REPLAY_MEMORY = 1000000  # number of previous transitions to remember
BATCH = 32  # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 0.00025

img_rows, img_cols = 80, 80
# Convert image into Black and white
img_channels = 4  # We stack 4 frames


def buildmodel():
    print("Now we build the model")
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

    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse', optimizer=adam)
    print("We finish building the model")
    return model


def trainNetwork(trs, model, args):
    # open up a game state to communicate with emulator
    game_state = Game(trs)

    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)

    print(x_t.shape)
    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t, (80, 80))
    x_t = skimage.exposure.rescale_intensity(x_t, out_range=(0, 255))

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    print (s_t.shape)

    # In Keras, need to reshape
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 1*80*80*4

    if args['mode'] == 'Run':
        OBSERVE = 999999999  # We keep observe, never train
        epsilon = -1
        print("Now we load weight")
        model.load_weights(config["name"] + ".h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        print("Weight load successfully")
    else:  # We go to training mode
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON

    t = 0
    last_action = 0
    while (True):
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([ACTIONS])
        # choose an action epsilon greedy
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
            else:
                q = model.predict(s_t)  # input a stack of 4 images, get the prediction
                max_Q = np.argmax(q)
                action_index = max_Q
            last_action = action_index
        else:
            action_index = last_action
        a_t[action_index] = 1
        # We reduced the epsilon gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observed next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)

        x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1, (80, 80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # 1x80x80x1
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        # store the transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))  # 32, 80, 80, 4
            #print(inputs.shape)
            targets = np.zeros((inputs.shape[0], ACTIONS))  # 32, 2

            # Now we do the experience replay
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]  # This is action index
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                terminal = minibatch[i][4]
                # if terminated, only equals reward

                inputs[i:i + 1] = state_t  # I saved down s_t

                targets[i] = model.predict(state_t)  # Hitting each buttom probability
                Q_sa = model.predict(state_t1)

                if terminal:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

            # targets2 = normalize(targets)
            loss += model.train_on_batch(inputs, targets)

        s_t = s_t1
        t = t + 1

        # save progress every 10000 iterations
        if t % 1000 == 0:
            print("Now we save model")
            model.save_weights(config["name"] + ".h5", overwrite=True)
            with open(config["name"] + ".json", "w") as outfile:
                json.dump(model.to_json(), outfile)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
              "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
              "/ Q_MAX ", np.max(Q_sa), "/ Loss ", loss)

    print("Episode finished!")
    print("************************")


def playGame(trs, args):
    model = buildmodel()
    trainNetwork(trs, model, args)


def main():
    global config
    parser = argparse.ArgumentParser(description='TRS DeepQ Network')
    parser.add_argument('-m', '--mode', help='Train/Run/Play', required=True)
    parser.add_argument('--no-ui', help='Do not show UI during training', action='store_true')
    args = vars(parser.parse_args())
    fps = 20.0
    original_speed = 1
    if args["mode"] == "Play":
        trs = TRS(config, 1, fps, args["no_ui"])
        trs.run()
        return
    elif args["mode"] == "Train":
        original_speed = 0
        fps = 2.0
    trs = TRS(config, original_speed, fps, args["no_ui"])
    trs.boot()
    playGame(trs, args)


if __name__ == "__main__":
    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    sess = tf.Session(config=conf)
    from keras import backend as K

    K.set_session(sess)
    main()

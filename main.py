from __future__ import print_function

from trs import TRS
from trs import Key
from trs import Screenshot

import argparse
import skimage as skimage
from skimage import transform, color, exposure
from threading import Thread
from perf_timer import PerformanceTimerImpl, PerformanceTimer

import tensorflow as tf

from absl import logging as log

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import sys

perf = PerformanceTimerImpl()


class RewardCosmicFighter:

    default_reward = (-0.2, False)

    def __init__(self, ram):
        self.ram = ram
        self.score = 0

    def reset(self):
        self.score = 0

    def compute(self):
        b = bytearray()
        for i in range(15):
            b.append(self.ram.peek(0x3C00 + i))

        if 0x20 not in b:
            # Score not shown right now
            # if 0xbf in b:
            # Evil eye appears to be attacking
            # return (-0.2, False)
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

        if b.count(b"\x5b") == 2:
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
    "actions": [
        None,
        [Key.SPACE],
        [Key.LEFT],
        [Key.LEFT, Key.SPACE],
        [Key.RIGHT],
        [Key.RIGHT, Key.SPACE],
        # , [Key.D]
    ],
    "reward": RewardCosmicFighter,
}


class RewardBreakdown:

    default_reward = (0.0, False, False)  # (Reward, Lost life, Game Over)

    def __init__(self, ram):
        self.ram = ram
        self.score = 0

    def reset(self):
        self.score = 0

    def compute(self):
        b = bytearray()
        for i in range(5):
            b.append(self.ram.peek(0x3C00 + 6 + i))

        # Get score
        try:
            new_score = int(b[:])
        except (ValueError, IndexError):
            # Score was not fully rendered yet
            return RewardBreakdown.default_reward

        ch = self.ram.peek(0x3C00 + 673)
        if ch == ord("O"):
            return (0.0, True, True)  # Game Over
        if ch in [ord("B"), ord("H")]:  # Lost life: Pass Ball/That Hurts
            return (0.0, True, False)

        delta = new_score - self.score
        if delta != 0:
            # Score increased
            self.score = new_score
            return (1.0, False, False)
        return RewardBreakdown.default_reward


config = {
    "name": "breakdown",
    "cmd": "var/breakdown.cmd",
    "boot": [
        1000000,
        1000000,
        1000000,
        Key.SPACE,
        1000000,
        1000000,
        1000000,
        1000000,
        1000000,
        1000000,
        800000,
    ],
    "viewport": (0, 2, 64, 14),
    "step": 50000,
    "actions": [None, [Key.LEFT], [Key.RIGHT], [Key.SPACE]],
    "reward": RewardBreakdown,
}


class Game:

    def __init__(self, trs):
        self.trs = trs
        self.config = trs.config
        self.reward = self.config["reward"](trs.ram)
        self.steps = self.config["step"]
        self.actions = self.config["actions"]
        self.action_repeat = 4  # Repeat the same action N times like DeepMind
        viewport = self.config["viewport"]
        self.screenshot = Screenshot(trs.ram, viewport)
        self.steps_survived = 0
        self.reset()

    def reset(self):
        self.trs.boot()
        self.reward.reset()
        self.delta_tstates = 0
        self.steps_survived = 0

        x_t, r_0, terminal, _ = self.frame_step(0)
        x_t = skimage.transform.resize(x_t, (84, 84))
        self.state = np.stack((x_t, x_t, x_t, x_t), axis=2)
        return self.state

    def frame_step(self, action):
        total_reward = 0
        terminal = False
        game_over = False
        last_screenshot = None

        for _ in range(self.action_repeat):
            self.trs.keyboard.all_keys_up()
            keys = self.actions[action]
            if keys:
                for key in keys:
                    self.trs.keyboard.key_down(key)
            tstates = self.steps - self.delta_tstates
            self.delta_tstates = self.trs.run_for_tstates(tstates)
            reward, term, over = self.reward.compute()

            self.steps_survived += 1
            reward += min(0.001 * self.steps_survived, 0.5)

            total_reward += reward
            terminal = terminal or term
            game_over = game_over or over

            last_screenshot = self.screenshot.screenshot()

            if game_over or terminal:
                break

        return (last_screenshot, total_reward, terminal, game_over)

    def step(self, action):
        screenshot, reward, terminal, game_over = self.frame_step(action)

        x_t1 = skimage.transform.resize(screenshot, (84, 84))
        x_t1 = x_t1.reshape(x_t1.shape[0], x_t1.shape[1], 1)
        self.state = np.append(x_t1, self.state[:, :, :3], axis=2)

        return self.state, reward, terminal or game_over, None
# ------------------------------------------------------------------------------------
# The following is adopted from https://keras.io/examples/rl/deep_q_network_breakout/
# ------------------------------------------------------------------------------------


def create_q_model():
    inputs = layers.Input(shape=(84, 84, 4))
    x = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    x = layers.Conv2D(64, 4, strides=2, activation="relu")(x)
    x = layers.Conv2D(64, 3, strides=1, activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    outputs = layers.Dense(len(config["actions"]), activation="linear")(x)
    return keras.Model(inputs=inputs, outputs=outputs)

def train_network(env):
    seed = 42
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_max = 1.0
    epsilon_interval = epsilon_max - epsilon_min
    batch_size = 32
    max_steps_per_episode = 10000
    min_replay_history = 10000

    # The first model makes the predictions for Q-values which are used to
    # make a action.
    model = create_q_model()
    # Build a target model for the prediction of future rewards.
    # The weights of a target model get updated every 10000 steps thus when the
    # loss between the Q-values is calculated the target Q-value is stable.
    model_target = create_q_model()

    """
    ## Train
    """
    # In the Deepmind paper they use RMSProp however then Adam optimizer
    # improves training time
    optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

    # Experience replay buffers
    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    done_history = []
    episode_reward_history = []
    running_reward = 0
    episode_count = 0
    frame_count = 0
    # Number of frames to take random action and observe output
    epsilon_random_frames = 50000
    # Number of frames for exploration
    epsilon_greedy_frames = 1000000.0
    # Maximum replay length
    # Note: The Deepmind paper suggests 1000000 however this causes memory issues
    max_memory_length = 100000
    # Train the model after 4 actions
    update_after_actions = 4
    # How often to update the target network
    update_target_network = 10000
    # Using huber loss for stability
    loss_function = keras.losses.Huber()

    state = np.array(env.reset())

    while True:
        episode_reward = 0

        for timestep in range(1, max_steps_per_episode):
            # env.render(); Adding this line would show the attempts
            # of the agent in a pop up window.
            frame_count += 1
            log.info("Frame #%d", frame_count)

            # Use epsilon-greedy for exploration
            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                # Take random action
                action = np.random.choice(len(config["actions"]))
            else:
                # Predict action Q-values
                # From environment state
                perf.quick_start()
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = model(state_tensor, training=False)
                # Take best action
                action = tf.argmax(action_probs[0]).numpy()
                perf.quick_end("Predict action Q-values")

            # Decay probability of taking random action
            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)

            # Apply the sampled action in our environment
            state_next, reward, done, _ = env.step(action)
            state_next = np.array(state_next)

            episode_reward += reward

            # Save actions and states in replay buffer
            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(done)
            rewards_history.append(reward)
            state = state_next

            # Update every fourth frame and once batch size is over 32
            if (
                frame_count > min_replay_history
                and frame_count % update_after_actions == 0
                and len(done_history) > batch_size
            ):

                # Get indices of samples for replay buffers
                indices = np.random.choice(range(len(done_history)), size=batch_size)

                # Using list comprehension to sample from replay buffer
                state_sample = np.array([state_history[i] for i in indices])
                state_next_sample = np.array([state_next_history[i] for i in indices])
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = tf.convert_to_tensor([float(done_history[i]) for i in indices])

                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                perf.quick_start()
                future_rewards = model_target.predict(state_next_sample)
                perf.quick_end("Predict")
                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1) * (1 - done_sample)

                # Create a mask so we only calculate loss on the updated Q-values
                masks = tf.one_hot(action_sample, len(config["actions"]))

                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    q_values = model(state_sample)

                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = loss_function(updated_q_values, q_action)

                # Backpropagation
                perf.quick_start()
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                perf.quick_end("Back Prop")

            if frame_count % update_target_network == 0:
                # update the the target network with new weights
                model_target.set_weights(model.get_weights())
                print(f"running reward: {running_reward:.2f} at episode {episode_count}, frame count {frame_count}")

            # Save progress and update training model
            if frame_count % 100_000 == 0:
                name = config["name"]
                model.save_weights(name + f"-{frame_count}.weights.h5", overwrite=True)
                print(f"==> Progress saved. Frame count: {frame_count}, Running rewards: {running_reward}")

            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            if done:
                print("==> DONE")
                break

        # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)
        reward_counts = {
            "positive": sum(1 for r in rewards_history if r > 0),
            "negative": sum(1 for r in rewards_history if r < 0),
            "zero": sum(1 for r in rewards_history if r == 0),
        }
        print(f"==> Episode {episode_count}, Reward stats: {reward_counts}, Running reward: {running_reward:.2f}")
        episode_count += 1
        state = np.array(env.reset())
        
        if running_reward > 200:
            print("Solved at episode {}!".format(episode_count))
            break


# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------

import os


def run(modelName, env):
    if not os.path.isfile(modelName):
        print("Model " + modelName + " not found!")
        return
    model = create_q_model()
    model.load_weights(modelName)

    while True:
        state = np.array(env.reset())
        while True:
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()
            state, reward, done, _ = env.step(action)
            state = np.array(state)


def single_step():
    global config
    import sys

    trs = TRS(config, False, 20, False)

    # trs.boot()
    # trs.boot()
    # trs.boot()
    # trs.boot()
    def step_thread():
        trs.boot()
        game_state = Game(trs)
        while True:
            (screenshot, reward, terminal) = game_state.frame_step(0)
            print(reward, terminal)
            sys.stdin.readline()

    thread = Thread(target=step_thread)
    thread.start()
    trs.mainloop()


def main():
    log.set_verbosity(log.DEBUG)

    global config
    parser = argparse.ArgumentParser(description="TRS DeepQ Network")
    parser.add_argument("-m", "--mode", help="Train/Run/Play/Single", required=True)
    parser.add_argument("--model", help="Model filename")
    parser.add_argument(
        "--no-ui", help="Do not show UI during training", action="store_true"
    )
    args = vars(parser.parse_args())
    if args["mode"] == "Single":
        single_step()
        return
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
    elif args["mode"] == "Run":
        modelName = args["model"]
        if modelName == None:
            print("Missing model filename")
            return
        trs = TRS(config, 1, fps, False)
        game = Game(trs)

        def running_thread():
            run(modelName, game)

        thread = Thread(target=running_thread)
        thread.start()
        trs.mainloop()
        return

    trs = TRS(config, original_speed, fps, args["no_ui"])

    game = Game(trs)

    def training_thread():
        train_network(game)

    thread = Thread(target=training_thread)
    thread.start()
    trs.mainloop()


if __name__ == "__main__":
    main()

from __future__ import print_function

from trs import TRS
from trs import Key
from trs import Screenshot

import argparse
import skimage as skimage
from skimage import transform, color, exposure
from threading import Thread
#from perf_timer import PerformanceTimerImpl, PerformanceTimer

import tensorflow as tf

from absl import logging as log

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts, LearningRateSchedule
import csv, os, time, math, sys


#perf = PerformanceTimerImpl()


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


class RewardBreakdownOrig:

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


class RewardBreakdown:

    default_reward = (0.0, False, False)  # (Reward, Lost life, Game Over)

    def __init__(self, ram):
        self.ram = ram
        self.score = 0

    def reset(self):
        self.score = 0
        self.lives = -1
        self.ycount = 1

    def compute(self, pc):
        if pc == 0x52A4:
            new_score = self.ram.peek(0x6314) + 256 * self.ram.peek(0x6315)
            if new_score > self.score:
                # Score increased
                #log.info('Score increased')
                self.score = new_score
                return (1.0, False, False)
            new_ycount = self.ram.peek(0x5ea1)
            if new_ycount != self.ycount:
                self.ycount = new_ycount
                if new_ycount & 0x80: # Y count is negative, ball was reflected
                    #log.info('Ball reflected')
                    return (2.0, False, False)
                return (0.0, False, False)
            return RewardBreakdown.default_reward
        if pc == 0x5d17:
            return (-1.0, False, True)  # Game Over
        # Breakpoints 0x5CA4, 0x5C57 indicate that the player lost a life
        return (-1.0, True, False) # Lost life


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
    "viewport": (0, 1, 64, 15),
    #"step": 50000,
    "breakpoints": [0x52A4, 0x5d17, 0x5CA4, 0x5C57],
    "actions": [None, [Key.LEFT], [Key.RIGHT]],
    "biased_weights": [0.50, 0.10, 0.40],
    "reward": RewardBreakdown,
}


class Game:

    def __init__(self, trs):
        self.trs = trs
        self.config = trs.config
        self.reward = self.config["reward"](trs.ram)
        self.steps = self.config.get("step", None)
        self.breakpoints = self.config.get("breakpoints", None)
        self.actions = self.config["actions"]
        self.action_repeat = 1  # Repeat the same action N times (DeepMind uses 4)
        viewport = self.config["viewport"]
        self.screenshot = Screenshot(trs.ram, viewport)
        self.steps_survived = 0
        if self.breakpoints is not None:
            self.trs.z80.clear_breakpoints()
            for bp in self.breakpoints:
                self.trs.z80.add_breakpoint(bp)
        self.reset()

    def reset(self):
        self.trs.boot()
        self.reward.reset()
        self.delta_tstates = 0
        self.steps_survived = 0

        x_t, r_0, terminal, _ = self.frame_step(0)
        #x_t = skimage.transform.resize(x_t, (84, 84))
        self.state = np.stack((x_t, x_t), axis=2)
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
            if self.steps is not None:
              tstates = self.steps - self.delta_tstates
              self.delta_tstates = self.trs.run_for_tstates(tstates)
            else:
              pc = self.trs.resume()
            #print(f"running reward: {running_reward:.2f} at episode {episode_count}, frame count {frame_count}")

            reward, term, over = self.reward.compute(pc)
            if term:
                self.trs.keyboard.key_down(Key.ENTER)
                self.trs.resume()
                self.trs.keyboard.all_keys_up()

            #if not term and not over:
            #    self.steps_survived += 1
            #    reward += min(0.05 * self.steps_survived, 0.5)

            total_reward += reward
            terminal = terminal or term
            game_over = game_over or over

            last_screenshot = self.screenshot.screenshot()

            if game_over or terminal:
                break

        return (last_screenshot, total_reward, terminal, game_over)

    def step(self, action):
        screenshot, reward, terminal, game_over = self.frame_step(action)

        #x_t1 = skimage.transform.resize(screenshot, (84, 84))
        x_t1 = screenshot.reshape(screenshot.shape[0], screenshot.shape[1], 1)
        self.state = np.append(x_t1, self.state[:, :, :1], axis=2)

        return self.state, reward, terminal, game_over
# ------------------------------------------------------------------------------------
# The following is adopted from https://keras.io/examples/rl/deep_q_network_breakout/
# ------------------------------------------------------------------------------------


def create_q_model():
    inputs = keras.Input(shape=(48, 128, 2))  # height, width, channels

    x = layers.Conv2D(32, kernel_size=(8, 4), strides=(4, 2), activation="relu")(inputs)
    x = layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation="relu")(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation="relu")(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)

    outputs = layers.Dense(len(config["actions"]), activation="linear")(x)

    return keras.Model(inputs=inputs, outputs=outputs)

def train_network(env):
    seed = 42
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_max = 1.0
    epsilon_interval = epsilon_max - epsilon_min
    batch_size = 64
    max_steps_per_episode = 10000
    min_replay_history = 10_000

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
    epsilon_random_frames = 10_000
    # Number of frames for exploration
    epsilon_greedy_frames = 750000.0
    # Maximum replay length
    # Note: The Deepmind paper suggests 1000000 however this causes memory issues
    max_memory_length = 20000
    # Train the model after n actions
    update_after_actions = 1
    # How often to update the target network
    update_target_network = 2_000
    # Using huber loss for stability
    loss_function = keras.losses.Huber()

    # --- Learning rate: hold at 1e-4 until ε <= 0.5, then cosine restarts on *update* steps ---
    class HoldThenCosine(LearningRateSchedule):
        """Keeps LR flat at `initial_lr` until `start_update` is set, then runs a wrapped schedule
        using (step - start_update) as the phase."""
        def __init__(self, initial_lr, inner_sched):
            self.initial_lr = tf.convert_to_tensor(initial_lr, tf.float32)
            self.inner_sched = inner_sched
            # Large sentinel so we 'hold' until explicitly started
            self.start_update = tf.Variable(2**31 - 1, dtype=tf.int64, trainable=False)
        def start_at(self, update_step: int):
            self.start_update.assign(tf.cast(update_step, tf.int64))
        def __call__(self, step):
            step = tf.cast(step, tf.int64)
            offset = tf.maximum(0, step - self.start_update)
            # CosineDecayRestarts accepts numeric steps; cast offset for safety
            lr_cos = tf.cast(self.inner_sched(tf.cast(offset, tf.float32)), tf.float32)
            return tf.where(step < self.start_update, self.initial_lr, lr_cos)
        def get_config(self):
            # Minimal config to satisfy Keras serialization if needed
            return {"initial_lr": float(self.initial_lr.numpy())}


    initial_lr = 1e-4
    # Key the cosine cycle length to *updates*, not frames
    updates_per_frame = 1.0 / update_after_actions
    U_updates = max(0, (epsilon_greedy_frames - max(min_replay_history, epsilon_random_frames))) * updates_per_frame
    first_decay_steps = int(0.65 * U_updates)
    cosine_inner = CosineDecayRestarts(
        initial_learning_rate=initial_lr,
        first_decay_steps=first_decay_steps,
        t_mul=1.4, m_mul=0.9, alpha=0.30
    )
    lr_schedule = HoldThenCosine(initial_lr, cosine_inner)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
    cosine_started = False

    action_counts = [0] * len(env.config["actions"])
    
    # ---------- CSV LOGGING SETUP ----------
    reward_stats = {
      "bounce": 0,
      "game_over": 0,
      "lost_life": 0,
      "neutral": 0,
      "shaping": 0
    }
    csv_path = "logs.csv"
    csv_fields = [
        # bookkeeping
        "event", "wall_time_s", "frame", "updates",
        # schedules
        "epsilon", "lr",
        # optimizer / td update
        "td_loss", "td_ema", "grad_norm",
        "target_mean", "target_std", "target_min", "target_max",
        "q_action_mean", "q_action_std",
        # action selection (cumulative)
        "action0_count", "action1_count", "action2_count",
        "path_stick", "path_rand_biased", "path_rand", "path_greedy",
        # replay tail (last ~10k frames or all if smaller)
        "reward_mean", "reward_std", "reward_p10", "reward_p50", "reward_p90", "term_ratio",
        # reward counters (cumulative)
        "r_bounce", "r_game_over", "r_lost_life", "r_neutral", "r_shaping",
        # probe set (fixed 256 samples once buffer is warm)
        "probe_online_mean", "probe_online_std",
        "probe_target_mean", "probe_target_std",
        "probe_gap_mean",
        "probe_argmax_0", "probe_argmax_1", "probe_argmax_2",
        "target_sync_updates",
    ]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()
    csv_file.flush()

    # --- rolling diagnostics state ---
    td_ema = None
    last_td = math.nan
    last_gn = math.nan
    last_t_mean = last_t_std = last_t_min = last_t_max = [math.nan]*4
    last_q_mean = last_q_std = math.nan

    # action-path attribution (counts are cumulative)
    action_path = {"stick": 0, "rand_biased": 0, "rand": 0, "greedy": 0}

    # fixed probe set will be created once when buffer is warm
    probe_states = None
    # last observed optimizer-step at which we synced target
    last_target_sync_updates = -1
    probe_stats = {
        "on_mean": math.nan, "on_std": math.nan,
        "tg_mean": math.nan, "tg_std": math.nan,
        "gap": math.nan, "argmax": [0]*len(config["actions"])
    }

    def _replay_tail_stats():
        # last ~10k rewards (or all if shorter)
        rh = rewards_history[-10000:] if len(rewards_history) > 10000 else rewards_history
        dh = done_history[-10000:]     if len(done_history)     > 10000 else done_history
        if len(rh) == 0:
            return (math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)
        import numpy as np
        r = np.array(rh, dtype=np.float32)
        d = np.array(dh, dtype=np.float32)
        return (float(r.mean()), float(r.std()),
                float(np.percentile(r, 10)), float(np.median(r)), float(np.percentile(r, 90)),
                float(d.mean()))

    def _emit_csv_row(event_label):
        # optimizer state
        updates = int(optimizer.iterations.numpy()) if hasattr(optimizer, "iterations") else 0
        # current LR
        try:  # preferred when using lr_var
            curr_lr = float(lr_var.numpy())
        except Exception:  # fallback for schedule-based LR
            try:
                curr_lr = float(optimizer.learning_rate(updates).numpy())
            except Exception:
                try:
                    curr_lr = float(optimizer.learning_rate.numpy())
                except Exception:
                    curr_lr = math.nan

        # action counts (3 actions in your config)
        a0 = action_counts[0] if len(action_counts) > 0 else 0
        a1 = action_counts[1] if len(action_counts) > 1 else 0
        a2 = action_counts[2] if len(action_counts) > 2 else 0

        # replay tail snapshot
        r_mean, r_std, r_p10, r_p50, r_p90, term_ratio = _replay_tail_stats()

        # event: "update" (every ~500 optimizer steps) or "tick" (every 10k frames)
        # wall_time_s: time.time() at log moment
        # frame, updates: env frames and optimizer iterations
        # epsilon, lr: current ε and evaluated learning rate
        # td_loss, td_ema, grad_norm: current TD loss, EMA(0.99), and global grad-norm
        # target_mean/std/min/max: stats of Bellman targets in the last update batch
        # q_action_mean/std: stats of the Q(s,a) selected by the batch actions
        # action0/1/2_count: cumulative action counts
        # path_stick/rand_biased/rand/greedy: cumulative action-path attributions
        # reward_mean/std/p10/p50/p90, term_ratio: replay tail (last ~10k)
        # r_bounce / r_game_over / r_lost_life / r_neutral / r_shaping: reward counters
        # probe_online_mean/std / probe_target_mean/std / probe_gap_mean: value calibration on the fixed probe set
        # probe_argmax_0/1/2: action distribution of the online net on the probe set
        row = {
            "event": event_label,
            "wall_time_s": time.time(),
            "frame": frame_count,
            "updates": updates,
            "epsilon": float(epsilon),
            "lr": curr_lr,
            "td_loss": last_td,
            "td_ema": (float(td_ema) if td_ema is not None else math.nan),
            "grad_norm": last_gn,
            "target_mean": last_t_mean,
            "target_std": last_t_std,
            "target_min": last_t_min,
            "target_max": last_t_max,
            "q_action_mean": last_q_mean,
            "q_action_std": last_q_std,
            "action0_count": a0, "action1_count": a1, "action2_count": a2,
            "path_stick": action_path["stick"],
            "path_rand_biased": action_path["rand_biased"],
            "path_rand": action_path["rand"],
            "path_greedy": action_path["greedy"],
            "reward_mean": r_mean, "reward_std": r_std,
            "reward_p10": r_p10, "reward_p50": r_p50, "reward_p90": r_p90,
            "term_ratio": term_ratio,
            "r_bounce": reward_stats["bounce"],
            "r_game_over": reward_stats["game_over"],
            "r_lost_life": reward_stats["lost_life"],
            "r_neutral": reward_stats["neutral"],
            "r_shaping": reward_stats["shaping"],
            "probe_online_mean": probe_stats["on_mean"],
            "probe_online_std": probe_stats["on_std"],
            "probe_target_mean": probe_stats["tg_mean"],
            "probe_target_std": probe_stats["tg_std"],
            "probe_gap_mean": probe_stats["gap"],
            "probe_argmax_0": probe_stats["argmax"][0] if len(probe_stats["argmax"]) > 0 else 0,
            "probe_argmax_1": probe_stats["argmax"][1] if len(probe_stats["argmax"]) > 1 else 0,
            "probe_argmax_2": probe_stats["argmax"][2] if len(probe_stats["argmax"]) > 2 else 0,
            "target_sync_updates": last_target_sync_updates,
        }
        csv_writer.writerow(row)
        csv_file.flush()
    # --------------------------------------

    state = np.array(env.reset())

    while True:
        episode_reward = 0
        last_action = 0

        for timestep in range(1, max_steps_per_episode):
            # env.render(); Adding this line would show the attempts
            # of the agent in a pop up window.
            frame_count += 1
            #log.info("==> Frame #%d", frame_count)

            if np.random.rand() < 0.25:
              action = last_action
              action_path["stick"] += 1
            else:
              # Use epsilon-greedy for exploration
              if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                  # Take random action
                  if episode_count < 100:
                    # for the first 100 episodes, use biased action to the right
                    action = np.random.choice(len(env.config["actions"]), p=env.config["biased_weights"])
                    action_path["rand_biased"] += 1
                  else: 
                    action = np.random.choice(len(env.config["actions"]))
                    action_path["rand"] += 1

              else:
                  # Predict action Q-values
                  # From environment state
                  #perf.quick_start()
                  state_tensor = tf.convert_to_tensor(state)
                  state_tensor = tf.expand_dims(state_tensor, 0)
                  action_probs = model(state_tensor, training=False)
                  # Take best action
                  action = tf.argmax(action_probs[0]).numpy()
                  #perf.quick_end("Predict action Q-values")
                  action_path["greedy"] += 1

            last_action = action

            # Decay probability of taking random action
            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)
            # Start cosine schedule the moment ε <= 0.5 (once), keyed to optimizer updates
            if (not cosine_started) and (epsilon <= 0.5):
                cosine_started = True
                lr_schedule.start_at(int(optimizer.iterations.numpy()))
            # (No per-step LR assignments needed; optimizer reads from lr_schedule(step))

            # --- LR control: flat until epsilon <= 0.5, then cosine restarts keyed to updates ---
            if not cosine_started and epsilon <= 0.5:
                cosine_started = True
                cosine_start_updates = int(optimizer.iterations.numpy())

            # Apply the sampled action in our environment
            state_next, reward, _, game_over = env.step(action)
            state_next = np.array(state_next)

            episode_reward += reward

            # Save actions and states in replay buffer
            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(game_over)
            rewards_history.append(reward)
            state = state_next

            action_counts[action] += 1
            if reward > 1.0:
              reward_stats["bounce"] += 1
            elif reward < -0.9:
              reward_stats["game_over"] += 1
            elif reward == 0.0:
              reward_stats["neutral"] += 1
            elif reward > 0:
              reward_stats["shaping"] += 1
            else:
              reward_stats["lost_life"] += 1

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
                #perf.quick_start()
                future_rewards = model_target.predict(state_next_sample)
                #perf.quick_end("Predict")
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
                #perf.quick_start()
                last_td = float(loss.numpy())
                td_ema = last_td if td_ema is None else (0.99 * td_ema + 0.01 * last_td)
                grads = tape.gradient(loss, model.trainable_variables)
                last_gn = float(tf.linalg.global_norm(grads).numpy())
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                #perf.quick_end("Back Prop")
                uq = np.asarray(updated_q_values, dtype=np.float32)
                qa = np.asarray(q_action.numpy(), dtype=np.float32)
                last_t_mean, last_t_std, last_t_min, last_t_max = float(uq.mean()), float(uq.std()), float(uq.min()), float(uq.max())
                last_q_mean, last_q_std = float(qa.mean()), float(qa.std())

                # emit a CSV row every 500 optimizer updates
                if int(optimizer.iterations.numpy()) % 500 == 0:
                    _emit_csv_row("update")

            # Build a fixed probe set once (after buffer warms up)
            if probe_states is None and len(state_history) > 2000:
                idx = np.random.choice(range(len(state_history)), size=256, replace=False)
                probe_states = np.array([state_history[i] for i in idx])

            # Every 10k frames: evaluate probe, emit a CSV "tick"
            if frame_count % 10000 == 0:
                if probe_states is not None:
                    qs_online = model.predict(probe_states, verbose=0)
                    qs_target = model_target.predict(probe_states, verbose=0)
                    a_online = np.argmax(qs_online, axis=1)
                    probe_stats["on_mean"] = float(qs_online.mean())
                    probe_stats["on_std"]  = float(qs_online.std())
                    probe_stats["tg_mean"] = float(qs_target.mean())
                    probe_stats["tg_std"]  = float(qs_target.std())
                    probe_stats["gap"]     = float(np.mean(np.abs(qs_online - qs_target)))
                    probe_stats["argmax"]  = np.bincount(a_online, minlength=len(config["actions"])).tolist()

                _emit_csv_row("tick")

            if frame_count % update_target_network == 0:
                # update the target network *after* logging so probe_gap reflects pre-sync drift
                model_target.set_weights(model.get_weights())
                last_target_sync_updates = int(optimizer.iterations.numpy())
                _emit_csv_row("target_sync")

            # Save progress and update training model
            if frame_count % 100_000 == 0:
                name = config["name"]
                model.save_weights(name + f"-{frame_count}.weights.h5", overwrite=True)
                #log.info(f"==> Progress saved. Frame count: {frame_count}, Running rewards: {running_reward}")

            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            if game_over:
                #log.info("==> Game Over")
                break

        # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)
        #log.info(f"==> Episode {episode_count}, Running reward: {running_reward:.2f}, Episode reward: {episode_reward:.2f}")
        episode_count += 1
        state = np.array(env.reset())
        
        if running_reward > 10000:
            #log.info("Solved at episode {}!".format(episode_count))
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
            state, reward, _, game_over = env.step(action)
            if game_over:
                break
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
            (screenshot, reward, terminal, game_over) = game_state.frame_step(0)
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

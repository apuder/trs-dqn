# TRS-80 DQN Project: Status and Plan (2025)

This project replicates DeepMind's DQN work using a custom TRS-80 emulator to train neural networks to play classic games like Breakdown (Breakout-style brick breaker).

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [How the Code Works](#how-the-code-works)
4. [Current Problems & Solutions](#current-problems--solutions)
5. [Performance Optimization](#performance-optimization)
6. [Vectorization Strategy](#vectorization-strategy)

---

## Project Overview

### Goal
Train a Deep Q-Network (DQN) to play TRS-80 games from raw pixel inputs, replicating the DeepMind Atari breakthrough but on vintage hardware emulation.

### Current State
- **Game**: Breakdown (Breakout-style game)
- **Architecture**: Dueling Double DQN with experience replay
- **Status**: Training but not converging reliably

### Key Components
```
Python (main.py) ─→ TRS Module ─→ Native Wrapper (libtrs.so) ─→ libz80 (Z80 CPU)
       │                │
       ↓                ↓
    DQN Model     Screenshot/RAM/Keyboard
   (TensorFlow)      (ctypes)
```

---

## Architecture

### Neural Network (Dueling Double DQN)

```
Input: (48, 128, 3) - 3 stacked grayscale screenshots
  │
  ├─ Conv2D(32, kernel=8×4, stride=4×2, ReLU)  → (5, 31, 32)
  ├─ Conv2D(64, kernel=4×4, stride=2×2, ReLU)  → (1, 14, 64)
  ├─ Conv2D(64, kernel=3×3, stride=1×1, ReLU)  → (1, 12, 64)
  │
  ├─ Flatten() → (768,)
  ├─ Dense(256, ReLU)
  │
  ├─ Split into two streams:
  │   ├─ Value Stream: Dense(1) → V(s)
  │   └─ Advantage Stream: Dense(num_actions) → A(s,a)
  │
  └─ Q(s,a) = V(s) + (A(s,a) - mean(A))  [Dueling decomposition]

Output: Q-values for each of 3 actions (None, Left, Right)
```

### Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 64 | Training batch size |
| `gamma` | 0.97 | Discount factor |
| `epsilon_random_frames` | 50,000 | Pure random exploration |
| `epsilon_greedy_frames` | 700,000 | Epsilon decay phase |
| `epsilon_min` | 0.10 | Minimum exploration |
| `learning_starts` | 2,000 | Frames before training begins |
| `update_target_network` | 10,000 | Steps between target syncs |
| `max_memory_length` | 400,000 | Replay buffer capacity |
| `learning_rate` | 1e-4 | Initial learning rate |

### Reward Structure

| Event | Reward | Detection Method |
|-------|--------|------------------|
| Score increase | +1.0 | Breakpoint 0x52A4 |
| Ball bounce (near paddle) | +2.0 | Y-count change detection |
| Lost life | -1.0 | Breakpoints 0x5CA4, 0x5C57 |
| Game over | -1.0 | Breakpoint 0x5d17 |
| Shaping (proximity) | +0.1 × (1 - distance/128) | Paddle-ball distance |

---

## How the Code Works

### Training Loop (`main.py:543-767`)

```python
for episode in range(episodes):
    state = game.reset()  # Boot game, get initial frame stack

    for timestep in range(max_steps_per_episode):
        # 1. Epsilon-greedy action selection
        if random() < epsilon:
            action = random_action()  # Biased early, uniform later
        else:
            action = argmax(model.predict(state))

        # 2. Execute action in emulator
        #    - Adaptive action_repeat based on ball Y-position
        #    - Runs Z80 CPU cycles until breakpoint or T-state count
        state_next, reward, terminal, game_over = game.step(action)

        # 3. Store transition in replay buffer
        replay_buffer.add(state, action, reward, state_next, terminal)

        # 4. Train (after learning_starts frames)
        if frame_count > learning_starts and len(buffer) > batch_size:
            # PER-lite: 50% important samples, 50% uniform
            batch = sample_prioritized_batch()

            # Double DQN update
            actions_online = argmax(model(next_states))
            q_targets = target_model(next_states)[actions_online]
            targets = rewards + gamma * q_targets * (1 - done)

            loss = huber_loss(model(states)[actions], targets)
            optimizer.minimize(loss)

        # 5. Update target network periodically
        if optimizer_steps % update_target_network == 0:
            target_model.set_weights(model.get_weights())
```

### Emulator Integration

**Frame Execution** (`main.py:235-268`):
1. Release all keys from previous action
2. Press new action keys (Left/Right/None)
3. Execute Z80 CPU cycles (via `trs.resume()` or `trs.run_for_tstates()`)
4. Compute reward from breakpoint triggers or RAM inspection
5. Capture screenshot from video RAM (0x3C00)
6. Return (screenshot, reward, terminal, game_over)

**Screenshot Capture** (`trs/screenshot.py`):
- Video RAM at 0x3C00 contains 64×16 character grid
- Characters rendered to pixels via TTF font
- Output: 48×128 grayscale image (3 pixels/char height, 2 pixels/char width)

---

## Current Problems & Solutions

### CRITICAL: Training Does Not Converge

#### Problem 1: Sparse and Delayed Rewards
**Severity**: Critical

**Symptoms**:
- Agent takes many random actions before receiving any positive reward
- Score increases are rare (hitting bricks requires precision)
- Long delay between action and outcome

**Current Mitigations**:
- Distance-based reward shaping (+0.1 for being close to ball)
- Bounce detection (+2.0 for successful paddle contact)

**Recommended Solutions**:
1. **Stronger reward shaping**: Increase shaping reward magnitude (0.1 → 0.3-0.5)
2. **Potential-based shaping**: Use `F(s,a,s') = γΦ(s') - Φ(s)` where Φ is potential function
3. **Curiosity-driven exploration**: Add intrinsic reward for novel states
4. **Human demonstrations**: Bootstrap with imitation learning from recorded gameplay

#### Problem 2: Inefficient Exploration
**Severity**: Critical

**Symptoms**:
- Random exploration doesn't discover rewarding states efficiently
- 50,000 random frames may not be enough to see meaningful game events
- Biased action selection helps but is a crude heuristic

**Recommended Solutions**:
1. **Increase `epsilon_random_frames`**: Try 100,000-200,000
2. **Noisy Networks**: Replace ε-greedy with NoisyNet layers for state-dependent exploration
3. **Count-based exploration**: Add bonus for visiting infrequent states
4. **Thompson Sampling**: Use uncertainty estimates for exploration
5. **Curriculum learning**: Start with easier scenarios (ball closer to paddle)

#### Problem 3: High Variance in Q-estimates
**Severity**: High

**Symptoms**:
- Probe set shows high standard deviation in Q-values
- Large gap between online and target networks
- TD loss fluctuates significantly

**Recommended Solutions**:
1. **Reduce learning rate**: Try 5e-5 or 3e-5 instead of 1e-4
2. **Increase `update_target_network`**: Try 20,000-50,000 instead of 10,000
3. **Use Polyak averaging**: Soft updates τ=0.001 instead of hard copy
4. **Gradient clipping**: Already using norm=1.0, could try 0.5
5. **Increase batch size**: 128 or 256 if memory permits

#### Problem 4: Replay Buffer Sampling Bias
**Severity**: Medium

**Symptoms**:
- Important transitions (bounces, deaths) are rare in uniform sampling
- PER-lite helps but is a simplified version

**Recommended Solutions**:
1. **Full Prioritized Experience Replay (PER)**: Priority = |TD error|^α
2. **Increase k_pos ratio**: Try 60-70% important samples instead of 50%
3. **Hindsight Experience Replay**: Relabel failed episodes with achieved goals

#### Problem 5: Network Architecture May Be Suboptimal
**Severity**: Medium

**Symptoms**:
- Conv layers may not capture relevant features from TRS-80 graphics
- Small hidden layer (256) may limit capacity

**Recommended Solutions**:
1. **Increase network capacity**: Try 512 or 1024 hidden units
2. **Add more conv layers**: Deeper network for better features
3. **Attention mechanisms**: Focus on ball and paddle regions
4. **Residual connections**: Help gradient flow

### NICE-TO-HAVE: Stability Improvements

#### Problem 6: Learning Rate Schedule
**Severity**: Low

**Current**: Cosine decay with restarts after ε ≤ 0.5

**Potential Improvements**:
- Linear warmup for first 10,000 steps
- More aggressive decay early, slower later
- Adaptive learning rate (Adam is already adaptive, but could try LAMB)

#### Problem 7: Frame Stacking
**Severity**: Low

**Current**: 3 frames stacked

**Potential Improvements**:
- Try 4 frames (standard in Atari DQN)
- Consider LSTM/GRU for temporal modeling
- Frame differencing instead of stacking

#### Problem 8: Action Repeat Logic
**Severity**: Low

**Current**: Adaptive based on ball Y-position (1-3 frames)

**Potential Improvements**:
- Fixed action repeat (4 is standard for Atari)
- Learn action repeat as part of action space

---

## Performance Optimization

### Current Bottlenecks

The code runs single-threaded and doesn't utilize all CPU cores.

#### Bottleneck 1: Serial Emulation
**Impact**: High

The Z80 emulator runs synchronously in the main training loop:
```python
# main.py:270-277
def step(self, action, action_repeat):
    for _ in range(action_repeat):
        x_t1, reward, terminal, game_over = self.frame_step(action)
        # ... (serial execution)
```

**Solutions**:
1. **Batch environment execution**: Process multiple environments in parallel
2. **Async execution**: Use `asyncio` or `concurrent.futures` for non-blocking emulation
3. **C extension optimization**: Profile `libtrs.so` for optimization opportunities

#### Bottleneck 2: Screenshot Capture
**Impact**: Medium

Font rendering and image processing are CPU-bound:
```python
# trs/screenshot.py
# Renders each character using PIL/Pillow
```

**Solutions**:
1. **Cache character bitmaps**: Pre-render all 256 possible characters
2. **Use numpy directly**: Avoid PIL overhead for known character set
3. **GPU-accelerated rendering**: Use CUDA/OpenGL for bitmap blitting

#### Bottleneck 3: Single-threaded Training
**Impact**: High

TensorFlow/Keras training uses GPU but CPU preprocessing is serial.

**Solutions**:
1. **tf.data pipeline**: Use `tf.data.Dataset.from_generator()` with prefetching
2. **Background data loading**: Separate thread for replay buffer sampling
3. **Mixed precision**: Use float16 for faster GPU computation

### Quick Wins

1. **Enable XLA compilation**:
```python
tf.config.optimizer.set_jit(True)
```

2. **Enable mixed precision**:
```python
tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

3. **Increase prefetch buffer**:
```python
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

4. **Profile with TensorBoard**:
```python
tf.profiler.experimental.start('logdir')
# ... training ...
tf.profiler.experimental.stop()
```

---

## Vectorization Strategy

Running multiple emulator instances in parallel can dramatically speed up training by collecting more experience per unit time.

### Approach 1: Synchronous Vectorized Environments

**Concept**: Run N emulators in parallel, collect N transitions per step.

**Implementation**:

```python
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

class VectorizedTRSEnv:
    def __init__(self, num_envs, config):
        self.num_envs = num_envs
        self.envs = [Game(config) for _ in range(num_envs)]

    def reset(self):
        """Reset all environments, return stacked states."""
        states = [env.reset() for env in self.envs]
        return np.stack(states)  # (num_envs, 48, 128, 3)

    def step(self, actions):
        """Execute actions in all environments."""
        # Parallel execution using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=self.num_envs) as executor:
            futures = [
                executor.submit(env.step, action, repeat)
                for env, action, repeat in zip(self.envs, actions, repeats)
            ]
            results = [f.result() for f in futures]

        states = np.stack([r[0] for r in results])
        rewards = np.array([r[1] for r in results])
        terminals = np.array([r[2] for r in results])
        game_overs = np.array([r[3] for r in results])

        return states, rewards, terminals, game_overs

# Training loop modification
vec_env = VectorizedTRSEnv(num_envs=8, config=config)
states = vec_env.reset()

for step in range(total_steps):
    # Batch inference
    q_values = model.predict(states)  # (8, 3)
    actions = epsilon_greedy_batch(q_values, epsilon)

    # Parallel execution
    next_states, rewards, terminals, game_overs = vec_env.step(actions)

    # Store all transitions
    for i in range(num_envs):
        replay_buffer.add(states[i], actions[i], rewards[i],
                         next_states[i], terminals[i])

    states = next_states
```

**Pros**:
- Simple to implement
- Predictable timing
- Easy to debug

**Cons**:
- Slowest environment determines step time
- ProcessPoolExecutor has overhead for short tasks

### Approach 2: Asynchronous Environments (A3C-style)

**Concept**: Each environment runs in its own process, sends transitions to central learner.

**Implementation**:

```python
import multiprocessing as mp
import queue

def worker_process(env_id, config, action_queue, result_queue, model_weights_queue):
    """Worker process running one environment."""
    game = Game(config)
    state = game.reset()

    # Local model for inference (no GPU)
    local_model = create_model(num_actions=3)

    while True:
        # Check for new weights
        try:
            weights = model_weights_queue.get_nowait()
            local_model.set_weights(weights)
        except queue.Empty:
            pass

        # Get action from local model
        q_values = local_model.predict(state[np.newaxis, ...])[0]
        action = epsilon_greedy(q_values, epsilon)

        # Execute
        next_state, reward, terminal, game_over = game.step(action, action_repeat)

        # Send transition to learner
        result_queue.put((env_id, state, action, reward, next_state, terminal))

        if game_over:
            state = game.reset()
        else:
            state = next_state

class AsyncVectorizedEnv:
    def __init__(self, num_workers, config):
        self.num_workers = num_workers
        self.result_queue = mp.Queue()
        self.weight_queues = [mp.Queue() for _ in range(num_workers)]

        self.workers = []
        for i in range(num_workers):
            p = mp.Process(
                target=worker_process,
                args=(i, config, None, self.result_queue, self.weight_queues[i])
            )
            p.start()
            self.workers.append(p)

    def collect_transitions(self, n):
        """Collect n transitions from workers."""
        transitions = []
        for _ in range(n):
            transitions.append(self.result_queue.get())
        return transitions

    def broadcast_weights(self, weights):
        """Send updated weights to all workers."""
        for q in self.weight_queues:
            # Clear old weights, put new
            while not q.empty():
                try:
                    q.get_nowait()
                except:
                    pass
            q.put(weights)

# Training loop
async_env = AsyncVectorizedEnv(num_workers=8, config=config)

for step in range(total_steps):
    # Collect batch of transitions
    transitions = async_env.collect_transitions(n=64)

    # Add to replay buffer
    for t in transitions:
        replay_buffer.add(*t[1:])  # Skip env_id

    # Train
    if len(replay_buffer) > batch_size:
        batch = replay_buffer.sample(batch_size)
        loss = train_step(batch)

    # Periodically sync weights to workers
    if step % 100 == 0:
        async_env.broadcast_weights(model.get_weights())
```

**Pros**:
- Maximum throughput (no waiting for slow envs)
- Scales well with many cores
- Matches A3C/IMPALA architectures

**Cons**:
- More complex implementation
- Potential staleness in worker policies
- Debugging is harder

### Approach 3: SubprocessVecEnv (Stable-Baselines3 style)

**Concept**: Use proven vectorization patterns from RL libraries.

**Implementation**:

```python
# Option A: Use stable-baselines3 directly
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

def make_env(config, rank):
    def _init():
        return TRSGymWrapper(config)
    return _init

# Create vectorized environment
num_envs = 8
env = SubprocVecEnv([make_env(config, i) for i in range(num_envs)])

# Option B: Custom implementation using shared memory
import numpy as np
from multiprocessing import shared_memory

class SharedMemoryVecEnv:
    def __init__(self, num_envs, obs_shape, config):
        self.num_envs = num_envs
        self.obs_shape = obs_shape

        # Create shared memory for observations
        obs_size = num_envs * np.prod(obs_shape)
        self.obs_shm = shared_memory.SharedMemory(create=True, size=obs_size * 4)
        self.observations = np.ndarray(
            (num_envs,) + obs_shape,
            dtype=np.float32,
            buffer=self.obs_shm.buf
        )

        # ... similar for rewards, actions, etc.
```

### Recommended Vectorization Plan

1. **Phase 1: Basic Multiprocessing** (Immediate)
   - Wrap `Game` class with `ProcessPoolExecutor`
   - Run 4-8 environments in parallel
   - Expected speedup: 3-6x

2. **Phase 2: Shared Memory** (Short-term)
   - Use `multiprocessing.shared_memory` for zero-copy data transfer
   - Reduce serialization overhead
   - Expected additional speedup: 1.5-2x

3. **Phase 3: Async Architecture** (Medium-term)
   - Implement A3C-style async training
   - Scale to 16-32 workers
   - Expected additional speedup: 2-4x

4. **Phase 4: Distributed Training** (Long-term)
   - Use Ray or similar for multi-machine scaling
   - GPU cluster for model training
   - Unlimited horizontal scaling

### Sample Vectorized Training Code

```python
# vectorized_training.py

import multiprocessing as mp
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import tensorflow as tf

class VectorizedDQNTrainer:
    def __init__(self, config, num_envs=8):
        self.num_envs = num_envs
        self.config = config

        # Create environments
        self.envs = [Game(config) for _ in range(num_envs)]

        # Shared replay buffer
        self.replay_buffer = ReplayBuffer(max_size=400000)

        # Models
        self.model = create_model(num_actions=3)
        self.target_model = create_model(num_actions=3)
        self.target_model.set_weights(self.model.get_weights())

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    def collect_batch(self, states, epsilon):
        """Collect one step from all environments."""
        # Batch inference
        q_values = self.model.predict(states, verbose=0)

        # Epsilon-greedy action selection
        actions = np.where(
            np.random.random(self.num_envs) < epsilon,
            np.random.randint(0, 3, self.num_envs),
            np.argmax(q_values, axis=1)
        )

        # Parallel environment steps
        with ProcessPoolExecutor(max_workers=self.num_envs) as executor:
            futures = [
                executor.submit(self._env_step, i, actions[i])
                for i in range(self.num_envs)
            ]
            results = [f.result() for f in futures]

        # Unpack results
        next_states = np.stack([r[0] for r in results])
        rewards = np.array([r[1] for r in results])
        terminals = np.array([r[2] for r in results])
        game_overs = np.array([r[3] for r in results])

        # Store transitions
        for i in range(self.num_envs):
            self.replay_buffer.add(
                states[i], actions[i], rewards[i],
                next_states[i], terminals[i]
            )

        # Handle game overs (reset those environments)
        for i in range(self.num_envs):
            if game_overs[i]:
                next_states[i] = self.envs[i].reset()

        return next_states, rewards.sum(), game_overs.sum()

    def _env_step(self, env_idx, action):
        """Execute step in single environment (called in subprocess)."""
        return self.envs[env_idx].step(action, action_repeat=2)

    def train(self, total_frames=1_000_000):
        """Main training loop."""
        # Initialize states
        states = np.stack([env.reset() for env in self.envs])

        frame_count = 0
        epsilon = 1.0

        while frame_count < total_frames:
            # Decay epsilon
            epsilon = max(0.1, 1.0 - frame_count / 500000)

            # Collect batch of experience
            states, reward_sum, num_resets = self.collect_batch(states, epsilon)
            frame_count += self.num_envs

            # Train if buffer is ready
            if len(self.replay_buffer) > 1000:
                loss = self._train_step()

            # Update target network
            if frame_count % 10000 == 0:
                self.target_model.set_weights(self.model.get_weights())

            # Logging
            if frame_count % 1000 == 0:
                print(f"Frames: {frame_count}, Epsilon: {epsilon:.3f}")

    @tf.function
    def _train_step(self):
        """Single training step."""
        batch = self.replay_buffer.sample(64)
        states, actions, rewards, next_states, dones = batch

        # Double DQN target
        next_actions = tf.argmax(self.model(next_states), axis=1)
        next_q = tf.gather(
            self.target_model(next_states),
            next_actions,
            batch_dims=1
        )
        targets = rewards + 0.97 * next_q * (1 - dones)

        with tf.GradientTape() as tape:
            q_values = self.model(states)
            q_action = tf.gather(q_values, actions, batch_dims=1)
            loss = tf.keras.losses.huber(targets, q_action)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )

        return loss

if __name__ == "__main__":
    trainer = VectorizedDQNTrainer(config, num_envs=8)
    trainer.train(total_frames=2_000_000)
```

---

## Summary of Recommendations

### Immediate Actions (This Week)
1. **Increase reward shaping magnitude**: 0.1 → 0.3
2. **Reduce learning rate**: 1e-4 → 5e-5
3. **Increase target network update interval**: 10,000 → 20,000
4. **Enable XLA compilation**: `tf.config.optimizer.set_jit(True)`

### Short-term (Next 2-4 Weeks)
1. **Implement basic vectorization**: 4-8 parallel environments
2. **Add soft target updates**: Polyak averaging with τ=0.005
3. **Increase exploration**: 50,000 → 100,000 random frames
4. **Profile and optimize screenshot capture**

### Medium-term (1-2 Months)
1. **Implement full PER**: Priority-based sampling with IS correction
2. **Add NoisyNet layers**: State-dependent exploration
3. **Async training architecture**: A3C-style workers
4. **Curriculum learning**: Progressive difficulty

### Long-term
1. **Distributed training**: Ray/multi-machine
2. **Model improvements**: Attention, larger networks
3. **Human demonstrations**: Imitation learning bootstrap

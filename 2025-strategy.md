# TRS-80 DQN: Deep Reinforcement Learning for Retro Gaming

## Project Overview

This project replicates DeepMind's seminal Deep Q-Network (DQN) work, but applied to games running on a TRS-80 (Radio Shack) computer emulator. Rather than using Atari games through ALE (Arcade Learning Environment), this implementation trains a neural network to play classic TRS-80 games like **Breakdown** (a Breakout clone) directly on an emulated Z80 processor.

The system consists of three major components:
1. **Z80/TRS-80 Emulator** - A faithful emulation of the TRS-80 hardware in C
2. **Python Environment Wrapper** - A Gym-like interface for RL training
3. **DQN Training Pipeline** - Keras/TensorFlow implementation of the DeepMind algorithm

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py (DQN Training Loop)              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │   Q-Network  │    │ Target Net   │    │ Experience Replay │  │
│  │  (84×84×4)   │    │  (frozen)    │    │   (100k buffer)   │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Game Class (RL Interface)                     │
│  • reset() → initial state (84×84×4)                            │
│  • step(action) → (next_state, reward, done, info)              │
│  • Action repeat (4 frames per action)                          │
│  • Frame stacking (4 consecutive frames)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TRS Class (Emulator Core)                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │
│  │   Z80    │  │   RAM    │  │ Keyboard │  │  Screenshot  │    │
│  │  (CPU)   │  │  (64KB)  │  │ (Matrix) │  │  (84×84)     │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Native Layer (C Libraries)                      │
│  ┌────────────────────────┐    ┌────────────────────────────┐  │
│  │      libtrs.so         │    │       libz80.so            │  │
│  │  • TRS-80 hardware     │    │  • Z80 CPU emulation       │  │
│  │  • RAM management      │    │  • All opcodes             │  │
│  │  • Screenshot capture  │    │  • Registers & flags       │  │
│  └────────────────────────┘    └────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Deep-Dive

### 1. Z80/TRS-80 Emulator

The emulator faithfully recreates a TRS-80 Model I:

| Specification | Value |
|--------------|-------|
| CPU | Zilog Z80 @ 1.77408 MHz |
| RAM | 64 KB |
| Timer | 40 Hz interrupt |
| Video | 64×16 character grid |
| Video RAM | Address 0x3C00 |
| Keyboard | Memory-mapped at 0x3800 |

**Key Files:**
- `native/trs.c` - Main TRS-80 emulation driver
- `libz80/` - Complete Z80 CPU emulator (git submodule)
- `trs/z80.py` - Python ctypes wrapper for the C library

**How it works:**
```python
# Python calls into C via ctypes
z80.run_for_tstates(50000)  # Execute 50,000 T-states (clock cycles)
```

The emulator executes real Z80 machine code from game ROM files (`.cmd` format) stored in `var/`.

### 2. Python Environment (trs/ module)

The Python layer provides a clean interface to the emulator:

| Module | Purpose |
|--------|---------|
| `trs/__init__.py` | Main TRS class, orchestrates components |
| `trs/z80.py` | Z80 CPU wrapper via ctypes |
| `trs/ram.py` | 64KB RAM access (peek/poke/backup/restore) |
| `trs/keyboard.py` | Keyboard matrix emulation (46 keys) |
| `trs/screenshot.py` | Video capture to 84×84 images |
| `trs/video.py` | Optional Pyglet UI for visualization |

**Memory Map:**
```
0x0000 - 0x37FF : Program memory
0x3800 - 0x3BFF : Keyboard matrix (memory-mapped I/O)
0x3C00 - 0x3FFF : Video memory (64×16 characters = 1024 bytes)
0x4000 - 0xFFFF : Additional RAM
```

### 3. Game Environment

The `Game` class provides a Gym-like interface for RL training:

```python
class Game:
    def reset(self):
        """Boot game, return initial state (84×84×4)"""

    def step(self, action):
        """Execute action, return (state, reward, done, info)"""
```

**Frame Stacking:** Following DeepMind's approach, 4 consecutive 84×84 grayscale frames are stacked to form the state tensor (84×84×4). This provides temporal information to the network.

**Action Repeat:** Each action is repeated for 4 frames, accumulating rewards. This matches the original DQN paper methodology.

### 4. Game Configurations

Two games are currently configured:

#### Breakdown (Breakout Clone)
```python
config = {
    "name": "breakdown",
    "cmd": "var/breakdown.cmd",
    "viewport": (0, 2, 64, 14),
    "step": 50000,  # T-states per frame
    "actions": [None, [Key.LEFT], [Key.RIGHT], [Key.SPACE]],
}
```
- **Actions:** No-op, Left, Right, Fire
- **Reward:** +1.0 when score increases
- **Terminal:** Game over when all lives lost

#### Cosmic Fighter
```python
config_cosmic = {
    "name": "cosmic",
    "cmd": "var/cosmic.cmd",
    "actions": [None, [Key.SPACE], [Key.LEFT], [Key.LEFT, Key.SPACE],
                [Key.RIGHT], [Key.RIGHT, Key.SPACE]],
}
```
- **Actions:** 6 combinations of movement and fire
- **Reward:** +1.0 for enemy kills, -0.2 per frame, -1.0 on death

---

## DQN Implementation

### Network Architecture

The neural network follows the original DeepMind Atari architecture:

```
Input: 84×84×4 (4 stacked grayscale frames)
    ↓
Conv2D: 32 filters, 8×8 kernel, stride 4, ReLU → 20×20×32
    ↓
Conv2D: 64 filters, 4×4 kernel, stride 2, ReLU → 9×9×64
    ↓
Conv2D: 64 filters, 3×3 kernel, stride 1, ReLU → 7×7×64
    ↓
Flatten → 3136 units
    ↓
Dense: 512 units, ReLU
    ↓
Output: N actions (linear activation)
```

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `gamma` | 0.99 | Discount factor |
| `epsilon_start` | 1.0 | Initial exploration rate |
| `epsilon_end` | 0.1 | Final exploration rate |
| `epsilon_decay_frames` | 1,000,000 | Frames to reach final epsilon |
| `batch_size` | 32 | Training batch size |
| `learning_rate` | 0.00025 | Adam optimizer LR |
| `replay_buffer_size` | 100,000 | Max stored transitions |
| `target_update_freq` | 10,000 | Frames between target net updates |
| `train_freq` | 4 | Train every N frames |
| `min_replay_history` | 10,000 | Min samples before training |
| `warmup_frames` | 50,000 | Random actions before learning |

### Training Algorithm

```
1. Initialize replay buffer D with capacity 100,000
2. Initialize Q-network with random weights θ
3. Initialize target network with weights θ⁻ = θ

for each episode:
    Reset environment, get initial state s₀

    for each step t:
        # ε-greedy action selection
        With probability ε: select random action a
        Otherwise: a = argmax_a Q(s, a; θ)

        # Execute action (with 4-frame repeat)
        Execute a for 4 frames, observe r, s', done

        # Store transition
        Store (s, a, r, s', done) in D

        # Sample and train (every 4 frames)
        Sample minibatch of 32 transitions from D

        # Compute target Q-values
        y = r + γ × max_a' Q(s', a'; θ⁻) × (1 - done)

        # Update Q-network (Huber loss)
        Minimize (y - Q(s, a; θ))²

        # Update target network (every 10,000 frames)
        if frame_count % 10000 == 0:
            θ⁻ ← θ
```

### Reward Shaping

The latest version includes reward shaping to encourage longer gameplay:

```python
self.steps_survived += 1
reward += 0.001 * self.steps_survived
```

This provides a small, increasing bonus for staying alive, which helps bootstrap learning when game rewards are sparse.

---

## Data Flow

### Single Step Execution

```
1. Agent selects action (e.g., Key.LEFT)
       ↓
2. Keyboard.key_down(Key.LEFT)
   → Writes to RAM[0x3800+row] |= col_mask
       ↓
3. Z80.run_for_tstates(50000)
   → Executes ~28ms of game time at 1.77 MHz
   → Game reads keyboard, updates game state
   → Game updates video memory at 0x3C00
       ↓
4. Screenshot.screenshot()
   → Reads character data from RAM[0x3C00]
   → Renders using TRS-80 font
   → Resizes to 84×84 grayscale
       ↓
5. RewardFunction.compute()
   → Reads score from video memory
   → Calculates reward delta
       ↓
6. Return (state, reward, done)
```

### Experience Replay Buffer

The replay buffer stores transitions as separate arrays for memory efficiency:

```python
action_history = []      # Actions taken
state_history = []       # States (84×84×4)
state_next_history = []  # Next states
rewards_history = []     # Rewards received
done_history = []        # Terminal flags
```

When the buffer exceeds 100,000 entries, old transitions are removed (FIFO).

---

## Running the Project

### Prerequisites

```bash
# Clone with submodules
git clone --recursive <repo-url>
cd trs-dqn-2025

# Create virtual environment (Python 3.11 recommended)
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Build native libraries
export LD_LIBRARY_PATH=$(pwd)
make
```

### Execution Modes

| Mode | Command | Description |
|------|---------|-------------|
| **Train** | `python main.py -m Train --no-ui` | Train DQN from scratch |
| **Play** | `python main.py -m Play` | Human plays with keyboard |
| **Run** | `python main.py -m Run --model=<file>` | Run trained model |
| **Debug** | `python main.py -m Single` | Step-by-step execution |

### Training Output

During training, the system logs:
- Frame count and episode number
- Running reward (100-episode average)
- Epsilon value (exploration rate)
- Episode reward

Model weights are saved every 100,000 frames:
```
breakdown-100000.weights.h5
breakdown-200000.weights.h5
...
```

Training stops when running mean reward exceeds 40.

---

## Key Design Decisions

### Why T-states instead of frames?

The TRS-80 doesn't have a traditional frame buffer like Atari. Instead, execution is measured in T-states (Z80 clock cycles). Running 50,000 T-states at 1.77 MHz equals roughly 28ms of game time, providing smooth gameplay.

### Why 84×84 resolution?

This matches DeepMind's original DQN paper. The TRS-80's native 128×48 pixel output is resized to 84×84 grayscale, providing sufficient detail while keeping the network tractable.

### Why frame stacking?

Single frames lack velocity information. By stacking 4 frames, the network can perceive motion (e.g., ball direction in Breakout).

### Why action repeat?

Reduces computational load and adds consistency. Human players don't change direction every frame, and this allows the network to learn more stable policies.

---

## File Reference

| Path | Purpose |
|------|---------|
| `main.py` | DQN training loop, model definition, game configs |
| `trs/__init__.py` | Main TRS emulator class |
| `trs/z80.py` | Z80 CPU interface |
| `trs/ram.py` | Memory management |
| `trs/keyboard.py` | Input handling |
| `trs/screenshot.py` | Frame capture |
| `trs/video.py` | Pyglet display |
| `native/trs.c` | C emulator core |
| `libz80/` | Z80 CPU library |
| `var/breakdown.cmd` | Breakdown game ROM |
| `var/cosmic.cmd` | Cosmic Fighter game ROM |
| `var/*.ttf` | TRS-80 character fonts |

---

## Dependencies

**Core ML Stack:**
- TensorFlow 2.19.0
- Keras 3.9.2
- NumPy 2.1.3

**Image Processing:**
- Pillow 11.2.1
- scikit-image 0.25.2

**Visualization:**
- Pyglet 2.0.7 (optional, for UI)

**GPU Support:**
- CUDA 12 packages included in `requirements.txt`
- Alternative `requirements_cuda.txt` for CUDA 11.8

---

## Performance Considerations

1. **Headless Training:** Use `--no-ui` to disable Pyglet rendering for ~3× speedup
2. **GPU Acceleration:** TensorFlow will auto-detect CUDA for faster training
3. **Replay Buffer:** Limited to 100k transitions to manage memory
4. **Batch Training:** Updates every 4 frames to balance speed vs. sample efficiency

---

## Comparison to Original DQN

| Aspect | DeepMind Atari | This Project |
|--------|---------------|--------------|
| Platform | Atari 2600 via ALE | TRS-80 via custom emulator |
| Input | 210×160 RGB → 84×84 gray | 128×48 → 84×84 gray |
| Actions | 4-18 per game | 4-6 per game |
| Frame skip | 4 | 4 |
| Network | CNN + FC | Identical architecture |
| Replay size | 1M | 100K |
| Target update | 10K frames | 10K frames |

---

## Future Improvements

1. **Prioritized Experience Replay** - Sample important transitions more often
2. **Double DQN** - Reduce Q-value overestimation
3. **Dueling Networks** - Separate value and advantage streams
4. **More Games** - Add additional TRS-80 games to test generalization
5. **Curriculum Learning** - Start with easier game configurations

---

## Why the Model Isn't Converging: Analysis & Fixes

This section identifies critical bugs, performance bottlenecks, and improvements needed to achieve convergence.

### Critical Bugs (Must Fix)

#### 1. **BROKEN: Reward Shaping Destroys Q-Value Scale**

**Location:** `main.py:198-199`

```python
self.steps_survived += 1
reward += 0.001 * self.steps_survived
```

**Problem:** This reward grows **unboundedly** throughout an episode:
- Step 100: bonus = 0.1
- Step 1000: bonus = 1.0 (same magnitude as hitting a brick!)
- Step 5000: bonus = 5.0 (5× a real reward!)

This completely overwhelms the actual game rewards and makes Q-values explode. The network learns "survive longer = infinite reward" instead of "break bricks = good".

**Fix Options:**
```python
# Option A: Remove it entirely (recommended for initial convergence)
# reward += 0.001 * self.steps_survived  # REMOVE THIS LINE

# Option B: Use a small constant bonus (doesn't grow)
reward += 0.001  # Fixed small survival bonus

# Option C: Clip the bonus
reward += min(0.001 * self.steps_survived, 0.1)  # Cap at 0.1
```

#### 2. **BUG: Epsilon Decays During Random Warmup**

**Location:** `main.py:311-313`

```python
# Decay probability of taking random action
epsilon -= epsilon_interval / epsilon_greedy_frames
epsilon = max(epsilon, epsilon_min)
```

**Problem:** This runs on **every frame**, but it should only start after `epsilon_random_frames` (50,000). Currently, by frame 50,000, epsilon has already decayed from 1.0 to ~0.955.

**Fix:**
```python
# Only decay epsilon after warmup period
if frame_count > epsilon_random_frames:
    epsilon -= epsilon_interval / epsilon_greedy_frames
    epsilon = max(epsilon, epsilon_min)
```

#### 3. **BUG: Terminal vs Game Over Confusion**

**Location:** `main.py:196, 207-208, 219`

The reward function returns `(reward, terminal, game_over)` where:
- `terminal` = lost a life
- `game_over` = all lives lost

But `step()` returns `terminal or game_over` as `done`. This means losing a single life ends the episode and resets, but **`steps_survived` is not reset on life loss** during `frame_step`.

**Impact:** The agent gets inconsistent episode boundaries. In Breakout, losing a life shouldn't end the episode—it should continue with remaining lives.

**Fix:** Only end episode on `game_over`, not `terminal`:
```python
# In step():
return self.state, reward, game_over, None  # NOT terminal or game_over
```

#### 4. **MISSING: Frame Max-Pooling**

DeepMind's original paper takes the **pixel-wise maximum** over 2 consecutive frames to handle sprite flickering (common in old games). This is not implemented.

**Impact:** The network may see inconsistent/flickering states, making learning harder.

**Fix:** In `frame_step`, track last 2 frames and take max:
```python
prev_screenshot = None
for _ in range(self.action_repeat):
    # ... existing code ...
    curr_screenshot = self.screenshot.screenshot()
    if prev_screenshot is not None:
        last_screenshot = np.maximum(prev_screenshot, curr_screenshot)
    else:
        last_screenshot = curr_screenshot
    prev_screenshot = curr_screenshot
```

---

### Performance Issues

#### Why GPU is SLOWER Than CPU

This is actually expected for this workload. Here's why:

| Factor | Impact |
|--------|--------|
| **Small batch size (32)** | GPU excels at large batches (256+). With batch=32, CPU cache is more efficient |
| **Small model (~1.7M params)** | GPU overhead (memory transfer, kernel launch) exceeds compute benefit |
| **Frequent small operations** | Each `model.predict()` transfers 84×84×4 floats to GPU and back |
| **Python GIL + TensorFlow** | Single-threaded Python can't feed GPU fast enough |

**Recommendation:** Use CPU for this model size. Disable GPU:
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU
```

Or for TensorFlow:
```python
tf.config.set_visible_devices([], 'GPU')
```

#### Replay Buffer is O(n) Per Delete

**Location:** `main.py:383-388`

```python
if len(rewards_history) > max_memory_length:
    del rewards_history[:1]
    del state_history[:1]
    # ... etc
```

**Problem:** `del list[:1]` is O(n) because Python lists shift all elements. At 100,000 entries, this is slow.

**Fix:** Use `collections.deque`:
```python
from collections import deque

action_history = deque(maxlen=max_memory_length)
state_history = deque(maxlen=max_memory_length)
# ... etc

# No manual deletion needed - deque handles it automatically in O(1)
```

#### Inefficient Batch Sampling

**Location:** `main.py:340-344`

```python
state_sample = np.array([state_history[i] for i in indices])
```

**Problem:** List comprehension with indexing into a list is slow. Converting to numpy array allocates memory.

**Fix:** Store as numpy arrays from the start:
```python
# Pre-allocate numpy arrays
state_buffer = np.zeros((max_memory_length, 84, 84, 4), dtype=np.float32)
# ... fill buffer[index] = state ...

# Sample directly
state_sample = state_buffer[indices]  # Zero-copy view
```

#### Screenshot Resize is Slow

**Location:** `main.py:178, 215`

```python
x_t = skimage.transform.resize(x_t, (84, 84))
```

`skimage.transform.resize` is CPU-intensive and runs on every frame.

**Faster alternatives:**
```python
# Option 1: OpenCV (much faster)
import cv2
x_t = cv2.resize(x_t, (84, 84), interpolation=cv2.INTER_AREA)

# Option 2: PIL (also faster)
from PIL import Image
x_t = np.array(Image.fromarray(x_t).resize((84, 84), Image.BILINEAR))
```

---

### Framework Alternatives for Speed

#### Option 1: JAX + Flax (Recommended)

JAX compiles the entire training loop to optimized code via XLA. For small models, this can be 3-5× faster than TensorFlow.

```python
import jax
import jax.numpy as jnp
from flax import linen as nn

class QNetwork(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(32, (8, 8), strides=(4, 4))(x)
        x = nn.relu(x)
        x = nn.Conv(64, (4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(64, (3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        return nn.Dense(self.num_actions)(x)

# JIT compile the forward pass
@jax.jit
def get_action(params, state):
    q_values = model.apply(params, state[None])
    return jnp.argmax(q_values[0])
```

**Pros:** Fastest for small batches on CPU, easy to parallelize
**Cons:** Learning curve, different ecosystem

#### Option 2: PyTorch with TorchScript

PyTorch is often faster than TensorFlow for small models due to less overhead:

```python
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.conv(x.permute(0, 3, 1, 2))  # NHWC -> NCHW

# Compile for speed
model = torch.compile(QNetwork(4))
```

#### Option 3: Use Stable-Baselines3

Don't reinvent the wheel—use a battle-tested implementation:

```python
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# Wrap your game in a Gym interface
env = DummyVecEnv([lambda: YourGameEnv()])

model = DQN(
    "CnnPolicy",
    env,
    buffer_size=100000,
    learning_rate=2.5e-4,
    batch_size=32,
    gamma=0.99,
    exploration_fraction=0.1,
    target_update_interval=10000,
    verbose=1
)
model.learn(total_timesteps=1_000_000)
```

**Pros:** Proven to work, handles all DQN variants (Double, Dueling, PER)
**Cons:** Need to wrap your environment in Gym API

---

### Parallelization Opportunities

#### 1. Vectorized Environments

Run multiple game instances in parallel. This is the **single biggest speedup** for RL:

```python
# Run 8 parallel environments
from multiprocessing import Pool

def run_env_step(args):
    env_id, action = args
    return envs[env_id].step(action)

with Pool(8) as p:
    results = p.map(run_env_step, [(i, actions[i]) for i in range(8)])
```

Or use existing frameworks:
```python
from stable_baselines3.common.vec_env import SubprocVecEnv
envs = SubprocVecEnv([make_env for _ in range(8)])
```

#### 2. Async Environment Stepping

While training on batch N, collect batch N+1 in parallel:

```python
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=2)

# Submit next env step while training
future = executor.submit(env.step, action)
# ... do training ...
state_next, reward, done, info = future.result()
```

#### 3. Prefetch Screenshots

The C emulator could run ahead and buffer screenshots:

```c
// In trs.c - run emulator in separate thread, buffer frames
#define FRAME_BUFFER_SIZE 4
float frame_buffer[FRAME_BUFFER_SIZE][SCREENSHOT_SIZE];
int frame_write_idx = 0;
int frame_read_idx = 0;
```

---

### Algorithm Improvements to Try

#### 1. Double DQN (Reduces Q-value Overestimation)

The current implementation overestimates Q-values because it uses `max` for both action selection and evaluation.

**Current (problematic):**
```python
future_rewards = model_target.predict(state_next_sample)
updated_q_values = rewards + gamma * tf.reduce_max(future_rewards, axis=1)
```

**Fix (Double DQN):**
```python
# Use online network to SELECT action
future_q_online = model(state_next_sample)
best_actions = tf.argmax(future_q_online, axis=1)

# Use target network to EVALUATE action
future_q_target = model_target(state_next_sample)
best_q = tf.gather_nd(future_q_target,
    tf.stack([tf.range(batch_size), best_actions], axis=1))

updated_q_values = rewards + gamma * best_q * (1 - done_sample)
```

#### 2. Prioritized Experience Replay (PER)

Sample transitions with higher TD-error more often:

```python
# Instead of uniform sampling:
# indices = np.random.choice(len(buffer), batch_size)

# Use priority-weighted sampling:
priorities = np.abs(td_errors) + 0.01  # Small constant to avoid zero
probs = priorities ** alpha / sum(priorities ** alpha)
indices = np.random.choice(len(buffer), batch_size, p=probs)

# Importance sampling weights to correct bias
weights = (len(buffer) * probs[indices]) ** (-beta)
weights /= max(weights)
loss = weights * huber_loss(td_errors)
```

#### 3. Dueling Architecture

Separate value and advantage streams:

```python
def create_dueling_q_model():
    inputs = layers.Input(shape=(84, 84, 4))
    x = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    x = layers.Conv2D(64, 4, strides=2, activation="relu")(x)
    x = layers.Conv2D(64, 3, strides=1, activation="relu")(x)
    x = layers.Flatten()(x)

    # Value stream
    v = layers.Dense(512, activation="relu")(x)
    v = layers.Dense(1)(v)

    # Advantage stream
    a = layers.Dense(512, activation="relu")(x)
    a = layers.Dense(num_actions)(a)

    # Combine: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
    q = v + (a - tf.reduce_mean(a, axis=1, keepdims=True))

    return keras.Model(inputs=inputs, outputs=q)
```

#### 4. N-step Returns

Use multi-step TD targets for faster credit assignment:

```python
# Instead of 1-step: r + γ * max Q(s')
# Use 3-step: r₁ + γr₂ + γ²r₃ + γ³ * max Q(s''')

n_steps = 3
gamma_n = gamma ** n_steps

# Store n-step transitions
n_step_buffer = deque(maxlen=n_steps)
# ... accumulate rewards over n steps ...
```

---

### Recommended Fix Priority

| Priority | Fix | Effort | Impact |
|----------|-----|--------|--------|
| 🔴 **P0** | Remove/fix reward shaping | 1 line | **Critical** - currently breaks learning |
| 🔴 **P0** | Fix epsilon decay timing | 2 lines | High - proper exploration |
| 🟠 **P1** | Use `deque` for replay buffer | 10 lines | Medium - 10× buffer speedup |
| 🟠 **P1** | Force CPU (disable GPU) | 2 lines | Medium - faster inference |
| 🟡 **P2** | Implement Double DQN | 20 lines | Medium - stable Q-values |
| 🟡 **P2** | Use OpenCV for resize | 5 lines | Low - small speedup |
| 🟢 **P3** | Pre-allocated numpy buffers | 50 lines | Medium - memory efficiency |
| 🟢 **P3** | Vectorized environments | 100+ lines | **High** - 4-8× throughput |
| 🔵 **P4** | Switch to JAX/Flax | Major rewrite | Medium - faster training |
| 🔵 **P4** | Use Stable-Baselines3 | Major rewrite | High - proven implementation |

---

### Quick Start: Minimal Fixes for Convergence

Apply these changes to `main.py` to have the best chance of convergence:

```python
# 1. Remove broken reward shaping (line 199)
# DELETE: reward += 0.001 * self.steps_survived

# 2. Fix epsilon decay (after line 313)
if frame_count > epsilon_random_frames:
    epsilon -= epsilon_interval / epsilon_greedy_frames
    epsilon = max(epsilon, epsilon_min)

# 3. Use deque for replay buffers (at initialization)
from collections import deque
action_history = deque(maxlen=max_memory_length)
state_history = deque(maxlen=max_memory_length)
state_next_history = deque(maxlen=max_memory_length)
rewards_history = deque(maxlen=max_memory_length)
done_history = deque(maxlen=max_memory_length)

# 4. Remove manual deletion code (delete lines 383-388)
# The deque handles this automatically

# 5. Disable GPU (add at top of file)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

After these fixes, run training for at least 500,000 frames before evaluating. The original DQN paper trained for 50 million frames to achieve superhuman performance.

---

## References

1. Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." *Nature*, 518(7540), 529-533.
2. Original TRS-DQN fork: https://github.com/apuder/trs-dqn
3. Keras Breakout DQN: https://keras.io/examples/rl/deep_q_network_breakout/
4. Z80 CPU Reference: https://www.zilog.com/docs/z80/um0080.pdf
5. Van Hasselt, H., et al. (2016). "Deep Reinforcement Learning with Double Q-learning." *AAAI*.
6. Schaul, T., et al. (2016). "Prioritized Experience Replay." *ICLR*.
7. Wang, Z., et al. (2016). "Dueling Network Architectures for Deep Reinforcement Learning." *ICML*.

# Running AlphaZero Abalone on Google Colab

This guide provides step-by-step instructions for training your AlphaZero Abalone agent on Google Colab's free GPU.

## Quick Start (Copy-Paste into Colab)

Open a new notebook at [colab.research.google.com](https://colab.research.google.com) and follow these steps:

---

## Step 1: Enable GPU Runtime

**Task:** Configure Colab to use a GPU

1. Click `Runtime` → `Change runtime type`
2. Under `Hardware accelerator`, select `GPU` (usually T4)
3. Click `Save`

**Verification:** Run this in a code cell:
```python
!nvidia-smi
```
You should see GPU information (Tesla T4 or similar).

---

## Step 2: Install JAX with GPU Support

**Task:** Install JAX with CUDA support for GPU acceleration

```python
# Install JAX with GPU support
!pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install other dependencies
!pip install flax optax
```

**Verification:** Run this to confirm GPU is detected:
```python
import jax
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")
```

Expected output should show `gpu` as the default backend.

---

## Step 3: Clone the Repository

**Task:** Download the AlphaZero Abalone code to Colab

```python
# Clone your repository
!git clone https://github.com/YOUR_USERNAME/BetaAbalone.git

# Change to the project directory
%cd BetaAbalone

# List files to verify
!ls -la
```

**Note:** Replace `YOUR_USERNAME` with your actual GitHub username, or use the full repository URL.

---

## Step 4: Verify Installation

**Task:** Test that all components work correctly

```python
# Test the network
!python -c "from abalone_network import create_network; import jax; network, params = create_network(jax.random.PRNGKey(42)); print('Network: OK')"

# Test the game
!python -c "from abalone_game import AbaloneGame; game = AbaloneGame(); print(f'Game: OK, {len(game.get_legal_moves())} legal moves')"

# Test MCTS
!python mcts_simple.py
```

All tests should pass with "OK" messages.

---

## Step 5: Run Quick Test Training

**Task:** Verify training works with a minimal configuration

```python
# Quick test: 1 iteration, 1 game, 10 simulations
!python train_simple.py \
    --iterations 1 \
    --games 1 \
    --simulations 10 \
    --filters 16 \
    --blocks 1 \
    --batch-size 16 \
    --epochs 1
```

**Expected time:** ~30-60 seconds

This should complete successfully and create a checkpoint in `checkpoints/checkpoint_1.pkl`.

---

## Step 6: Run Real Training (Recommended Settings for Colab)

**Task:** Start actual training with GPU-optimized settings

### Option A: Quick Training (2-3 hours)
```python
!python train_simple.py \
    --iterations 50 \
    --games 10 \
    --simulations 50 \
    --filters 64 \
    --blocks 5 \
    --batch-size 128 \
    --epochs 5
```

### Option B: Full Training (6-8 hours)
```python
!python train_simple.py \
    --iterations 100 \
    --games 25 \
    --simulations 100 \
    --filters 128 \
    --blocks 10 \
    --batch-size 256 \
    --epochs 10
```

### Option C: Maximum Quality (12+ hours, may need Colab Pro)
```python
!python train_simple.py \
    --iterations 200 \
    --games 50 \
    --simulations 200 \
    --filters 128 \
    --blocks 10 \
    --batch-size 256 \
    --epochs 10
```

**Note:** Colab free tier disconnects after ~12 hours of inactivity. See Step 8 for checkpoint management.

---

## Step 7: Monitor Training Progress

**Task:** Track training while it runs

The training script will print:
- Iteration number
- Number of games generated
- Replay buffer size
- Training loss (total, policy, value)
- Checkpoint save confirmations

**Example output:**
```
Iteration 5/50
Generating 10 games...
Replay buffer size: 543
Training...
Loss: 1.2341, Policy: 0.8234, Value: 0.4107
Saved checkpoint at iteration 5
```

---

## Step 8: Save Checkpoints to Google Drive

**Task:** Prevent losing your trained model when Colab disconnects

### Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Create backup directory
```python
!mkdir -p /content/drive/MyDrive/AbaloneCheckpoints
```

### Copy checkpoints periodically (run this manually or add to training script)
```python
!cp -r checkpoints/* /content/drive/MyDrive/AbaloneCheckpoints/
print("Checkpoints backed up to Google Drive")
```

### Auto-backup script (optional - run in separate cell)
```python
import time
import os
import shutil

def auto_backup(interval_seconds=600):
    """Backup checkpoints every interval_seconds"""
    while True:
        time.sleep(interval_seconds)
        if os.path.exists('checkpoints'):
            shutil.copytree('checkpoints', '/content/drive/MyDrive/AbaloneCheckpoints', dirs_exist_ok=True)
            print(f"Auto-backup completed at {time.strftime('%H:%M:%S')}")

# Run in background (start this before training)
import threading
backup_thread = threading.Thread(target=auto_backup, args=(600,), daemon=True)
backup_thread.start()
print("Auto-backup started (every 10 minutes)")
```

---

## Step 9: Resume Training from Checkpoint

**Task:** Continue training if Colab disconnects

### If you need to restart:
```python
# Restore from Google Drive
!cp -r /content/drive/MyDrive/AbaloneCheckpoints/* checkpoints/

# Find latest checkpoint
!ls -lth checkpoints/

# Resume training from specific checkpoint
!python train_simple.py \
    --checkpoint checkpoints/checkpoint_50.pkl \
    --iterations 100 \
    --games 25 \
    --simulations 100 \
    --filters 128 \
    --blocks 10
```

**Note:** The `--checkpoint` flag loads the model and continues from that iteration.

---

## Step 10: Download Trained Model

**Task:** Get your trained model to use locally

### Option A: Download directly from Colab
```python
from google.colab import files

# Download the final checkpoint
files.download('checkpoints/final_checkpoint.pkl')
```

### Option B: It's already in Google Drive
If you used Step 8, your checkpoints are already saved to:
`Google Drive/AbaloneCheckpoints/`

You can download them from drive.google.com.

---

## Step 11: Test the Trained Model (Without GUI)

**Task:** Verify your trained model works

Since Colab doesn't support Pygame GUI, you can test the AI logic:

```python
import pickle
import jax
from abalone_game import AbaloneGame
from abalone_network import create_network
from mcts_simple import SimpleMCTSPlayer, MCTSConfig

# Load checkpoint
with open('checkpoints/final_checkpoint.pkl', 'rb') as f:
    checkpoint = pickle.load(f)

# Create network
key = jax.random.PRNGKey(0)
config = checkpoint['config']
network, _ = create_network(key, num_filters=config.num_filters, num_blocks=config.num_blocks)
params = checkpoint['params']

# Create AI player
mcts_config = MCTSConfig(num_simulations=50)
player = SimpleMCTSPlayer(network, params, mcts_config)

# Play a game
game = AbaloneGame()
move_count = 0

print("Starting AI self-play test...")
while not game.is_game_over() and move_count < 50:
    move, policy = player.select_move(game, temperature=0.0)
    if move is None:
        break
    print(f"Move {move_count + 1}: {move}")
    game.make_move(move)
    move_count += 1

print(f"\nGame ended after {move_count} moves")
print(f"Winner: {game.get_winner()}")
```

---

## Troubleshooting

### "RuntimeError: CUDA out of memory"
**Solution:** Reduce batch size or network size:
```python
--batch-size 64 --filters 64 --blocks 5
```

### "Session crashed" or disconnection
**Solution:**
- Your checkpoints are saved every iteration
- Use Step 9 to resume from last checkpoint
- Consider Colab Pro for longer sessions

### "No GPU available"
**Solution:**
- Check Runtime → Change runtime type → GPU is selected
- Free tier has limited GPU availability, try again later
- Run `!nvidia-smi` to verify GPU

### Training is slow
**Solution:**
- Verify GPU is being used: `print(jax.devices())`
- Reduce `--simulations` for faster iterations
- Parallel MCTS is enabled by default with batch_size=8

### ImportError for packages
**Solution:**
```python
!pip install flax optax jax[cuda12]
```

---

## Performance Expectations on Colab T4 GPU

| Configuration | Time per Iteration | Total Time (100 iters) |
|--------------|-------------------|------------------------|
| Small (16 filters, 1 block, 10 sims) | ~1 min | ~2 hours |
| Medium (64 filters, 5 blocks, 50 sims) | ~3-5 min | ~6 hours |
| Large (128 filters, 10 blocks, 100 sims) | ~8-12 min | ~15 hours |

**Note:** Times assume 10-25 games per iteration. Parallel MCTS provides ~3-5x speedup.

---

## Complete Colab Notebook Template

Here's a complete notebook you can copy-paste:

```python
# Cell 1: Setup GPU and install dependencies
!nvidia-smi
!pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
!pip install flax optax

# Cell 2: Verify JAX GPU
import jax
print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

# Cell 3: Clone repository
!git clone https://github.com/YOUR_USERNAME/BetaAbalone.git
%cd BetaAbalone

# Cell 4: Mount Google Drive for checkpoints
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/AbaloneCheckpoints

# Cell 5: Quick test
!python train_simple.py --iterations 1 --games 1 --simulations 10 --filters 16 --blocks 1

# Cell 6: Start training
!python train_simple.py \
    --iterations 100 \
    --games 25 \
    --simulations 100 \
    --filters 128 \
    --blocks 10 \
    --batch-size 256 \
    --epochs 10

# Cell 7: Backup to Drive (run after training or periodically)
!cp -r checkpoints/* /content/drive/MyDrive/AbaloneCheckpoints/
```

---

## Next Steps After Training

1. Download `final_checkpoint.pkl` from Google Drive
2. Use it locally with `play_game.py`:
   ```bash
   python play_game.py --mode vs-ai --checkpoint final_checkpoint.pkl
   ```
3. Continue training with more iterations if needed
4. Experiment with different network architectures

---

## Tips for Best Results

1. **Use Google Drive backups** - Don't lose hours of training
2. **Start small** - Test with 1-2 iterations first
3. **Monitor GPU usage** - Run `!nvidia-smi` to check utilization
4. **Save often** - Checkpoints save every iteration automatically
5. **Experiment** - Try different architectures and hyperparameters
6. **Be patient** - Quality training takes time (6-12 hours recommended)

---

## Questions or Issues?

- Check the main `STATUS.md` for project status
- See `ARCHITECTURE.md` for technical details
- Review `train_simple.py` command-line options with `--help`

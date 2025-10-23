# BetaAbalone

AlphaZero implementation for the board game Abalone.

## Installation

```bash
git clone https://github.com/yourusername/BetaAbalone.git
cd BetaAbalone

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- JAX (with GPU support recommended for training)
- Flax (neural network library)
- mctx (Monte Carlo Tree Search)
- Pygame (for GUI)
- NumPy, tqdm, optax

## Quick Start

### 1. Human vs Human

```bash
python main.py --mode test
```

Controls:
- Click marbles to select (up to 3)
- Press direction keys to move:
  - `Q` = Northwest, `W` = Northeast, `E` = East
  - `A` = West, `S` = Southeast, `D` = Southwest
- `R` = Reset selection
- `Space` = Toggle legal move indicators

### 2. Full Training

```bash
python main.py --mode train --iterations 1000 --games-per-iter 100 --filters 128 --blocks 10 --simulations 800
```

### 3. Play Against Trained AI

```bash
python main.py --mode play --checkpoint checkpoints/final_checkpoint.pkl
```

Add `--play-black` to play as Black (default is White).

### 4. Watch AI vs AI

```bash
python main.py --mode watch --checkpoint checkpoints/final_checkpoint.pkl --simulations 400
```

## Project Structure

```
BetaAbalone/
├── main.py                    # Main entry point
├── abalone_game.py            # Core game logic and rules
├── abalone_gui.py             # Pygame GUI for visualization
├── abalone_network.py         # Neural network architecture
├── mcts_player.py             # MCTS player using mctx
├── alphazero_trainer.py       # Training loop and self-play
├── play_game.py               # Play utilities
├── requirements.txt           # Dependencies
└── checkpoints/               # Saved model checkpoints
```

## Command Line

### Training Mode

```bash
python main.py --mode train [options]
```

**Training Options:**
- `--iterations`: Number of training iterations (default: 100)
- `--games-per-iter`: Self-play games per iteration (default: 50)
- `--simulations`: MCTS simulations per move (default: 800)
- `--batch-size`: Training batch size (default: 256)
- `--epochs`: Training epochs per iteration (default: 10)
- `--lr`: Learning rate (default: 0.001)

**Network Options:**
- `--filters`: Number of CNN filters (default: 128)
- `--blocks`: Number of residual blocks (default: 10)

**Checkpointing:**
- `--checkpoint`: Path to load existing checkpoint
- `--checkpoint-dir`: Directory for saving checkpoints (default: checkpoints)
- `--save-freq`: Save every N iterations (default: 10)

**Other:**
- `--seed`: Random seed (default: 42)

## Game Rules

Abalone is a two-player strategy game played on a hexagonal board:

- **Objective**: Push 6 of your opponent's marbles off the board
- **Moves**: Move 1-3 marbles in a line
  - **Sidestep**: Move perpendicular to the line (all spaces must be empty)
  - **Inline**: Move along the line (can push opponent marbles)
- **Pushing**: You can push opponent marbles if you have numerical superiority
  - 2 vs 1 or 3 vs 2 or 3 vs 1
  - Can't push your own marbles
  - Maximum 2 opponent marbles can be pushed

## References

- [AlphaZero Paper](https://arxiv.org/abs/1712.01815) - Silver et al., 2017
- [mctx Library](https://github.com/google-deepmind/mctx) - Google DeepMind
- [Abalone Rules](https://en.wikipedia.org/wiki/Abalone_(board_game))

## License

MIT License - see LICENSE file for details


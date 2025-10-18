# BetaAbalone

AlphaZero implementation for the board game Abalone using JAX, Flax, and Google DeepMind's mctx library.

## Features

- **Complete Abalone game engine** with move generation, validation, and win detection
- **Interactive GUI** using Pygame for visualization and human play
- **AlphaZero training** with self-play, MCTS, and neural network learning
- **CNN architecture** designed for hexagonal board representation
- **MCTS integration** using Google DeepMind's mctx library
- **End-to-end training pipeline** with checkpointing and replay buffer

## Installation

```bash
# Clone the repository
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

### 1. Test the Game (Human vs Human)

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

### 2. Quick Training Test

Train a small network for testing (fast, suitable for CPU):

```bash
python main.py --mode train --iterations 3 --games-per-iter 5 --filters 64 --blocks 3 --simulations 100
```

### 3. Full Training

Train a full AlphaZero agent (requires GPU, will take hours/days):

```bash
python main.py --mode train --iterations 1000 --games-per-iter 100 --filters 128 --blocks 10 --simulations 800
```

### 4. Play Against Trained AI

```bash
python main.py --mode play --checkpoint checkpoints/final_checkpoint.pkl
```

Add `--play-black` to play as Black (default is White).

### 5. Watch AI vs AI

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

## Architecture

### Game Engine (`abalone_game.py`)

- **Board Representation**: Hexagonal grid using axial coordinates (q, r) with flat-top orientation
- **Starting Setup**: 14 marbles per side - 5 in bottom/top row, 6 in second row, 3 in center of third row
- **Move Generation**: Generates all legal moves including inline pushes and sidesteps
- **Move Validation**: Checks validity of single, double, and triple marble moves
- **Game State**: Tracks current player, captured marbles, and win conditions

### Neural Network (`abalone_network.py`)

The network uses a ResNet-style architecture adapted for hexagonal boards:

- **Input**: 9×9×3 tensor (black marbles, white marbles, valid positions)
- **Architecture**:
  - Initial conv layer
  - 10 residual blocks (configurable)
  - Policy head → probability distribution over moves
  - Value head → win probability estimate
- **Output**:
  - Policy logits (500 dimensions for all possible moves)
  - Value estimate (scalar in [-1, 1])

### MCTS Player (`mcts_player.py`)

Integrates the neural network with Monte Carlo Tree Search:

- Uses Google DeepMind's `mctx` library
- Implements recurrent function for game dynamics
- Supports Gumbel MuZero for improved exploration
- Temperature-based move selection

### Training (`alphazero_trainer.py`)

AlphaZero training loop:

1. **Self-Play**: Generate games using current network + MCTS
2. **Experience Replay**: Store positions, policies, and outcomes
3. **Network Training**: Train on replay buffer with policy and value losses
4. **Iteration**: Repeat for continuous improvement

## Command Line Options

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

### Play Mode

```bash
python main.py --mode play --checkpoint <path> [options]
```

**Options:**
- `--checkpoint`: Path to model checkpoint (required)
- `--play-black`: Play as Black instead of White
- `--simulations`: MCTS simulations for AI moves (default: 400)

### Watch Mode

```bash
python main.py --mode watch --checkpoint <path> [options]
```

Watch the trained AI play against itself.

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

## Training Tips

1. **Start Small**: Test with small networks (64 filters, 3 blocks) and few games first
2. **GPU Recommended**: Training is much faster with CUDA-enabled JAX
3. **Monitoring**: Check `checkpoints/` for saved models and training progress
4. **Hyperparameters**: Adjust based on your hardware:
   - Reduce `--simulations` if training is too slow
   - Reduce `--filters` and `--blocks` for smaller networks
   - Increase `--games-per-iter` for better data diversity

## CNN Design for Hexagonal Board

The network processes the hexagonal Abalone board as a 9×9 grid:

1. **Input Encoding**:
   - Channel 0: Current player's marbles
   - Channel 1: Opponent's marbles
   - Channel 2: Valid position mask

2. **Convolution Strategy**:
   - Standard 2D convolutions work on the embedded hex grid
   - Valid position mask helps network learn board boundaries
   - Residual connections help with training deeper networks

3. **Move Encoding**:
   - Each move encoded as: (position, direction, num_marbles, orientation)
   - ~500 possible move types covering all valid actions
   - Output policy head has 500 dimensions

## Future Improvements

- [ ] Implement symmetry augmentation (rotations/reflections)
- [ ] Add opening book for stronger early game play
- [ ] Experiment with different network architectures
- [ ] Implement evaluation metrics (Elo rating, win rate tracking)
- [ ] Add tournament mode for comparing model versions
- [ ] Web-based GUI for online play

## References

- [AlphaZero Paper](https://arxiv.org/abs/1712.01815) - Silver et al., 2017
- [mctx Library](https://github.com/google-deepmind/mctx) - Google DeepMind
- [Abalone Rules](https://en.wikipedia.org/wiki/Abalone_(board_game))

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

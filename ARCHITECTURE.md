# AlphaZero for Abalone - Architecture Documentation

## Overview

This implementation uses a **hybrid Python/JAX architecture** with a **dynamic action space** approach for efficient learning.

## Key Design Decisions

### 1. Hybrid Architecture

**Why Hybrid?**
- Initial attempts to use pure JAX with mctx failed due to JAX tracing limitations with Python objects
- Abalone has complex game logic (move generation, validation) that's difficult to express in pure JAX
- Most production AlphaZero implementations use this hybrid approach

**Architecture:**
- **Python:** Game logic (`abalone_game.py`), MCTS tree search (`mcts_simple.py`)
- **JAX:** Neural network forward passes, gradient computation, training step

### 2. Dynamic Action Space

**Why Dynamic?**
- Original fixed vocabulary approach: 61 positions × 6 directions × 7 move types = 2,562 possible moves
- In any position, only ~40-50 moves are legal (~2% utilization)
- Network was wasting 98% of its policy head output on invalid moves

**Solution:**
- Network outputs 200 logits (maximum legal moves in any position)
- For each position, only the first N logits correspond to the N legal moves
- Use masking during training to ignore logits beyond N
- Much more efficient learning signal

### 3. Component Details

#### Game Logic (`abalone_game.py`)
- Uses axial coordinates (q, r) for hexagonal grid
- 61 valid positions on the board
- Each side starts with 14 marbles in 5-6-3 formation
- Complete move generation and validation

#### Neural Network (`abalone_network.py`)
- **Input:** (9, 9, 3) board representation
  - Channel 0: Current player's marbles
  - Channel 1: Opponent's marbles
  - Channel 2: Valid positions mask
- **Architecture:** ResNet-style CNN
  - Initial conv block
  - 10 residual blocks (default)
  - Policy head: outputs 200 logits
  - Value head: outputs scalar in [-1, 1]
- **Outputs:** (policy_logits, value)
  - policy_logits: (200,) for dynamic action space
  - value: scalar win probability

#### MCTS (`mcts_simple.py`)
- Python-based tree search (not JAX)
- Uses UCB for node selection
- Network provides:
  - Prior probabilities for legal moves
  - Value estimate for positions
- Returns visit count distribution as policy

#### Training (`train_simple.py`)
- Self-play generates training data
- Policy targets: visit count distributions
- Value targets: game outcome from each player's perspective
- Masked policy loss: only compute loss on legal move logits
- BatchNorm state properly tracked during training

## File Structure

### Core Components
- `abalone_game.py` - Game logic and rules
- `abalone_network.py` - Neural network architecture
- `mcts_simple.py` - Python-based MCTS player
- `move_encoding.py` - Dynamic action space encoding
- `train_simple.py` - AlphaZero training loop

### GUI and Utilities
- `abalone_gui.py` - Pygame visualization (flat-top hexagons)
- `play_game.py` - Human vs AI, AI vs AI interfaces
- `main.py` - Entry point (needs update to use new trainer)

### Experimental/Legacy
- `abalone_jax.py` - Partial pure-JAX implementation (incomplete)
- `alphazero_trainer.py` - Original trainer with fixed action space
- `mcts_player.py` - Original mctx-based MCTS (had tracing issues)

## Training Pipeline

1. **Self-Play**
   - MCTS player plays against itself
   - Each move: run N simulations, select based on visit counts
   - Temperature: high early (exploration), low late (exploitation)
   - Store: (state, policy, player) for each position

2. **Value Assignment**
   - After game ends, assign values based on outcome
   - Winner's positions: +1.0
   - Loser's positions: -1.0
   - Draw: 0.0

3. **Training**
   - Sample batches from replay buffer
   - Create masks for legal moves
   - Compute masked policy loss + value loss
   - Update network parameters

4. **Iteration**
   - Repeat: self-play → train → checkpoint
   - Network improves, play quality increases

## Performance Notes

### Current Performance
- MCTS: ~0.02 seconds per simulation
- Single game with 20 simulations/move: ~80 seconds
- 2 games @ 20 sims: ~160 seconds (2.7 minutes)

### Bottlenecks
1. **Game state cloning** - Each MCTS simulation clones the game state
2. **Python overhead** - Tree search is in Python, not JIT-compiled
3. **Network evaluation** - Called frequently during MCTS

### Optimization Opportunities
1. Implement game state cloning with copy-on-write
2. Batch network evaluations during MCTS
3. Reduce MCTS simulations (trade quality for speed)
4. Profile and optimize game.get_legal_moves()
5. Consider Numba or Cython for game logic

## Usage

### Basic Training
```python
from train_simple import SimpleTrainer, TrainingConfig
import jax

config = TrainingConfig(
    num_self_play_games=10,
    num_mcts_simulations=20,
    num_iterations=100,
    num_filters=64,
    num_blocks=5
)

key = jax.random.PRNGKey(42)
trainer = SimpleTrainer(config, key)
trainer.train()
```

### Quick Test
```bash
python train_simple.py  # Runs with minimal config
```

### Testing Components
```bash
python abalone_network.py  # Test network
python mcts_simple.py      # Test MCTS
python move_encoding.py    # Test encoding
python test_training.py    # Test full pipeline
```

## Known Issues

1. **Performance** - Training is slow, needs optimization
2. **Main.py** - Still uses old trainer, needs update
3. **Pure JAX** - `abalone_jax.py` incomplete (not critical)

## Future Work

1. Optimize MCTS performance
2. Implement batched network evaluation
3. Update main.py to use new trainer
4. Add more comprehensive testing
5. Hyperparameter tuning
6. Consider alternative MCTS implementations

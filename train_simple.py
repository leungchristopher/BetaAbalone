"""
Simplified AlphaZero Trainer using dynamic action space

This uses the simple MCTS implementation with dynamic action encoding.

Usage:
    python train_simple.py --iterations 100 --games 10 --simulations 50
    python train_simple.py --checkpoint checkpoints/checkpoint_50.pkl --iterations 100
"""

import jax
import jax.numpy as jnp
import optax
import numpy as np
from typing import List
from dataclasses import dataclass
from collections import deque
import pickle
import os
import argparse

from abalone_game import AbaloneGame, Player
from abalone_network import create_network, prepare_input
from mcts_simple import SimpleMCTSPlayer, MCTSConfig
from move_encoding import DynamicMoveEncoder


@dataclass
class TrainingConfig:
    num_self_play_games: int = 50
    num_mcts_simulations: int = 400
    temperature_threshold: int = 15
    batch_size: int = 128
    num_epochs: int = 5
    learning_rate: float = 0.001
    num_iterations: int = 100
    num_filters: int = 128
    num_blocks: int = 10
    checkpoint_dir: str = "checkpoints"


@dataclass
class GameExample:
    state: np.ndarray  # (9, 9, 3)
    policy: np.ndarray  # (200,) with only first num_legal moves filled
    value: float
    num_legal: int  # Number of legal moves


class SimpleTrainer:
    def __init__(self, config: TrainingConfig, rng_key):
        self.config = config
        self.rng_key = rng_key

        # Create network
        network_key, self.rng_key = jax.random.split(self.rng_key)
        self.network, self.params = create_network(
            network_key,
            num_filters=config.num_filters,
            num_blocks=config.num_blocks
        )

        # Create optimizer
        self.optimizer = optax.adam(learning_rate=config.learning_rate)
        self.opt_state = self.optimizer.init(self.params)

        # Replay buffer
        self.replay_buffer = deque(maxlen=10000)

        # Encoder
        self.encoder = DynamicMoveEncoder()

        # Iteration counter
        self.iteration = 0

    def self_play_game(self) -> List[GameExample]:
        game = AbaloneGame()
        examples = []
        move_count = 0

        mcts_config = MCTSConfig(num_simulations=self.config.num_mcts_simulations)
        player = SimpleMCTSPlayer(self.network, self.params, mcts_config)

        while not game.is_game_over() and move_count < 200:
            state = prepare_input(game)
            legal_moves = game.get_legal_moves()
            num_legal = len(legal_moves)

            # Select move
            temperature = 1.0 if move_count < self.config.temperature_threshold else 0.1
            move, visit_dist = player.select_move(game, temperature=temperature, add_noise=(move_count < 30))

            if move is None:
                break

            # Create policy vector (200 elements, first num_legal filled)
            policy = np.zeros(200, dtype=np.float32)
            policy[:len(visit_dist)] = visit_dist

            examples.append({
                'state': state,
                'policy': policy,
                'player': game.current_player,
                'num_legal': num_legal
            })

            game.make_move(move)
            move_count += 1

        # Assign values
        winner = game.get_winner()
        final_value = 0.0 if winner is None else 1.0

        training_examples = []
        for ex in examples:
            value = 0.0 if winner is None else (final_value if ex['player'] == winner else -final_value)
            training_examples.append(GameExample(
                state=ex['state'],
                policy=ex['policy'],
                value=value,
                num_legal=ex['num_legal']
            ))

        return training_examples

    def create_train_step(self):
        """Create JIT-compiled training step function."""
        network = self.network
        optimizer = self.optimizer

        @jax.jit
        def train_step(params, opt_state, batch_states, batch_policies, batch_values, batch_masks):
            def loss_fn(params):
                # For training, we need mutable batch_stats
                variables = {'params': params['params'], 'batch_stats': params['batch_stats']}
                (policy_logits, values), updated_state = network.apply(
                    variables, batch_states, train=True, mutable=['batch_stats']
                )

                # Policy loss (cross-entropy with masking)
                policy_probs = jax.nn.softmax(policy_logits, axis=-1)
                policy_loss = -jnp.sum(batch_policies * jnp.log(policy_probs + 1e-8) * batch_masks) / jnp.sum(batch_masks)

                # Value loss
                values = jnp.squeeze(values, -1)
                value_loss = jnp.mean((values - batch_values) ** 2)

                return policy_loss + value_loss, (policy_loss, value_loss, updated_state)

            (loss, (policy_loss, value_loss, updated_state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            # Update batch_stats
            new_params = {
                'params': new_params['params'],
                'batch_stats': updated_state['batch_stats']
            }

            return new_params, new_opt_state, loss, policy_loss, value_loss

        return train_step

    def train(self, start_iteration: int = 0):
        """
        Train the network.

        Args:
            start_iteration: Iteration to start from (for resuming training)
        """
        print(f"Starting training from iteration {start_iteration + 1}...")

        # Create JIT-compiled training step
        train_step = self.create_train_step()

        for iteration in range(start_iteration, self.config.num_iterations):
            print(f"\nIteration {iteration + 1}/{self.config.num_iterations}")

            # Self-play
            print(f"Generating {self.config.num_self_play_games} games...")
            for _ in range(self.config.num_self_play_games):
                examples = self.self_play_game()
                self.replay_buffer.extend(examples)

            print(f"Replay buffer size: {len(self.replay_buffer)}")

            # Train
            if len(self.replay_buffer) >= self.config.batch_size:
                print("Training...")
                for epoch in range(self.config.num_epochs):
                    indices = np.random.choice(len(self.replay_buffer), self.config.batch_size, replace=False)
                    batch = [self.replay_buffer[i] for i in indices]

                    batch_states = jnp.array([ex.state for ex in batch])
                    batch_policies = jnp.array([ex.policy for ex in batch])
                    batch_values = jnp.array([ex.value for ex in batch], dtype=jnp.float32)

                    # Create masks for legal moves
                    batch_masks = jnp.zeros((self.config.batch_size, 200), dtype=jnp.float32)
                    for i, ex in enumerate(batch):
                        batch_masks = batch_masks.at[i, :ex.num_legal].set(1.0)

                    self.params, self.opt_state, loss, p_loss, v_loss = train_step(
                        self.params, self.opt_state, batch_states, batch_policies, batch_values, batch_masks
                    )

                    if epoch == 0:
                        print(f"Loss: {float(loss):.4f}, Policy: {float(p_loss):.4f}, Value: {float(v_loss):.4f}")

            # Save checkpoint after every iteration
            self._save_checkpoint(iteration)

        # Save final checkpoint
        self._save_checkpoint(self.config.num_iterations - 1, final=True)
        print("\nTraining complete!")

    def _save_checkpoint(self, iteration: int, final: bool = False):
        """Save training checkpoint."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        checkpoint = {
            'params': self.params,
            'iteration': iteration,
            'config': self.config,
            'replay_buffer_size': len(self.replay_buffer)
        }

        if final:
            filename = f"{self.config.checkpoint_dir}/final_checkpoint.pkl"
            print(f"Saving final checkpoint...")
        else:
            filename = f"{self.config.checkpoint_dir}/checkpoint_{iteration + 1}.pkl"
            print(f"Saved checkpoint at iteration {iteration + 1}")

        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load training state from checkpoint.

        Returns:
            iteration: The iteration number to resume from
        """
        print(f"Loading checkpoint from {checkpoint_path}...")
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)

        self.params = checkpoint['params']
        iteration = checkpoint.get('iteration', 0)

        # Reinitialize optimizer state with loaded params
        self.opt_state = self.optimizer.init(self.params)

        print(f"Checkpoint loaded successfully")
        print(f"  Iteration: {iteration + 1}")
        if 'replay_buffer_size' in checkpoint:
            print(f"  Replay buffer was: {checkpoint['replay_buffer_size']}")

        return iteration + 1  # Return next iteration to start from


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train AlphaZero for Abalone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (3 iterations, small network)
  python train_simple.py --iterations 3 --games 2 --simulations 20 --filters 32 --blocks 2

  # Standard training
  python train_simple.py --iterations 100 --games 10 --simulations 50

  # Resume from checkpoint
  python train_simple.py --checkpoint checkpoints/checkpoint_50.pkl --iterations 100

  # Full training with large network
  python train_simple.py --iterations 1000 --games 50 --simulations 100 --filters 128 --blocks 10
        """
    )

    # Training parameters
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of training iterations (default: 10)')
    parser.add_argument('--games', type=int, default=5,
                       help='Self-play games per iteration (default: 5)')
    parser.add_argument('--simulations', type=int, default=20,
                       help='MCTS simulations per move (default: 20)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=2,
                       help='Training epochs per iteration (default: 2)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')

    # Network architecture
    parser.add_argument('--filters', type=int, default=64,
                       help='Number of CNN filters (default: 64)')
    parser.add_argument('--blocks', type=int, default=5,
                       help='Number of residual blocks (default: 5)')

    # Checkpointing
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory for saving checkpoints (default: checkpoints)')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    # Create configuration
    config = TrainingConfig(
        num_self_play_games=args.games,
        num_mcts_simulations=args.simulations,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        num_iterations=args.iterations,
        num_filters=args.filters,
        num_blocks=args.blocks,
        checkpoint_dir=args.checkpoint_dir
    )

    # Print configuration
    print("="*80)
    print("AlphaZero Training for Abalone")
    print("="*80)
    print(f"Configuration:")
    print(f"  Iterations: {config.num_iterations}")
    print(f"  Games per iteration: {config.num_self_play_games}")
    print(f"  MCTS simulations: {config.num_mcts_simulations}")
    print(f"  Network: {config.num_filters} filters, {config.num_blocks} blocks")
    print(f"  Checkpointing: After every iteration")
    print(f"  Checkpoint directory: {config.checkpoint_dir}")
    print()

    # Initialize trainer
    key = jax.random.PRNGKey(args.seed)
    trainer = SimpleTrainer(config, key)

    # Load checkpoint if provided
    start_iteration = 0
    if args.checkpoint:
        start_iteration = trainer.load_checkpoint(args.checkpoint)
        print()

    # Train
    trainer.train(start_iteration=start_iteration)

"""
AlphaZero for Abalone - Main Entry Point

This is the main script for training and playing Abalone with AlphaZero.

Usage:
    # Train a new model
    python main.py --mode train --iterations 100 --games-per-iter 50

    # Continue training from checkpoint
    python main.py --mode train --checkpoint checkpoints/checkpoint_iter_10.pkl

    # Play against trained AI
    python main.py --mode play --checkpoint checkpoints/final_checkpoint.pkl

    # Watch AI vs AI
    python main.py --mode watch --checkpoint checkpoints/final_checkpoint.pkl

    # Test the game (human vs human)
    python main.py --mode test
"""

import argparse
import jax
from train_simple import SimpleTrainer, TrainingConfig


def train(args):
    """Train AlphaZero agent."""
    print("="*80)
    print("AlphaZero Training for Abalone")
    print("="*80)

    # Create configuration
    config = TrainingConfig(
        num_self_play_games=args.games_per_iter,
        num_mcts_simulations=args.simulations,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        num_iterations=args.iterations,
        num_filters=args.filters,
        num_residual_blocks=args.blocks,
        learning_rate=args.lr,
        save_frequency=args.save_freq,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Initialize trainer
    key = jax.random.PRNGKey(args.seed)
    trainer = SimpleTrainer(config, key)

    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # Run training
    trainer.train()


def play(args):
    """Play against trained agent."""
    from play_game import play_human_vs_ai
    play_human_vs_ai(
        args.checkpoint,
        human_plays_black=args.play_black,
        num_simulations=args.simulations
    )


def watch(args):
    """Watch AI vs AI game."""
    from play_game import play_ai_vs_ai
    play_ai_vs_ai(args.checkpoint, args.simulations)


def test(args):
    """Test game with human vs human."""
    from play_game import play_human_vs_human
    play_human_vs_human()


def main():
    parser = argparse.ArgumentParser(
        description="AlphaZero for Abalone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test training (small network, few games)
  python main.py --mode train --iterations 3 --games-per-iter 5 --filters 64 --blocks 3

  # Full training
  python main.py --mode train --iterations 1000 --games-per-iter 100

  # Play against AI
  python main.py --mode play --checkpoint checkpoints/final_checkpoint.pkl

  # Watch AI vs AI
  python main.py --mode watch --checkpoint checkpoints/final_checkpoint.pkl

  # Test the game
  python main.py --mode test
        """
    )

    # Mode selection
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'play', 'watch', 'test'],
                       help='Mode: train, play, watch, or test')

    # Training arguments
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of training iterations (default: 100)')
    parser.add_argument('--games-per-iter', type=int, default=50,
                       help='Self-play games per iteration (default: 50)')
    parser.add_argument('--simulations', type=int, default=800,
                       help='MCTS simulations per move (default: 800)')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Training batch size (default: 256)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Training epochs per iteration (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')

    # Network architecture
    parser.add_argument('--filters', type=int, default=128,
                       help='Number of CNN filters (default: 128)')
    parser.add_argument('--blocks', type=int, default=10,
                       help='Number of residual blocks (default: 10)')

    # Checkpointing
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file (for loading or continuing training)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory for saving checkpoints (default: checkpoints)')
    parser.add_argument('--save-freq', type=int, default=10,
                       help='Save checkpoint every N iterations (default: 10)')

    # Play mode arguments
    parser.add_argument('--play-black', action='store_true',
                       help='Human plays as Black (default: White)')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    # Route to appropriate function
    if args.mode == 'train':
        train(args)
    elif args.mode == 'play':
        if not args.checkpoint:
            parser.error("--checkpoint is required for play mode")
        play(args)
    elif args.mode == 'watch':
        if not args.checkpoint:
            parser.error("--checkpoint is required for watch mode")
        watch(args)
    elif args.mode == 'test':
        test(args)


if __name__ == "__main__":
    main()

"""
Play Abalone games with trained AlphaZero agent

This script allows you to:
1. Play human vs human
2. Watch AI vs AI games
3. Play against the AI
"""

import argparse
import pickle
import jax
import jax.numpy as jnp
from abalone_game import AbaloneGame
from abalone_gui import AbaloneGUI
from abalone_network import create_network
from mcts_simple import SimpleMCTSPlayer, MCTSConfig


def load_trained_model(checkpoint_path: str):
    """Load a trained model from checkpoint."""
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    # Recreate network (try to get config from checkpoint, fallback to defaults)
    key = jax.random.PRNGKey(0)

    if 'config' in checkpoint:
        config = checkpoint['config']
        network, _ = create_network(
            key,
            num_filters=config.num_filters,
            num_blocks=config.num_blocks
        )
    else:
        # Fallback to default architecture
        network, _ = create_network(key, num_filters=128, num_blocks=10)

    params = checkpoint['params']

    return network, params


def create_ai_player(network, params, num_simulations=400):
    """Create an AI player with given network."""
    config = MCTSConfig(num_simulations=num_simulations)
    return SimpleMCTSPlayer(network, params, config)


def ai_player_wrapper(game, mcts_player):
    """Wrapper function for GUI AI player."""
    move, _ = mcts_player.select_move(game, temperature=0.0, add_noise=False)
    return move


def play_ai_vs_ai(checkpoint_path: str, num_simulations: int = 400):
    """Watch AI play against itself."""
    print(f"Loading model from {checkpoint_path}...")
    network, params = load_trained_model(checkpoint_path)

    print("Creating AI player...")
    mcts_player = create_ai_player(network, params, num_simulations)

    print("Starting game...")
    game = AbaloneGame()
    gui = AbaloneGUI(game)

    # Run with AI for both players
    ai_fn = lambda g: ai_player_wrapper(g, mcts_player)
    gui.run_game_loop(ai_player=ai_fn, move_delay=500)


def play_human_vs_ai(checkpoint_path: str, human_plays_black: bool = True,
                     num_simulations: int = 400):
    """Play against the AI."""
    print(f"Loading model from {checkpoint_path}...")
    network, params = load_trained_model(checkpoint_path)

    print("Creating AI player...")
    mcts_player = create_ai_player(network, params, num_simulations)

    print("Starting game...")
    game = AbaloneGame()
    gui = AbaloneGUI(game)

    # Create AI function that only plays on its turn
    def ai_fn(g):
        # Check if it's AI's turn
        is_ai_turn = (
            (g.current_player.value == 2 and not human_plays_black) or
            (g.current_player.value == 1 and human_plays_black)
        )

        if is_ai_turn:
            return ai_player_wrapper(g, mcts_player)
        return None

    print("You are playing as", "BLACK" if human_plays_black else "WHITE")
    print("Controls:")
    print("  Click marbles to select (up to 3)")
    print("  Press direction keys to move: Q=NW, W=NE, E=E, A=W, S=SE, D=SW")
    print("  Press R to reset selection")

    gui.run_game_loop(ai_player=ai_fn, move_delay=1000)


def play_human_vs_human():
    """Play human vs human."""
    print("Starting human vs human game...")
    game = AbaloneGame()
    gui = AbaloneGUI(game)

    print("Controls:")
    print("  Click marbles to select (up to 3)")
    print("  Press direction keys to move: Q=NW, W=NE, E=E, A=W, S=SE, D=SW")
    print("  Press R to reset selection")

    gui.run_game_loop()


def main():
    parser = argparse.ArgumentParser(description="Play Abalone with AlphaZero")
    parser.add_argument('--mode', type=str, default='human',
                       choices=['human', 'ai-vs-ai', 'vs-ai'],
                       help='Game mode')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/final_checkpoint.pkl',
                       help='Path to model checkpoint')
    parser.add_argument('--simulations', type=int, default=400,
                       help='Number of MCTS simulations per move')
    parser.add_argument('--play-black', action='store_true',
                       help='Human plays as black (only for vs-ai mode)')

    args = parser.parse_args()

    if args.mode == 'human':
        play_human_vs_human()
    elif args.mode == 'ai-vs-ai':
        play_ai_vs_ai(args.checkpoint, args.simulations)
    elif args.mode == 'vs-ai':
        play_human_vs_ai(args.checkpoint, args.play_black, args.simulations)


if __name__ == "__main__":
    main()

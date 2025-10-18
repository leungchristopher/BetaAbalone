"""
Basic tests to verify the implementation works

These tests can run without JAX dependencies installed.
"""

from abalone_game import AbaloneGame, Player, Move, Direction


def test_game_initialization():
    """Test that game initializes correctly."""
    print("Testing game initialization...")
    game = AbaloneGame()

    # Check board size
    assert game.board_size == 5
    assert len(game.valid_positions) == 61  # Standard Abalone board

    # Check starting position has 14 marbles per side
    black_count = sum(1 for p in game.board.values() if p == Player.BLACK)
    white_count = sum(1 for p in game.board.values() if p == Player.WHITE)

    assert black_count == 14, f"Expected 14 black marbles, got {black_count}"
    assert white_count == 14, f"Expected 14 white marbles, got {white_count}"
    print(f"  Black marbles: {black_count}, White marbles: {white_count}")

    # Check initial state
    assert game.current_player == Player.BLACK
    assert game.marbles_captured[Player.BLACK] == 0
    assert game.marbles_captured[Player.WHITE] == 0
    assert not game.is_game_over()

    print("✓ Game initialization test passed")


def test_move_generation():
    """Test that legal moves are generated."""
    print("\nTesting move generation...")
    game = AbaloneGame()

    legal_moves = game.get_legal_moves()

    # Should have moves available at start
    assert len(legal_moves) > 0, "No legal moves found at start position"
    assert len(legal_moves) < 200, f"Too many moves: {len(legal_moves)}"

    print(f"✓ Generated {len(legal_moves)} legal moves")


def test_move_execution():
    """Test that moves can be executed."""
    print("\nTesting move execution...")
    game = AbaloneGame()

    # Get a legal move
    legal_moves = game.get_legal_moves()
    assert len(legal_moves) > 0

    move = legal_moves[0]
    initial_player = game.current_player

    # Make the move
    success = game.make_move(move)
    assert success, "Legal move was rejected"

    # Check player switched
    assert game.current_player != initial_player, "Player didn't switch after move"

    print(f"✓ Move execution test passed")


def test_move_validation():
    """Test that invalid moves are rejected."""
    print("\nTesting move validation...")
    game = AbaloneGame()

    # Try to move opponent's marble
    # Find a white marble position
    white_pos = None
    for pos, player in game.board.items():
        if player == Player.WHITE:
            white_pos = pos
            break

    assert white_pos is not None

    # Try to move white marble when black is to move
    invalid_move = Move(marbles=(white_pos,), direction=Direction.E)
    success = game.make_move(invalid_move)

    assert not success, "Invalid move was accepted"

    print("✓ Move validation test passed")


def test_game_clone():
    """Test that game can be cloned."""
    print("\nTesting game cloning...")
    game = AbaloneGame()

    # Make a move
    move = game.get_legal_moves()[0]
    game.make_move(move)

    # Clone the game
    cloned_game = game.clone()

    # Check states match
    assert cloned_game.current_player == game.current_player
    assert cloned_game.marbles_captured == game.marbles_captured
    assert cloned_game.board == game.board

    # Modify clone
    clone_move = cloned_game.get_legal_moves()[0]
    cloned_game.make_move(clone_move)

    # Check original is unchanged
    assert cloned_game.current_player != game.current_player

    print("✓ Game cloning test passed")


def test_board_array():
    """Test board array conversion."""
    print("\nTesting board array conversion...")
    game = AbaloneGame()

    board_array = game.get_board_array()

    # Check shape (size is 2 * board_size - 1 = 2*5-1 = 9)
    expected_size = 2 * game.board_size - 1
    assert board_array.shape == (3, expected_size, expected_size), f"Wrong shape: {board_array.shape}"

    # Check that valid positions are marked
    valid_count = int(board_array[2].sum())
    assert valid_count == 61, f"Expected 61 valid positions, got {valid_count}"

    print("✓ Board array conversion test passed")


def run_all_tests():
    """Run all basic tests."""
    print("="*60)
    print("Running BetaAbalone Basic Tests")
    print("="*60)

    test_game_initialization()
    test_move_generation()
    test_move_execution()
    test_move_validation()
    test_game_clone()
    test_board_array()

    print("\n" + "="*60)
    print("All basic tests passed! ✓")
    print("="*60)
    print("\nTo test the full system with neural networks:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Test network: python abalone_network.py")
    print("3. Test MCTS: python mcts_player.py")
    print("4. Run quick training: python main.py --mode train --iterations 3 --games-per-iter 5 --filters 64 --blocks 3")


if __name__ == "__main__":
    run_all_tests()

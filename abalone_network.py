"""
Neural Network Architecture for AlphaZero Abalone

This implements a CNN-based network that outputs:
1. Policy head: Probability distribution over all possible moves
2. Value head: Win probability for current player
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Dict, Any
import numpy as np


class ResidualBlock(nn.Module):
    """Residual block for the CNN."""
    filters: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        """Forward pass."""
        residual = x

        # First conv
        x = nn.Conv(features=self.filters, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        # Second conv
        x = nn.Conv(features=self.filters, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)

        # Add residual
        x = x + residual
        x = nn.relu(x)

        return x


class HexConvBlock(nn.Module):
    """
    Convolutional block adapted for hexagonal grid.

    For hexagonal grids, we use standard 2D convolutions on the
    rectangular embedding of the hexagonal board.
    """
    filters: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        """Forward pass."""
        x = nn.Conv(features=self.filters, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        return x


class AbaloneNetwork(nn.Module):
    """
    AlphaZero neural network for Abalone.

    Input: Board state (3 channels, 9x9):
        - Channel 0: Current player's marbles
        - Channel 1: Opponent's marbles
        - Channel 2: Valid positions mask

    Output:
        - Policy logits: Shape (num_possible_moves,)
        - Value: Scalar in [-1, 1] (win probability)
    """

    num_filters: int = 128
    num_residual_blocks: int = 10
    policy_filters: int = 64
    value_filters: int = 32
    max_moves: int = 200  # Maximum legal moves in any position (dynamic action space)

    @nn.compact
    def __call__(self, x, train: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, 9, 9, 3) or (9, 9, 3)
            train: Whether in training mode

        Returns:
            policy_logits: (batch_size, max_moves) or (max_moves,)
            value: (batch_size, 1) or scalar
        """
        # Add batch dimension if needed
        needs_squeeze = False
        if x.ndim == 3:
            x = jnp.expand_dims(x, 0)
            needs_squeeze = True

        # Initial convolution
        x = HexConvBlock(self.num_filters)(x, train=train)

        # Residual tower
        for _ in range(self.num_residual_blocks):
            x = ResidualBlock(self.num_filters)(x, train=train)

        # Policy head
        policy = nn.Conv(features=self.policy_filters, kernel_size=(1, 1))(x)
        policy = nn.BatchNorm(use_running_average=not train)(policy)
        policy = nn.relu(policy)
        policy = policy.reshape((policy.shape[0], -1))  # Flatten
        policy_logits = nn.Dense(self.max_moves)(policy)

        # Value head
        value = nn.Conv(features=self.value_filters, kernel_size=(1, 1))(x)
        value = nn.BatchNorm(use_running_average=not train)(value)
        value = nn.relu(value)
        value = value.reshape((value.shape[0], -1))  # Flatten
        value = nn.Dense(256)(value)
        value = nn.relu(value)
        value = nn.Dense(1)(value)
        value = jnp.tanh(value)  # Output in [-1, 1]

        if needs_squeeze:
            policy_logits = jnp.squeeze(policy_logits, 0)
            value = jnp.squeeze(value, 0)

        return policy_logits, value


class MoveEncoder:
    """
    Encodes Abalone moves as integers and vice versa.

    Move encoding scheme:
    - Each move is encoded based on: starting position, direction, and number of marbles
    - We pre-compute all possible moves and assign them indices
    """

    def __init__(self, board_size: int = 5):
        """Initialize move encoder."""
        self.board_size = board_size
        self.move_to_index: Dict[Tuple, int] = {}
        self.index_to_move: Dict[int, Tuple] = {}

        self._build_move_vocabulary()

    def _build_move_vocabulary(self):
        """Build vocabulary of all possible moves."""
        from abalone_game import Direction, AbaloneGame

        # Get valid positions from a game instance
        temp_game = AbaloneGame(board_size=self.board_size)
        valid_positions = temp_game.valid_positions

        index = 0

        # For each VALID position on the board
        for pos in sorted(valid_positions):
            q, r = pos
            # For each direction
            for direction in Direction:
                # Single marble
                move_repr = ((q, r), direction.value, 1)
                self.move_to_index[move_repr] = index
                self.index_to_move[index] = move_repr
                index += 1

                # Two marbles (in each of 3 line orientations)
                for orientation in range(3):
                    move_repr = ((q, r), direction.value, 2, orientation)
                    self.move_to_index[move_repr] = index
                    self.index_to_move[index] = move_repr
                    index += 1

                # Three marbles (in each of 3 line orientations)
                for orientation in range(3):
                    move_repr = ((q, r), direction.value, 3, orientation)
                    self.move_to_index[move_repr] = index
                    self.index_to_move[index] = move_repr
                    index += 1

        self.num_moves = index
        print(f"Move vocabulary size: {self.num_moves}")

    def encode_move(self, move) -> int:
        """
        Encode a Move object to an integer index.

        Args:
            move: Move object from abalone_game

        Returns:
            Integer index
        """
        from abalone_game import Direction

        marbles = sorted(move.marbles)
        num_marbles = len(marbles)
        direction = move.direction.value

        if num_marbles == 1:
            move_repr = (marbles[0], direction, 1)
        else:
            # Determine orientation based on marble arrangement
            q_diff = marbles[1][0] - marbles[0][0]
            r_diff = marbles[1][1] - marbles[0][1]

            # Map to orientation index (0=E/W, 1=NE/SW, 2=NW/SE)
            if (q_diff, r_diff) in [(1, 0), (-1, 0)]:
                orientation = 0
            elif (q_diff, r_diff) in [(0, 1), (0, -1)]:
                orientation = 1
            else:
                orientation = 2

            move_repr = (marbles[0], direction, num_marbles, orientation)

        return self.move_to_index.get(move_repr, 0)

    def decode_move(self, index: int):
        """
        Decode an integer index to a Move object.

        Args:
            index: Integer move index

        Returns:
            Move object (needs to be validated against game state)
        """
        from abalone_game import Move, Direction

        if index not in self.index_to_move:
            return None

        move_repr = self.index_to_move[index]

        if len(move_repr) == 3:
            # Single marble
            pos, direction, _ = move_repr
            marbles = (pos,)
        else:
            # Multiple marbles
            pos, direction, num_marbles, orientation = move_repr

            # Reconstruct marble positions based on orientation
            if orientation == 0:  # E/W direction
                dq, dr = 1, 0
            elif orientation == 1:  # NE/SW direction
                dq, dr = 0, 1
            else:  # NW/SE direction
                dq, dr = -1, 1

            marbles = [pos]
            for i in range(1, num_marbles):
                next_pos = (pos[0] + i * dq, pos[1] + i * dr)
                marbles.append(next_pos)

            marbles = tuple(sorted(marbles))

        return Move(marbles=marbles, direction=Direction(direction))

    def create_move_mask(self, legal_moves: list) -> np.ndarray:
        """
        Create a mask of legal moves.

        Args:
            legal_moves: List of legal Move objects

        Returns:
            Binary mask array of shape (num_moves,)
        """
        mask = np.zeros(self.num_moves, dtype=np.float32)

        for move in legal_moves:
            try:
                index = self.encode_move(move)
                mask[index] = 1.0
            except (KeyError, IndexError):
                # Move not in vocabulary (shouldn't happen)
                continue

        return mask

    def create_policy_target(self, move_probs: Dict) -> np.ndarray:
        """
        Create policy target from move probabilities.

        Args:
            move_probs: Dict mapping Move objects to probabilities

        Returns:
            Probability array of shape (num_moves,)
        """
        target = np.zeros(self.num_moves, dtype=np.float32)

        for move, prob in move_probs.items():
            try:
                index = self.encode_move(move)
                target[index] = prob
            except (KeyError, IndexError):
                continue

        # Normalize
        total = target.sum()
        if total > 0:
            target = target / total

        return target


def prepare_input(game) -> np.ndarray:
    """
    Prepare neural network input from game state.

    Args:
        game: AbaloneGame instance

    Returns:
        Input array of shape (11, 11, 3)
    """
    board_array = game.get_board_array()  # Shape: (3, 11, 11)

    # Transpose to (11, 11, 3) for CNN
    board_array = np.transpose(board_array, (1, 2, 0))

    # If white is current player, flip the perspective
    if game.current_player.value == 2:  # WHITE
        # Swap black and white channels
        board_array = board_array[:, :, [1, 0, 2]]

    return board_array.astype(np.float32)


def create_network(key: jax.random.PRNGKey, num_filters: int = 128,
                   num_blocks: int = 10, max_moves: int = None) -> Tuple[AbaloneNetwork, Dict[str, Any]]:
    """
    Create and initialize the network.

    Args:
        key: JAX random key
        num_filters: Number of filters in conv layers
        num_blocks: Number of residual blocks
        max_moves: Number of possible moves (if None, uses MoveEncoder to determine)

    Returns:
        (network, params)
    """
    # Use dynamic action space size (200) if not provided
    if max_moves is None:
        max_moves = 200  # Maximum legal moves in any position

    network = AbaloneNetwork(
        num_filters=num_filters,
        num_residual_blocks=num_blocks,
        max_moves=max_moves
    )

    # Initialize with dummy input
    dummy_input = jnp.zeros((1, 9, 9, 3))
    params = network.init(key, dummy_input, train=False)

    return network, params


if __name__ == "__main__":
    # Test network creation
    key = jax.random.PRNGKey(0)
    network, params = create_network(key, num_filters=64, num_blocks=5)

    # Test forward pass
    dummy_input = jnp.zeros((2, 9, 9, 3))
    policy_logits, value = network.apply(params, dummy_input, train=False)

    print(f"Network created successfully!")
    print(f"Policy logits shape: {policy_logits.shape}")
    print(f"Value shape: {value.shape}")

    # Test move encoder
    print("\nTesting move encoder...")
    encoder = MoveEncoder()
    print(f"Total possible move encodings: {encoder.num_moves}")

    # Test with actual game
    from abalone_game import AbaloneGame
    game = AbaloneGame()
    legal_moves = game.get_legal_moves()
    print(f"Legal moves in start position: {len(legal_moves)}")

    # Create mask
    mask = encoder.create_move_mask(legal_moves)
    print(f"Mask sum (number of legal moves): {mask.sum()}")

    # Test encoding/decoding
    move = legal_moves[0]
    encoded = encoder.encode_move(move)
    decoded = encoder.decode_move(encoded)
    print(f"Original move: {move}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

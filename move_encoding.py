"""
Dynamic Move Encoding for Abalone

This module provides efficient move encoding that works with variable-length
action spaces (only legal moves in each position).
"""

import numpy as np
from typing import List, Dict
from abalone_game import Move, AbaloneGame


class DynamicMoveEncoder:
    """
    Encodes moves dynamically based on legal moves in each position.

    Instead of a fixed vocabulary of 2562 moves, we:
    1. Get legal moves for a position (~40-50 moves)
    2. Create a mapping from move -> index for just those moves
    3. Network outputs logits for MAX_LEGAL_MOVES (~200 to be safe)
    4. Mask invalid indices
    """

    MAX_LEGAL_MOVES = 200  # Maximum legal moves in any position

    def __init__(self):
        """Initialize encoder."""
        pass

    def encode_legal_moves(self, legal_moves: List[Move]) -> tuple:
        """
        Encode a list of legal moves.

        Args:
            legal_moves: List of legal Move objects

        Returns:
            (move_indices, num_legal) where:
            - move_indices: dict mapping Move -> index (0 to num_legal-1)
            - num_legal: number of legal moves
        """
        move_to_idx = {move: idx for idx, move in enumerate(legal_moves)}
        return move_to_idx, len(legal_moves)

    def decode_move_index(self, idx: int, legal_moves: List[Move]) -> Move:
        """
        Decode a move index back to a Move object.

        Args:
            idx: Move index (0 to num_legal-1)
            legal_moves: List of legal moves

        Returns:
            Move object
        """
        if 0 <= idx < len(legal_moves):
            return legal_moves[idx]
        return None

    def create_policy_vector(self, move_probs: Dict[Move, float],
                            legal_moves: List[Move]) -> np.ndarray:
        """
        Create policy vector for legal moves.

        Args:
            move_probs: Dictionary mapping Move -> probability
            legal_moves: List of legal moves

        Returns:
            (MAX_LEGAL_MOVES,) array with probabilities for legal moves
        """
        policy = np.zeros(self.MAX_LEGAL_MOVES, dtype=np.float32)

        for idx, move in enumerate(legal_moves):
            if move in move_probs:
                policy[idx] = move_probs[move]

        # Normalize
        total = policy.sum()
        if total > 0:
            policy = policy / total

        return policy

    def create_mask(self, num_legal: int) -> np.ndarray:
        """
        Create mask for legal moves.

        Args:
            num_legal: Number of legal moves

        Returns:
            (MAX_LEGAL_MOVES,) binary mask
        """
        mask = np.zeros(self.MAX_LEGAL_MOVES, dtype=np.float32)
        mask[:num_legal] = 1.0
        return mask


# For compatibility with trainer, create a simple move -> string converter
def move_to_string(move: Move) -> str:
    """Convert move to string for logging."""
    return str(move)


def string_to_move(s: str) -> Move:
    """Convert string back to move (for loading)."""
    # Not needed for training, just for completeness
    raise NotImplementedError("String to move conversion not implemented")


if __name__ == "__main__":
    # Test dynamic encoding
    print("Testing dynamic move encoding...")

    encoder = DynamicMoveEncoder()

    # Create a game and get legal moves
    game = AbaloneGame()
    legal_moves = game.get_legal_moves()

    print(f"Legal moves in start position: {len(legal_moves)}")
    print(f"MAX_LEGAL_MOVES: {encoder.MAX_LEGAL_MOVES}")

    # Encode legal moves
    move_to_idx, num_legal = encoder.encode_legal_moves(legal_moves)

    print(f"Number of legal moves: {num_legal}")
    print(f"First move: {legal_moves[0]}")
    print(f"First move index: {move_to_idx[legal_moves[0]]}")

    # Decode
    decoded = encoder.decode_move_index(0, legal_moves)
    print(f"Decoded move 0: {decoded}")

    # Create policy
    move_probs = {move: 1.0 / len(legal_moves) for move in legal_moves}
    policy = encoder.create_policy_vector(move_probs, legal_moves)

    print(f"\nPolicy shape: {policy.shape}")
    print(f"Policy sum: {policy.sum():.3f}")
    print(f"Non-zero elements: {np.count_nonzero(policy)}")

    # Create mask
    mask = encoder.create_mask(num_legal)
    print(f"\nMask shape: {mask.shape}")
    print(f"Mask sum: {mask.sum():.0f}")

    print("\nDynamic encoding test successful!")

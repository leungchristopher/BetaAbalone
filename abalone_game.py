"""
Abalone Game Logic Implementation

This module implements the complete game logic for Abalone, including:
- Board representation using axial coordinates
- Move generation and validation
- Win condition checking
- Game state management
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from enum import IntEnum
from dataclasses import dataclass


class Player(IntEnum):
    """Player enumeration."""
    EMPTY = 0
    BLACK = 1
    WHITE = 2


class Direction(IntEnum):
    """Six directions on hexagonal grid (axial coordinates)."""
    E = 0   # East: (1, 0)
    NE = 1  # Northeast: (0, 1)
    NW = 2  # Northwest: (-1, 1)
    W = 3   # West: (-1, 0)
    SW = 4  # Southwest: (0, -1)
    SE = 5  # Southeast: (1, -1)


# Direction vectors in axial coordinates (q, r)
DIRECTION_VECTORS = {
    Direction.E: (1, 0),
    Direction.NE: (0, 1),
    Direction.NW: (-1, 1),
    Direction.W: (-1, 0),
    Direction.SW: (0, -1),
    Direction.SE: (1, -1),
}


@dataclass(frozen=True)
class Move:
    """Represents a move in Abalone."""
    marbles: Tuple[Tuple[int, int], ...]  # Sorted tuple of (q, r) positions
    direction: Direction

    def __str__(self):
        return f"Move({len(self.marbles)} marbles {self.marbles[0]} dir={self.direction.name})"


class AbaloneGame:
    """
    Abalone game implementation.

    The board uses axial coordinates (q, r) where:
    - q is the column (increases going right)
    - r is the row (increases going down-left)

    Standard board layout:
         I  H  G  F  E  D  C  B  A
      5  .  .  .  .  .  W  W  W  W  W   (r=-4)
     6  .  .  .  .  W  W  W  W  W  W    (r=-3)
    7  .  .  .  .  .  W  W  .  .        (r=-2)
   8  .  .  .  .  .  .  .  .  .         (r=-1)
  9  .  .  .  .  .  .  .  .  .          (r=0)
        .  .  .  .  .  .  .  .  .       (r=1)
           .  .  .  B  B  .  .  .       (r=2)
              B  B  B  B  B  B           (r=3)
                 B  B  B  B  B            (r=4)
    """

    def __init__(self, board_size: int = 5):
        """Initialize Abalone game with standard setup.

        Args:
            board_size: Radius of the hexagonal board (default 5 for standard Abalone)
        """
        self.board_size = board_size
        self.board = {}  # Dict mapping (q, r) -> Player
        self.current_player = Player.BLACK
        self.marbles_captured = {Player.BLACK: 0, Player.WHITE: 0}
        self.move_history = []

        # Initialize valid positions
        self._init_valid_positions()

        # Set up standard starting position
        self._setup_standard_position()

    def _init_valid_positions(self):
        """Initialize all valid board positions."""
        self.valid_positions = set()

        # Hexagonal board with radius board_size
        for q in range(-self.board_size + 1, self.board_size):
            r1 = max(-self.board_size + 1, -q - self.board_size + 1)
            r2 = min(self.board_size - 1, -q + self.board_size - 1)
            for r in range(r1, r2 + 1):
                self.valid_positions.add((q, r))
                self.board[(q, r)] = Player.EMPTY

    def _setup_standard_position(self):
        """Set up the standard Abalone starting position.

        Standard setup (14 marbles per side):
        - Black (bottom): 5 in row r=4, 6 in row r=3, 3 in center of row r=2
        - White (top): 5 in row r=-4, 6 in row r=-3, 3 in center of row r=-2

        Valid positions:
        r=-4: q from 0 to 4 (5 positions)
        r=-3: q from -1 to 4 (6 positions)
        r=-2: q from -2 to 4 (7 positions)
        r=2: q from -4 to 2 (7 positions)
        r=3: q from -4 to 1 (6 positions)
        r=4: q from -4 to 0 (5 positions)
        """
        # Black marbles (bottom) - 14 marbles
        black_positions = [
            # Row r=4 (5 marbles, q from -4 to 0)
            (-4, 4), (-3, 4), (-2, 4), (-1, 4), (0, 4),
            # Row r=3 (6 marbles, q from -4 to 1)
            (-4, 3), (-3, 3), (-2, 3), (-1, 3), (0, 3), (1, 3),
            # Row r=2 (3 marbles in center, q from -4 to 2)
            (-2, 2), (-1, 2), (0, 2),
        ]

        # White marbles (top) - 14 marbles
        white_positions = [
            # Row r=-4 (5 marbles, q from 0 to 4)
            (0, -4), (1, -4), (2, -4), (3, -4), (4, -4),
            # Row r=-3 (6 marbles, q from -1 to 4)
            (-1, -3), (0, -3), (1, -3), (2, -3), (3, -3), (4, -3),
            # Row r=-2 (3 marbles in center, q from -2 to 4)
            (0, -2), (1, -2), (2, -2),
        ]

        for pos in black_positions:
            if pos in self.valid_positions:
                self.board[pos] = Player.BLACK

        for pos in white_positions:
            if pos in self.valid_positions:
                self.board[pos] = Player.WHITE

    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is on the board."""
        return pos in self.valid_positions

    def get_marble(self, pos: Tuple[int, int]) -> Player:
        """Get the marble at a position."""
        return self.board.get(pos, Player.EMPTY)

    def get_neighbor(self, pos: Tuple[int, int], direction: Direction) -> Optional[Tuple[int, int]]:
        """Get the neighboring position in a direction."""
        dq, dr = DIRECTION_VECTORS[direction]
        neighbor = (pos[0] + dq, pos[1] + dr)
        return neighbor if self.is_valid_position(neighbor) else None

    def get_legal_moves(self, player: Optional[Player] = None) -> List[Move]:
        """Generate all legal moves for the current player."""
        if player is None:
            player = self.current_player

        moves = []
        player_positions = [pos for pos, p in self.board.items() if p == player]

        # For each marble of the current player
        for pos in player_positions:
            # Try all 6 directions
            for direction in Direction:
                # Single marble move (inline or sidestep)
                move = Move(marbles=(pos,), direction=direction)
                if self._is_legal_move(move, player):
                    moves.append(move)

                # Try 2-marble combinations (inline or sidestep)
                neighbor = self.get_neighbor(pos, direction)
                if neighbor and self.get_marble(neighbor) == player:
                    # Found 2 in a line
                    marbles = tuple(sorted([pos, neighbor]))

                    # Try all 6 directions for this group
                    for move_dir in Direction:
                        move = Move(marbles=marbles, direction=move_dir)
                        if self._is_legal_move(move, player):
                            moves.append(move)

                    # Try 3-marble combinations
                    neighbor2 = self.get_neighbor(neighbor, direction)
                    if neighbor2 and self.get_marble(neighbor2) == player:
                        marbles = tuple(sorted([pos, neighbor, neighbor2]))

                        # Try all 6 directions for this group
                        for move_dir in Direction:
                            move = Move(marbles=marbles, direction=move_dir)
                            if self._is_legal_move(move, player):
                                moves.append(move)

        # Remove duplicates
        unique_moves = list(set(moves))
        return unique_moves

    def _is_legal_move(self, move: Move, player: Player) -> bool:
        """Check if a move is legal."""
        marbles = move.marbles
        direction = move.direction

        # Check all marbles belong to player
        for pos in marbles:
            if self.get_marble(pos) != player:
                return False

        # Check marbles form a valid line
        if len(marbles) > 1:
            if not self._marbles_in_line(marbles):
                return False

        # Determine if this is inline or sidestep move
        if len(marbles) == 1:
            is_inline = True
        else:
            marble_direction = self._get_line_direction(marbles)
            is_inline = (marble_direction == direction or
                        marble_direction == Direction((direction.value + 3) % 6))

        if is_inline:
            return self._is_legal_inline_move(marbles, direction, player)
        else:
            return self._is_legal_sidestep_move(marbles, direction, player)

    def _marbles_in_line(self, marbles: Tuple[Tuple[int, int], ...]) -> bool:
        """Check if marbles form a contiguous line."""
        if len(marbles) <= 1:
            return True

        # Check if all marbles are in one of the 3 line directions
        sorted_marbles = sorted(marbles)

        for direction in [Direction.E, Direction.NE, Direction.NW]:
            if self._check_line_in_direction(sorted_marbles, direction):
                return True

        return False

    def _check_line_in_direction(self, sorted_marbles: List[Tuple[int, int]],
                                  direction: Direction) -> bool:
        """Check if sorted marbles form a line in given direction."""
        dq, dr = DIRECTION_VECTORS[direction]

        for i in range(len(sorted_marbles) - 1):
            expected_next = (sorted_marbles[i][0] + dq, sorted_marbles[i][1] + dr)
            if sorted_marbles[i + 1] != expected_next:
                # Try opposite direction
                expected_next = (sorted_marbles[i][0] - dq, sorted_marbles[i][1] - dr)
                if sorted_marbles[i + 1] != expected_next:
                    return False

        return True

    def _get_line_direction(self, marbles: Tuple[Tuple[int, int], ...]) -> Optional[Direction]:
        """Get the direction of a line of marbles."""
        if len(marbles) <= 1:
            return None

        sorted_marbles = sorted(marbles)
        dq = sorted_marbles[1][0] - sorted_marbles[0][0]
        dr = sorted_marbles[1][1] - sorted_marbles[0][1]

        for direction, (vq, vr) in DIRECTION_VECTORS.items():
            if (dq, dr) == (vq, vr) or (dq, dr) == (-vq, -vr):
                return direction

        return None

    def _is_legal_inline_move(self, marbles: Tuple[Tuple[int, int], ...],
                              direction: Direction, player: Player) -> bool:
        """Check if inline move (pushing) is legal."""
        # Find the leading marble
        dq, dr = DIRECTION_VECTORS[direction]

        # Sort marbles by position in movement direction
        def projection(pos):
            return pos[0] * dq + pos[1] * dr

        sorted_marbles = sorted(marbles, key=projection, reverse=True)
        leading_marble = sorted_marbles[0]

        # Check position in front of leading marble
        next_pos = (leading_marble[0] + dq, leading_marble[1] + dr)

        if not self.is_valid_position(next_pos):
            # Pushing off the board is allowed only for opponent marbles
            return False

        next_marble = self.get_marble(next_pos)

        if next_marble == Player.EMPTY:
            # Moving into empty space is always legal
            return True

        if next_marble == player:
            # Can't push own marbles
            return False

        # Pushing opponent marbles - need superiority
        # Count opponent marbles in line
        opponent_count = 0
        check_pos = next_pos
        while self.is_valid_position(check_pos) and self.get_marble(check_pos) == next_marble:
            opponent_count += 1
            check_pos = (check_pos[0] + dq, check_pos[1] + dr)

        # Need more marbles than opponent (max 3 vs 2 or 2 vs 1)
        if len(marbles) <= opponent_count:
            return False

        # Can only push max 2 opponent marbles
        if opponent_count > 2:
            return False

        # Check if there's space to push (or can push off board)
        space_pos = (next_pos[0] + opponent_count * dq, next_pos[1] + opponent_count * dr)

        if not self.is_valid_position(space_pos):
            # Pushing off the board is legal
            return True

        return self.get_marble(space_pos) == Player.EMPTY

    def _is_legal_sidestep_move(self, marbles: Tuple[Tuple[int, int], ...],
                                direction: Direction, player: Player) -> bool:
        """Check if sidestep move is legal."""
        # All destination positions must be empty
        dq, dr = DIRECTION_VECTORS[direction]

        for marble in marbles:
            dest = (marble[0] + dq, marble[1] + dr)
            if not self.is_valid_position(dest):
                return False
            if self.get_marble(dest) != Player.EMPTY:
                return False

        return True

    def make_move(self, move: Move) -> bool:
        """Execute a move if legal. Returns True if successful."""
        if not self._is_legal_move(move, self.current_player):
            return False

        marbles = move.marbles
        direction = move.direction
        dq, dr = DIRECTION_VECTORS[direction]

        # Determine if inline or sidestep
        if len(marbles) == 1:
            is_inline = True
        else:
            marble_direction = self._get_line_direction(marbles)
            is_inline = (marble_direction == direction or
                        marble_direction == Direction((direction.value + 3) % 6))

        if is_inline:
            self._execute_inline_move(marbles, direction)
        else:
            self._execute_sidestep_move(marbles, direction)

        # Switch player
        self.current_player = Player.WHITE if self.current_player == Player.BLACK else Player.BLACK
        self.move_history.append(move)

        return True

    def _execute_inline_move(self, marbles: Tuple[Tuple[int, int], ...],
                            direction: Direction):
        """Execute an inline move."""
        dq, dr = DIRECTION_VECTORS[direction]

        # Sort marbles by projection in movement direction (process from front)
        def projection(pos):
            return pos[0] * dq + pos[1] * dr

        sorted_marbles = sorted(marbles, key=projection, reverse=True)

        # Check what we're pushing
        leading_marble = sorted_marbles[0]
        next_pos = (leading_marble[0] + dq, leading_marble[1] + dr)

        opponent = Player.WHITE if self.current_player == Player.BLACK else Player.BLACK

        # Find all marbles being pushed (opponent marbles in front)
        pushed_marbles = []
        check_pos = next_pos
        while self.is_valid_position(check_pos) and self.get_marble(check_pos) == opponent:
            pushed_marbles.append(check_pos)
            check_pos = (check_pos[0] + dq, check_pos[1] + dr)

        # Move pushed marbles first (from front to back)
        for pushed_pos in reversed(pushed_marbles):
            dest = (pushed_pos[0] + dq, pushed_pos[1] + dr)
            self.board[pushed_pos] = Player.EMPTY

            if self.is_valid_position(dest):
                self.board[dest] = opponent
            else:
                # Marble pushed off board
                self.marbles_captured[self.current_player] += 1

        # Move our marbles (from front to back)
        for marble_pos in sorted_marbles:
            dest = (marble_pos[0] + dq, marble_pos[1] + dr)
            self.board[marble_pos] = Player.EMPTY
            self.board[dest] = self.current_player

    def _execute_sidestep_move(self, marbles: Tuple[Tuple[int, int], ...],
                               direction: Direction):
        """Execute a sidestep move."""
        dq, dr = DIRECTION_VECTORS[direction]

        # Clear old positions
        for pos in marbles:
            self.board[pos] = Player.EMPTY

        # Place in new positions
        for pos in marbles:
            dest = (pos[0] + dq, pos[1] + dr)
            self.board[dest] = self.current_player

    def is_game_over(self) -> bool:
        """Check if game is over (6 marbles captured)."""
        return (self.marbles_captured[Player.BLACK] >= 6 or
                self.marbles_captured[Player.WHITE] >= 6)

    def get_winner(self) -> Optional[Player]:
        """Get the winner if game is over."""
        if self.marbles_captured[Player.BLACK] >= 6:
            return Player.BLACK
        if self.marbles_captured[Player.WHITE] >= 6:
            return Player.WHITE
        return None

    def get_board_array(self) -> np.ndarray:
        """
        Get board as numpy array for neural network input.

        Returns 3 channels (11x11):
        - Channel 0: Black marbles
        - Channel 1: White marbles
        - Channel 2: Valid positions
        """
        size = 2 * self.board_size - 1
        board_array = np.zeros((3, size, size), dtype=np.float32)

        for q in range(-self.board_size + 1, self.board_size):
            for r in range(-self.board_size + 1, self.board_size):
                pos = (q, r)
                if not self.is_valid_position(pos):
                    continue

                # Convert axial to array indices
                x = q + self.board_size - 1
                y = r + self.board_size - 1

                marble = self.get_marble(pos)
                if marble == Player.BLACK:
                    board_array[0, y, x] = 1.0
                elif marble == Player.WHITE:
                    board_array[1, y, x] = 1.0

                board_array[2, y, x] = 1.0  # Valid position

        return board_array

    def clone(self) -> 'AbaloneGame':
        """Create a deep copy of the game state."""
        new_game = AbaloneGame.__new__(AbaloneGame)
        new_game.board_size = self.board_size
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.marbles_captured = self.marbles_captured.copy()
        new_game.move_history = self.move_history.copy()
        new_game.valid_positions = self.valid_positions.copy()
        return new_game

    def __str__(self) -> str:
        """String representation of the board."""
        lines = []
        lines.append(f"Current player: {self.current_player.name}")
        lines.append(f"Captured - Black: {self.marbles_captured[Player.BLACK]}, "
                    f"White: {self.marbles_captured[Player.WHITE]}")
        lines.append("")

        # Print board
        for r in range(-self.board_size + 1, self.board_size):
            indent = " " * abs(r)
            line = indent
            for q in range(-self.board_size + 1, self.board_size):
                pos = (q, r)
                if not self.is_valid_position(pos):
                    continue
                marble = self.get_marble(pos)
                if marble == Player.BLACK:
                    line += "B "
                elif marble == Player.WHITE:
                    line += "W "
                else:
                    line += ". "
            lines.append(line)

        return "\n".join(lines)


if __name__ == "__main__":
    # Test basic functionality
    game = AbaloneGame()
    print(game)
    print(f"\nNumber of legal moves: {len(game.get_legal_moves())}")

    # Test a simple move
    moves = game.get_legal_moves()
    if moves:
        print(f"\nMaking move: {moves[0]}")
        game.make_move(moves[0])
        print(game)

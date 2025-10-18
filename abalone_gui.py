"""
Abalone GUI using Pygame

Provides a graphical interface to visualize Abalone games.
Can be used for human play, watching AI games, or debugging.
"""

import pygame
import math
from typing import Optional, Tuple, List
from abalone_game import AbaloneGame, Player, Move, Direction


class AbaloneGUI:
    """Pygame-based GUI for Abalone."""

    def __init__(self, game: AbaloneGame, width: int = 800, height: int = 800):
        """Initialize the GUI.

        Args:
            game: AbaloneGame instance to visualize
            width: Window width in pixels
            height: Window height in pixels
        """
        pygame.init()
        self.game = game
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Abalone")

        # Colors
        self.BG_COLOR = (245, 222, 179)  # Wheat
        self.GRID_COLOR = (139, 90, 43)  # Brown
        self.BLACK_MARBLE = (40, 40, 40)
        self.WHITE_MARBLE = (240, 240, 240)
        self.HIGHLIGHT_COLOR = (100, 200, 100)
        self.VALID_MOVE_COLOR = (150, 150, 255)

        # Layout
        self.center_x = width // 2
        self.center_y = height // 2
        self.hex_radius = min(width, height) // 15

        # Interaction state
        self.selected_marbles: List[Tuple[int, int]] = []
        self.valid_moves: List[Move] = []
        self.show_legal_moves = False

        # Font for text
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

    def axial_to_pixel(self, q: int, r: int) -> Tuple[float, float]:
        """Convert axial coordinates to pixel coordinates.

        Uses flat-top hexagon layout (flat edge at bottom).
        """
        x = self.center_x + self.hex_radius * (math.sqrt(3) * q + math.sqrt(3)/2 * r)
        y = self.center_y + self.hex_radius * (3/2 * r)
        return x, y

    def pixel_to_axial(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """Convert pixel coordinates to axial coordinates (approximate).

        Uses flat-top hexagon layout.
        """
        # Convert to fractional axial coordinates
        q = (math.sqrt(3)/3 * (x - self.center_x) - 1/3 * (y - self.center_y)) / self.hex_radius
        r = (2/3 * (y - self.center_y)) / self.hex_radius

        # Round to nearest hexagon
        return self._round_axial(q, r)

    def _round_axial(self, q: float, r: float) -> Tuple[int, int]:
        """Round fractional axial coordinates to nearest hex."""
        s = -q - r

        # Round to cube coordinates
        q_round = round(q)
        r_round = round(r)
        s_round = round(s)

        # Restore invariant
        q_diff = abs(q_round - q)
        r_diff = abs(r_round - r)
        s_diff = abs(s_round - s)

        if q_diff > r_diff and q_diff > s_diff:
            q_round = -r_round - s_round
        elif r_diff > s_diff:
            r_round = -q_round - s_round

        return (q_round, r_round)

    def draw_hexagon(self, x: float, y: float, color: Tuple[int, int, int],
                     filled: bool = True, width: int = 2):
        """Draw a flat-top hexagon at pixel coordinates."""
        points = []
        for i in range(6):
            # Offset by 30 degrees (pi/6) to make flat edge at bottom
            angle = math.pi / 3 * i + math.pi / 6
            px = x + self.hex_radius * math.cos(angle)
            py = y + self.hex_radius * math.sin(angle)
            points.append((px, py))

        if filled:
            pygame.draw.polygon(self.screen, color, points)
        else:
            pygame.draw.polygon(self.screen, color, points, width)

    def draw_marble(self, x: float, y: float, player: Player, selected: bool = False):
        """Draw a marble at pixel coordinates."""
        radius = self.hex_radius * 0.7

        if player == Player.BLACK:
            color = self.BLACK_MARBLE
        else:
            color = self.WHITE_MARBLE

        # Draw marble
        pygame.draw.circle(self.screen, color, (int(x), int(y)), int(radius))

        # Add shading for 3D effect
        pygame.draw.circle(self.screen, color, (int(x), int(y)), int(radius), 2)

        # Highlight if selected
        if selected:
            pygame.draw.circle(self.screen, self.HIGHLIGHT_COLOR,
                             (int(x), int(y)), int(radius + 5), 3)

    def draw_board(self):
        """Draw the game board."""
        self.screen.fill(self.BG_COLOR)

        # Draw all valid positions
        for q in range(-self.game.board_size + 1, self.game.board_size):
            for r in range(-self.game.board_size + 1, self.game.board_size):
                if not self.game.is_valid_position((q, r)):
                    continue

                x, y = self.axial_to_pixel(q, r)

                # Draw circular position marker (small circle to indicate valid position)
                position_radius = self.hex_radius * 0.15
                pygame.draw.circle(self.screen, self.GRID_COLOR,
                                 (int(x), int(y)), int(position_radius))

                # Draw marble if present
                marble = self.game.get_marble((q, r))
                if marble != Player.EMPTY:
                    selected = (q, r) in self.selected_marbles
                    self.draw_marble(x, y, marble, selected)

        # Draw valid move indicators if marbles selected
        if self.show_legal_moves and self.selected_marbles:
            self._draw_valid_move_indicators()

    def _draw_valid_move_indicators(self):
        """Draw indicators for valid moves from selected marbles."""
        for move in self.valid_moves:
            # Get destination positions
            dq, dr = {
                Direction.E: (1, 0), Direction.NE: (0, 1), Direction.NW: (-1, 1),
                Direction.W: (-1, 0), Direction.SW: (0, -1), Direction.SE: (1, -1),
            }[move.direction]

            # Draw small circle at first marble's destination
            first_marble = move.marbles[0]
            dest_q = first_marble[0] + dq
            dest_r = first_marble[1] + dr

            if self.game.is_valid_position((dest_q, dest_r)):
                x, y = self.axial_to_pixel(dest_q, dest_r)
                pygame.draw.circle(self.screen, self.VALID_MOVE_COLOR,
                                 (int(x), int(y)), 8)

    def draw_info(self):
        """Draw game information."""
        # Current player
        player_text = f"Current: {self.game.current_player.name}"
        text_surface = self.font.render(player_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 10))

        # Captured marbles
        captured_text = f"Captured - B: {self.game.marbles_captured[Player.BLACK]} W: {self.game.marbles_captured[Player.WHITE]}"
        text_surface = self.small_font.render(captured_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 50))

        # Move count
        move_text = f"Moves: {len(self.game.move_history)}"
        text_surface = self.small_font.render(move_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 75))

        # Game over
        if self.game.is_game_over():
            winner = self.game.get_winner()
            winner_text = f"GAME OVER - {winner.name} WINS!"
            text_surface = self.font.render(winner_text, True, (255, 0, 0))
            text_rect = text_surface.get_rect(center=(self.width // 2, self.height - 50))
            self.screen.blit(text_surface, text_rect)

        # Instructions
        instructions = [
            "Click marble to select",
            "Click again to add to selection",
            "Press direction key to move:",
            "Q=NW W=NE E=E",
            "A=W S=SE D=SW",
            "R=Reset selection",
            "Space=Show legal moves"
        ]

        y_offset = self.height - 200
        for instruction in instructions:
            text_surface = self.small_font.render(instruction, True, (60, 60, 60))
            self.screen.blit(text_surface, (self.width - 250, y_offset))
            y_offset += 25

    def handle_click(self, pos: Tuple[int, int]):
        """Handle mouse click."""
        axial_pos = self.pixel_to_axial(pos[0], pos[1])

        if not axial_pos or not self.game.is_valid_position(axial_pos):
            return

        marble = self.game.get_marble(axial_pos)

        # Can only select current player's marbles
        if marble != self.game.current_player:
            # Clicked empty space or opponent - clear selection
            self.selected_marbles = []
            self.valid_moves = []
            return

        # Toggle selection
        if axial_pos in self.selected_marbles:
            self.selected_marbles.remove(axial_pos)
        else:
            # Add to selection (max 3 marbles)
            if len(self.selected_marbles) < 3:
                self.selected_marbles.append(axial_pos)

        # Update valid moves for this selection
        self._update_valid_moves()

    def _update_valid_moves(self):
        """Update the list of valid moves from current selection."""
        if not self.selected_marbles:
            self.valid_moves = []
            return

        # Get all legal moves for current player
        all_moves = self.game.get_legal_moves()

        # Filter to moves matching selected marbles
        marbles_set = set(self.selected_marbles)
        self.valid_moves = [
            move for move in all_moves
            if set(move.marbles) == marbles_set
        ]

    def handle_key(self, key: int) -> bool:
        """Handle keyboard input. Returns True if move was made."""
        # Direction keys
        direction_map = {
            pygame.K_q: Direction.NW,
            pygame.K_w: Direction.NE,
            pygame.K_e: Direction.E,
            pygame.K_a: Direction.W,
            pygame.K_s: Direction.SE,
            pygame.K_d: Direction.SW,
        }

        if key in direction_map:
            direction = direction_map[key]
            return self._try_move(direction)

        # Reset selection
        if key == pygame.K_r:
            self.selected_marbles = []
            self.valid_moves = []

        # Toggle legal moves display
        if key == pygame.K_SPACE:
            self.show_legal_moves = not self.show_legal_moves

        return False

    def _try_move(self, direction: Direction) -> bool:
        """Try to make a move in the given direction."""
        if not self.selected_marbles:
            return False

        # Find matching move
        marbles_tuple = tuple(sorted(self.selected_marbles))
        move = Move(marbles=marbles_tuple, direction=direction)

        if self.game.make_move(move):
            self.selected_marbles = []
            self.valid_moves = []
            return True

        return False

    def render(self):
        """Render the current game state."""
        self.draw_board()
        self.draw_info()
        pygame.display.flip()

    def run_game_loop(self, ai_player=None, move_delay: int = 500):
        """
        Run the interactive game loop.

        Args:
            ai_player: Optional AI agent (callable that takes game state and returns Move)
            move_delay: Delay in ms between AI moves
        """
        clock = pygame.time.Clock()
        running = True
        last_ai_move_time = 0

        while running:
            current_time = pygame.time.get_ticks()

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_click(event.pos)

                elif event.type == pygame.KEYDOWN:
                    self.handle_key(event.key)

            # AI move
            if ai_player and not self.game.is_game_over():
                if current_time - last_ai_move_time > move_delay:
                    move = ai_player(self.game)
                    if move:
                        self.game.make_move(move)
                        last_ai_move_time = current_time

            # Render
            self.render()
            clock.tick(60)  # 60 FPS

        pygame.quit()


def random_ai(game: AbaloneGame) -> Optional[Move]:
    """Simple random AI for testing."""
    import random
    moves = game.get_legal_moves()
    return random.choice(moves) if moves else None


def main():
    """Run a demo game."""
    game = AbaloneGame()
    gui = AbaloneGUI(game)

    # Run with human vs human (no AI)
    # Or uncomment below to watch AI vs AI:
    # gui.run_game_loop(ai_player=random_ai, move_delay=1000)

    gui.run_game_loop()


if __name__ == "__main__":
    main()

"""
Simplified MCTS Player using Python-based tree search

This uses a Python-based MCTS implementation that works with the existing
Abalone game logic, avoiding JAX tracing issues. The network is still used
for evaluation, but tree search is done in Python.
"""

import numpy as np
import jax.numpy as jnp
from typing import List, Tuple, Optional
from dataclasses import dataclass
import math

from abalone_game import AbaloneGame, Move, Player
from abalone_network import prepare_input


@dataclass
class MCTSConfig:
    """Configuration for MCTS."""
    num_simulations: int = 800
    temperature: float = 1.0
    c_puct: float = 1.0
    dirichlet_alpha: float = 0.3
    exploration_fraction: float = 0.25
    batch_size: int = 8  # Number of parallel simulations per batch
    use_parallel: bool = True  # Enable parallel MCTS with batching


class Node:
    """
    MCTS tree node - optimized to avoid game state cloning.

    Instead of storing a cloned game state in each node, we store only the move
    that led to this node. Game states are reconstructed on-demand by replaying
    moves from the root.

    Supports virtual loss for parallel MCTS.
    """

    def __init__(self, parent=None, move: Move = None, prior: float = 0.0):
        self.parent = parent
        self.move = move  # Move that led to this node
        self.prior = prior  # Prior probability from network

        self.children: List['Node'] = []
        self.visit_count = 0
        self.value_sum = 0.0
        self.virtual_loss = 0  # For parallel MCTS

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        """Average value with virtual loss penalty."""
        if self.visit_count + self.virtual_loss == 0:
            return 0.0
        # Virtual loss makes node appear worse during parallel selection
        return self.value_sum / (self.visit_count + self.virtual_loss)

    def ucb_score(self, parent_visit_count: int, c_puct: float) -> float:
        """Calculate UCB score for this node."""
        if self.visit_count == 0:
            u = float('inf')
        else:
            u = c_puct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)

        q = self.value()
        return q + u

    def get_move_path(self) -> List[Move]:
        """Get the sequence of moves from root to this node."""
        path = []
        node = self
        while node.parent is not None:
            path.append(node.move)
            node = node.parent
        return list(reversed(path))


class SimpleMCTSPlayer:
    """Simple MCTS player using Python tree search."""

    def __init__(self, network, params, config: MCTSConfig = None):
        """
        Initialize MCTS player.

        Args:
            network: Flax neural network
            params: Network parameters
            config: MCTS configuration
        """
        self.network = network
        self.params = params
        self.config = config or MCTSConfig()
        self.root_game = None  # Store root game state

    def _get_game_state(self, node: Node) -> AbaloneGame:
        """
        Reconstruct game state for a node by replaying moves from root.

        This is more efficient than cloning the game state at each node.
        """
        game = self.root_game.clone()
        move_path = node.get_move_path()
        for move in move_path:
            game.make_move(move)
        return game

    def select_move(self, game: AbaloneGame, temperature: Optional[float] = None,
                   add_noise: bool = False) -> Tuple[Move, np.ndarray]:
        """
        Select a move using MCTS.

        Args:
            game: Current game state
            temperature: Sampling temperature (None = use config)
            add_noise: Whether to add Dirichlet noise to root

        Returns:
            (selected_move, visit_counts)
        """
        if temperature is None:
            temperature = self.config.temperature

        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return None, None

        if len(legal_moves) == 1:
            return legal_moves[0], np.array([1.0])

        # Store root game for state reconstruction
        self.root_game = game

        # Create root node (no game state stored)
        root = Node()

        # Get network evaluation for root
        root_prior, root_value = self._evaluate(game)

        # Add Dirichlet noise if requested
        if add_noise:
            noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(legal_moves))
            root_prior = (1 - self.config.exploration_fraction) * root_prior + \
                        self.config.exploration_fraction * noise

        # Expand root
        self._expand(root, legal_moves, root_prior)

        # Run simulations (parallel or sequential based on config)
        if self.config.use_parallel and self.config.batch_size > 1:
            self._run_parallel_simulations(root)
        else:
            # Sequential MCTS (original implementation)
            for _ in range(self.config.num_simulations):
                self._simulate(root)

        # Select move based on visit counts
        visit_counts = np.array([child.visit_count for child in root.children])

        if temperature == 0:
            # Argmax
            action_idx = int(np.argmax(visit_counts))
        else:
            # Sample proportional to visit counts^(1/temperature)
            probs = visit_counts ** (1.0 / temperature)
            probs = probs / probs.sum()
            action_idx = np.random.choice(len(probs), p=probs)

        selected_move = root.children[action_idx].move

        # Return visit count distribution as policy
        policy = visit_counts / visit_counts.sum()

        return selected_move, policy

    def _evaluate(self, game: AbaloneGame) -> Tuple[np.ndarray, float]:
        """
        Evaluate position with network.

        Returns:
            (prior_probs, value) - prior probabilities for legal moves, value estimate
        """
        state_input = prepare_input(game)
        state_input = jnp.expand_dims(state_input, 0)

        # Handle both old-style (dict) and new-style (FrozenDict) params
        if isinstance(self.params, dict) and 'params' in self.params:
            # Params dict with batch_stats
            policy_logits, value = self.network.apply(
                self.params, state_input, train=False
            )
        else:
            # Old-style params (just the params, no batch_stats)
            policy_logits, value = self.network.apply(
                self.params, state_input, train=False
            )

        policy_logits = jnp.squeeze(policy_logits)
        value = float(jnp.squeeze(value))

        # Get legal moves
        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return np.array([]), value

        # Extract logits for legal moves (first num_legal indices)
        num_legal = len(legal_moves)
        legal_logits = policy_logits[:num_legal]

        # Softmax to get probabilities
        legal_logits_np = np.array(legal_logits)
        exp_logits = np.exp(legal_logits_np - np.max(legal_logits_np))
        prior = exp_logits / exp_logits.sum()

        return prior, value

    def _evaluate_batch(self, games: List[AbaloneGame]) -> Tuple[List[np.ndarray], List[float]]:
        """
        Batch evaluate multiple positions (for parallel MCTS).

        Args:
            games: List of game states to evaluate

        Returns:
            (priors_list, values_list) - lists of priors and values for each game
        """
        if not games:
            return [], []

        # Prepare batch of inputs
        batch_inputs = []
        legal_moves_list = []

        for game in games:
            state_input = prepare_input(game)
            batch_inputs.append(state_input)
            legal_moves_list.append(game.get_legal_moves())

        # Stack into batch (batch_size, 9, 9, 3)
        batch_inputs = jnp.stack(batch_inputs)

        # Single network forward pass for entire batch
        if isinstance(self.params, dict) and 'params' in self.params:
            policy_logits_batch, values_batch = self.network.apply(
                self.params, batch_inputs, train=False
            )
        else:
            policy_logits_batch, values_batch = self.network.apply(
                self.params, batch_inputs, train=False
            )

        # Squeeze values if needed
        values_batch = jnp.squeeze(values_batch)
        if values_batch.ndim == 0:  # Single value case
            values_batch = jnp.expand_dims(values_batch, 0)

        # Process outputs for each position
        priors_list = []
        values_list = []

        for i, legal_moves in enumerate(legal_moves_list):
            if not legal_moves:
                priors_list.append(np.array([]))
                values_list.append(0.0)
                continue

            # Extract priors for this position
            num_legal = len(legal_moves)
            legal_logits = policy_logits_batch[i, :num_legal]

            # Softmax
            legal_logits_np = np.array(legal_logits)
            exp_logits = np.exp(legal_logits_np - np.max(legal_logits_np))
            prior = exp_logits / exp_logits.sum()

            priors_list.append(prior)
            values_list.append(float(values_batch[i]))

        return priors_list, values_list

    def _expand(self, node: Node, legal_moves: List[Move], priors: np.ndarray):
        """Expand node with children (without cloning game states)."""
        for move, prior in zip(legal_moves, priors):
            child = Node(parent=node, move=move, prior=prior)
            node.children.append(child)

    def _simulate(self, root: Node):
        """Run one MCTS simulation (sequential version)."""
        node = root
        search_path = [node]

        # Selection: traverse tree using UCB
        # We only need to check game state when we reach a leaf
        while node.is_expanded():
            node = self._select_child(node)
            search_path.append(node)

        # Reconstruct game state for this node
        game = self._get_game_state(node)

        # Check if game is over
        if game.is_game_over():
            # Terminal node
            winner = game.get_winner()
            if winner is None:
                value = 0.0
            elif winner == game.current_player:
                value = -1.0  # Lost (opponent won)
            else:
                value = 1.0  # Won
        else:
            # Expansion and evaluation
            legal_moves = game.get_legal_moves()
            if legal_moves:
                priors, value = self._evaluate(game)
                self._expand(node, legal_moves, priors)

        # Backpropagation
        self._backpropagate(search_path, value)

    def _run_parallel_simulations(self, root: Node):
        """
        Run MCTS simulations in parallel batches with virtual loss.

        This method:
        1. Selects batch_size leaf nodes (applying virtual loss)
        2. Evaluates them all at once (batched network call)
        3. Expands and backpropagates (removing virtual loss)
        """
        batch_size = self.config.batch_size
        num_batches = (self.config.num_simulations + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            # Determine current batch size (last batch may be smaller)
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, self.config.num_simulations)
            current_batch_size = batch_end - batch_start

            # Phase 1: Parallel selection with virtual loss
            leaf_nodes = []
            search_paths = []

            for _ in range(current_batch_size):
                node, path = self._select_leaf_with_virtual_loss(root)
                leaf_nodes.append(node)
                search_paths.append(path)

            # Phase 2: Reconstruct game states and categorize
            games_to_eval = []
            eval_indices = []
            terminal_results = {}

            for i, node in enumerate(leaf_nodes):
                game = self._get_game_state(node)

                if game.is_game_over():
                    # Terminal node - compute value directly
                    winner = game.get_winner()
                    if winner is None:
                        value = 0.0
                    elif winner == game.current_player:
                        value = -1.0  # Lost (opponent won)
                    else:
                        value = 1.0  # Won
                    terminal_results[i] = (None, value)
                else:
                    # Non-terminal - needs evaluation
                    games_to_eval.append(game)
                    eval_indices.append(i)

            # Phase 3: Batch evaluate non-terminal nodes
            if games_to_eval:
                priors_list, values_list = self._evaluate_batch(games_to_eval)

                # Phase 4: Expand and backprop
                for eval_idx, (i, prior, value) in enumerate(zip(eval_indices, priors_list, values_list)):
                    node = leaf_nodes[i]
                    path = search_paths[i]
                    game = games_to_eval[eval_idx]

                    # Expand if there are legal moves
                    legal_moves = game.get_legal_moves()
                    if legal_moves and len(prior) > 0:
                        self._expand(node, legal_moves, prior)

                    # Backprop with virtual loss removal
                    self._backprop_remove_virtual_loss(path, value)

            # Phase 5: Backprop terminal nodes
            for i, (prior, value) in terminal_results.items():
                path = search_paths[i]
                self._backprop_remove_virtual_loss(path, value)

    def _select_child(self, node: Node) -> Node:
        """Select best child using UCB."""
        best_score = -float('inf')
        best_child = None

        for child in node.children:
            score = child.ucb_score(node.visit_count, self.config.c_puct)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _backpropagate(self, search_path: List[Node], value: float):
        """Backpropagate value up the tree."""
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value  # Flip value for opponent

    def _select_leaf_with_virtual_loss(self, root: Node) -> Tuple[Node, List[Node]]:
        """
        Select leaf node for parallel MCTS, applying virtual loss along the path.

        Virtual loss makes the path temporarily look worse so that parallel
        threads will explore different paths.

        Returns:
            (leaf_node, search_path)
        """
        node = root
        search_path = [node]

        # Apply virtual loss to root
        node.virtual_loss += 1

        # Traverse tree using UCB (which accounts for virtual loss via value())
        while node.is_expanded():
            node = self._select_child(node)
            node.virtual_loss += 1
            search_path.append(node)

        return node, search_path

    def _backprop_remove_virtual_loss(self, search_path: List[Node], value: float):
        """
        Backpropagate value and remove virtual losses.

        This is used in parallel MCTS after evaluation completes.
        """
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            node.virtual_loss -= 1  # Remove the virtual loss
            value = -value  # Flip value for opponent


if __name__ == "__main__":
    # Test simple MCTS
    from abalone_network import create_network
    import jax

    print("Creating network...")
    key = jax.random.PRNGKey(42)
    network, params = create_network(key, num_filters=32, num_blocks=2)

    print("Creating MCTS player...")
    config = MCTSConfig(num_simulations=50)
    player = SimpleMCTSPlayer(network, params, config)

    print("Testing move selection...")
    game = AbaloneGame()

    move, policy = player.select_move(game, temperature=1.0, add_noise=True)

    print(f"Selected move: {move}")
    print(f"Policy shape: {policy.shape}")
    print(f"Policy sum: {policy.sum():.3f}")
    print("Simple MCTS test successful!")

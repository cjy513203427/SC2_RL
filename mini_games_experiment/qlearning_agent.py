"""
Reusable Q-Learning agent implementation for PySC2 mini-games.

This module intentionally DOES NOT define absl flags, so it can be imported safely
from different scripts (train / evaluate / test) without triggering DuplicateFlagError.
"""

from __future__ import annotations

import os
import pickle
from collections import defaultdict
from typing import DefaultDict, List, Optional, Tuple

import numpy as np
from pysc2.lib import actions

# Action IDs
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ALL = [0]  # Select all units
_NOT_QUEUED = [0]


class QLearningAgent:
    """Tabular Q-Learning Agent for the MoveToBeacon-style tasks."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.99,
        grid_size: int = 8,
        screen_size: int = 84,
        use_relative_state: bool = True,
        action_mode: str = "delta",
        action_step_pixels: int = 12,
        reward_shaping: bool = True,
        distance_reward_scale: float = 0.02,
    ) -> None:
        self.learning_rate = float(learning_rate)
        self.discount_factor = float(discount_factor)
        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.grid_size = int(grid_size)
        self.screen_size = int(screen_size)
        self.use_relative_state = bool(use_relative_state)
        self.action_mode = str(action_mode)
        self.action_step_pixels = int(action_step_pixels)
        self.reward_shaping = bool(reward_shaping)
        self.distance_reward_scale = float(distance_reward_scale)

        # Q-table: dictionary mapping (state, action) -> Q-value
        self.q_table: DefaultDict[Tuple[Tuple[int, ...], int], float] = defaultdict(float)

        # Track env reward (sparse, true task score) separately from shaped reward (dense learning signal)
        self.total_env_reward = 0.0
        self.total_shaped_reward = 0.0
        self.steps = 0
        self.previous_state: Optional[Tuple[int, ...]] = None
        self.previous_action: Optional[int] = None
        self.prev_distance: Optional[float] = None

        # Define discrete actions
        self.actions: List[Tuple[int, int]] = self._create_action_space()
        self.num_actions = len(self.actions)

        # Track if army is selected
        self.army_selected = False

    def _create_action_space(self) -> List[Tuple[int, int]]:
        """Create discrete action space.

        - delta: 9 relative moves (dx,dy) in pixels from current marine position.
        - grid_absolute: move to fixed grid points that cover the full screen (0..screen_size-1).
        """
        if self.action_mode == "delta":
            step = max(1, int(self.action_step_pixels))
            deltas = [-1, 0, 1]
            actions_list: List[Tuple[int, int]] = []
            for dy in deltas:
                for dx in deltas:
                    actions_list.append((dx * step, dy * step))
            return actions_list

        coords = np.linspace(0, self.screen_size - 1, self.grid_size)
        coords_int = [int(round(c)) for c in coords]
        actions_list = []
        for y in coords_int:
            for x in coords_int:
                actions_list.append((x, y))
        return actions_list

    def _get_positions(self, obs):
        """Extract marine and beacon center positions in pixel coordinates.

        Returns:
            (marine_x, marine_y, beacon_x, beacon_y, marine_found, beacon_found)
        """
        player_relative = obs.observation.feature_screen.player_relative

        marine_y, marine_x = (player_relative == 1).nonzero()
        beacon_y, beacon_x = (player_relative == 3).nonzero()

        marine_found = len(marine_x) > 0
        beacon_found = len(beacon_x) > 0

        if marine_found:
            marine_center_x = int(np.mean(marine_x))
            marine_center_y = int(np.mean(marine_y))
        else:
            marine_center_x = self.screen_size // 2
            marine_center_y = self.screen_size // 2

        if beacon_found:
            beacon_center_x = int(np.mean(beacon_x))
            beacon_center_y = int(np.mean(beacon_y))
        else:
            beacon_center_x = self.screen_size // 2
            beacon_center_y = self.screen_size // 2

        return (
            marine_center_x,
            marine_center_y,
            beacon_center_x,
            beacon_center_y,
            marine_found,
            beacon_found,
        )

    def _get_state(self, obs) -> Tuple[int, ...]:
        """Extract and discretize state from observation."""
        marine_center_x, marine_center_y, beacon_center_x, beacon_center_y, _, _ = self._get_positions(obs)

        cell_size = max(1, self.screen_size // self.grid_size)
        marine_grid_x = min(marine_center_x // cell_size, self.grid_size - 1)
        marine_grid_y = min(marine_center_y // cell_size, self.grid_size - 1)
        beacon_grid_x = min(beacon_center_x // cell_size, self.grid_size - 1)
        beacon_grid_y = min(beacon_center_y // cell_size, self.grid_size - 1)

        if self.use_relative_state:
            dx = int(beacon_grid_x - marine_grid_x)
            dy = int(beacon_grid_y - marine_grid_y)
            return (dx, dy)

        return (marine_grid_x, marine_grid_y, beacon_grid_x, beacon_grid_y)

    def _get_max_q_value(self, state: Tuple[int, ...]) -> float:
        q_values = [self.q_table[(state, a)] for a in range(self.num_actions)]
        return max(q_values)

    def _get_best_action(self, state: Tuple[int, ...]) -> int:
        q_values = [self.q_table[(state, a)] for a in range(self.num_actions)]
        max_q = max(q_values)
        best_actions = [a for a in range(self.num_actions) if q_values[a] == max_q]
        return int(np.random.choice(best_actions))

    def choose_action(self, state: Tuple[int, ...]) -> int:
        """Choose action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.num_actions))
        return self._get_best_action(state)

    def update_q_value(
        self,
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Tuple[int, ...],
        done: bool,
    ) -> None:
        """Q-learning update:
        Q(s,a) = Q(s,a) + α[r + γ*max_a' Q(s',a') - Q(s,a)]
        """
        current_q = self.q_table[(state, action)]
        if done:
            target_q = float(reward)
        else:
            max_next_q = self._get_max_q_value(next_state)
            target_q = float(reward) + self.discount_factor * max_next_q
        self.q_table[(state, action)] = current_q + self.learning_rate * (target_q - current_q)

    def reset(self) -> None:
        self.total_env_reward = 0.0
        self.total_shaped_reward = 0.0
        self.steps = 0
        self.previous_state = None
        self.previous_action = None
        self.prev_distance = None
        self.army_selected = False

    def step(self, obs):
        """Return a PySC2 action."""
        self.steps += 1
        env_reward = float(obs.reward)
        done = obs.last()

        current_state = self._get_state(obs)

        marine_x, marine_y, beacon_x, beacon_y, marine_found, beacon_found = self._get_positions(obs)
        current_distance: Optional[float] = None
        if marine_found and beacon_found:
            current_distance = float(np.hypot(marine_x - beacon_x, marine_y - beacon_y))

        shaped_reward = float(env_reward)
        if self.reward_shaping and (self.prev_distance is not None) and (current_distance is not None):
            shaped_reward += self.distance_reward_scale * (self.prev_distance - current_distance)

        self.total_env_reward += env_reward
        self.total_shaped_reward += shaped_reward

        if self.previous_state is not None and self.previous_action is not None:
            self.update_q_value(
                self.previous_state,
                self.previous_action,
                shaped_reward,
                current_state,
                done,
            )

        if (not self.army_selected) and (_SELECT_ARMY in obs.observation.available_actions):
            self.army_selected = True
            self.previous_state = current_state
            self.previous_action = None
            self.prev_distance = current_distance
            return actions.FUNCTIONS.select_army(_SELECT_ALL)

        if _MOVE_SCREEN in obs.observation.available_actions:
            action_idx = self.choose_action(current_state)
            if self.action_mode == "delta":
                dx, dy = self.actions[action_idx]
                target_x = int(np.clip(marine_x + dx, 0, self.screen_size - 1))
                target_y = int(np.clip(marine_y + dy, 0, self.screen_size - 1))
            else:
                target_x, target_y = self.actions[action_idx]

            self.previous_state = current_state
            self.previous_action = action_idx
            self.prev_distance = current_distance

            return actions.FUNCTIONS.Move_screen(_NOT_QUEUED, [target_x, target_y])

        return actions.FUNCTIONS.no_op()

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save_q_table(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load_q_table(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            loaded_table = pickle.load(f)
            self.q_table = defaultdict(float, loaded_table)
        return True


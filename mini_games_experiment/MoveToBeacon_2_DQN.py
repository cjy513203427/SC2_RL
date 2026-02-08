"""
DQN Training Script for StarCraft II MoveToBeacon Mini-game.
"""

import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import numpy as np
import pickle
import torch
from absl import app, flags
from pysc2.env import sc2_env
from pysc2.lib import features, actions

# Import DQNAgent
from dqn_agent import DQNAgent

FLAGS = flags.FLAGS
flags.DEFINE_string("map", "MoveToBeacon", "Name of the SC2 map to use")
flags.DEFINE_integer("episodes", 1000, "Number of episodes to train")
flags.DEFINE_bool("render", False, "Whether to render the game")
flags.DEFINE_integer("screen_size", 64, "Screen resolution")
flags.DEFINE_integer("minimap_size", 64, "Minimap resolution") 
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step")
flags.DEFINE_integer("max_steps", 250, "Maximum steps per episode")

# DQN Hyperparameters
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate")
flags.DEFINE_integer("buffer_size", 100000, "Replay buffer size")
flags.DEFINE_integer("batch_size", 64, "Mini-batch size")
flags.DEFINE_float("gamma", 0.99, "Discount factor")
flags.DEFINE_float("tau", 1e-3, "Soft update parameter")
flags.DEFINE_integer("update_every", 4, "Update frequency")
flags.DEFINE_integer("hidden_size", 64, "Hidden layer size")
flags.DEFINE_float("epsilon_start", 1.0, "Initial epsilon")
flags.DEFINE_float("epsilon_end", 0.01, "Final epsilon")
flags.DEFINE_float("epsilon_decay", 0.995, "Epsilon decay per episode")

flags.DEFINE_string("save_path", "models/dqn_movetobeacon.pth", "Path to save model")
flags.DEFINE_bool("load_model", False, "Whether to load existing model")

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ALL = [0]
_NOT_QUEUED = [0]

def get_positions(obs, screen_size):
    """Extracts marine and beacon positions."""
    player_relative = obs.observation.feature_screen.player_relative
    marine_y, marine_x = (player_relative == 1).nonzero()
    beacon_y, beacon_x = (player_relative == 3).nonzero()
    
    if len(marine_x) > 0:
        marine_pos = (int(np.mean(marine_x)), int(np.mean(marine_y)))
    else:
        marine_pos = (screen_size // 2, screen_size // 2) # Default center
        
    if len(beacon_x) > 0:
        beacon_pos = (int(np.mean(beacon_x)), int(np.mean(beacon_y)))
    else:
        beacon_pos = (screen_size // 2, screen_size // 2)

    return marine_pos, beacon_pos

def get_state(obs, screen_size):
    """Returns normalized state vector: [marine_x, marine_y, beacon_x, beacon_y]"""
    marine_pos, beacon_pos = get_positions(obs, screen_size)
    
    # Normalize to [0, 1]
    state = np.array([
        marine_pos[0] / screen_size,
        marine_pos[1] / screen_size,
        beacon_pos[0] / screen_size,
        beacon_pos[1] / screen_size
    ], dtype=np.float32)
    return state

def main(argv):
    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize Agent
    # State size: 4 (mx, my, bx, by)
    # Action size: 9 (deltas)
    agent = DQNAgent(state_size=4, action_size=9, seed=0,
                     learning_rate=FLAGS.learning_rate,
                     buffer_size=FLAGS.buffer_size,
                     batch_size=FLAGS.batch_size,
                     gamma=FLAGS.gamma,
                     tau=FLAGS.tau,
                     update_every=FLAGS.update_every,
                     device=device)

    if FLAGS.load_model and os.path.exists(FLAGS.save_path):
        agent.load(FLAGS.save_path)
        print(f"Loaded model from {FLAGS.save_path}")

    # Initialize epsilon
    agent.epsilon = FLAGS.epsilon_start

    # Tracking
    scores = []
    scores_window = []
    
    # Create Environment
    with sc2_env.SC2Env(
            map_name=FLAGS.map,
            players=[sc2_env.Agent(sc2_env.Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=FLAGS.screen_size, minimap=FLAGS.minimap_size),
                use_feature_units=True),
            step_mul=FLAGS.step_mul,
            game_steps_per_episode=0, # Use max_steps instead
            visualize=FLAGS.render) as env:

        for i_episode in range(1, FLAGS.episodes + 1):
            obs = env.reset()
            obs = obs[0]
            
            # Reset state
            state = get_state(obs, FLAGS.screen_size)
            score = 0
            
            # Calculate initial distance for reward shaping
            marine_pos, beacon_pos = get_positions(obs, FLAGS.screen_size)
            prev_dist = np.linalg.norm(np.array(marine_pos) - np.array(beacon_pos))

            
            # Select army first
            if _SELECT_ARMY in obs.observation.available_actions:
                obs = env.step([actions.FUNCTIONS.select_army(_SELECT_ALL)])[0]
            
            step_count = 0
            episode_debug_info = []  # Store debug info for this episode
            
            while step_count < FLAGS.max_steps:
                # Agent action
                action_idx = agent.act(state, agent.epsilon)
                
                # Convert action index to PySC2 action
                # The agent uses delta moves. We act based on current marine position.
                # IMPORTANT: Use the current state to get marine position
                marine_x, marine_y = int(state[0] * FLAGS.screen_size), int(state[1] * FLAGS.screen_size)
                beacon_x, beacon_y = int(state[2] * FLAGS.screen_size), int(state[3] * FLAGS.screen_size)
                marine_pos = (marine_x, marine_y)
                beacon_pos = (beacon_x, beacon_y)
                
                # action map from dqn_agent
                # (-1, -1), (0, -1), ...
                dx, dy = agent.actions_list[action_idx]
                
                # Scale delta (optional, but good to have some step size)
                # Let's say step size is ~10% of screen or fixed pixel amount
                step_size = 12 # Same as Q-Learning default
                
                target_x = int(np.clip(marine_pos[0] + dx * step_size, 0, FLAGS.screen_size - 1))
                target_y = int(np.clip(marine_pos[1] + dy * step_size, 0, FLAGS.screen_size - 1))
                
                if _MOVE_SCREEN in obs.observation.available_actions:
                    sc2_action = actions.FUNCTIONS.Move_screen(_NOT_QUEUED, [target_x, target_y])
                else:
                    sc2_action = actions.FUNCTIONS.no_op()

                # Step Env
                next_obs = env.step([sc2_action])[0]
                obs = next_obs  # Update obs for next iteration
                next_state = get_state(next_obs, FLAGS.screen_size)
                raw_reward = float(next_obs.reward)
                done = next_obs.last()

                # Reward Shaping (Distance based)
                marine_pos_next, beacon_pos_next = get_positions(next_obs, FLAGS.screen_size)
                curr_dist = np.linalg.norm(np.array(marine_pos_next) - np.array(beacon_pos_next))
                
                dist_delta = prev_dist - curr_dist
                shaped_reward_bonus = 0.0
                
                # If we got a reward (collected beacon), the beacon moved, so don't penalize for distance jump
                if raw_reward > 0:
                    reward = raw_reward
                else:
                    # Give small reward for moving closer, penalty for moving away
                    # Scale factor 0.01 ensures it doesn't overpower the main reward
                    shaped_reward_bonus = dist_delta * 0.01
                    reward = raw_reward + shaped_reward_bonus
                
                # Debug logging every 10 steps (only for episodes divisible by 10)
                if i_episode % 10 == 0 and step_count % 10 == 0:
                    episode_debug_info.append({
                        'step': step_count,
                        'marine': marine_pos,
                        'beacon': beacon_pos,
                        'action': (dx, dy),
                        'prev_dist': prev_dist,
                        'curr_dist': curr_dist,
                        'dist_delta': dist_delta,
                        'raw_reward': raw_reward,
                        'shaped_bonus': shaped_reward_bonus,
                        'total_reward': reward
                    })
                
                prev_dist = curr_dist

                
                agent.step(state, action_idx, reward, next_state, done)
                
                state = next_state
                score += reward
                step_count += 1
                
                if done:
                    break
            
            # Epsilon decay
            agent.epsilon = max(FLAGS.epsilon_end, agent.epsilon * FLAGS.epsilon_decay)
            
            scores.append(score)
            scores_window.append(score)
            if len(scores_window) > 100:
                scores_window.pop(0) # Keep last 100
            
            if i_episode % 10 == 0:
                print(f"Episode {i_episode}\tScore: {score}\tAvg Score: {np.mean(scores_window):.2f}\tEpsilon: {agent.epsilon:.2f}")
                
                # Print debug info for first 3 steps
                if episode_debug_info:
                    print("  [DEBUG] First 3 steps:")
                    for info in episode_debug_info[:3]:
                        print(f"    Step {info['step']:3d} | Marine:{info['marine']} Beacon:{info['beacon']} | "
                              f"Action:{info['action']} | Dist: {info['prev_dist']:.1f}->{info['curr_dist']:.1f} "
                              f"(Δ={info['dist_delta']:+.1f}) | Raw_R:{info['raw_reward']:.1f} Bonus:{info['shaped_bonus']:+.4f} Total:{info['total_reward']:+.4f}")

            
            if i_episode % 100 == 0:
                agent.save(FLAGS.save_path)
                
    # Final save
    if not os.path.exists(os.path.dirname(FLAGS.save_path)):
        os.makedirs(os.path.dirname(FLAGS.save_path))
    agent.save(FLAGS.save_path)
    print("Training finished.")

if __name__ == '__main__':
    app.run(main)

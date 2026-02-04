"""
Q-Learning training script for StarCraft II MoveToBeacon Mini-game.

The reusable agent implementation lives in `qlearning_agent.py` to avoid absl.flags
DuplicateFlagError when imported from multiple scripts (train / test / evaluate).
"""

import numpy as np
import pickle
from absl import app, flags
from pysc2.env import sc2_env
from pysc2.lib import features

from qlearning_agent import QLearningAgent

FLAGS = flags.FLAGS
flags.DEFINE_string("map", "MoveToBeacon", "Name of the SC2 map to use")
flags.DEFINE_integer("episodes", 500, "Number of episodes to train")
flags.DEFINE_bool("render", False, "Whether to render the game")
flags.DEFINE_integer("screen_size", 84, "Screen resolution")
flags.DEFINE_integer("minimap_size", 64, "Minimap resolution")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step")
flags.DEFINE_integer("max_steps", 200, "Maximum steps per episode")
flags.DEFINE_float("learning_rate", 0.1, "Learning rate (alpha)")
flags.DEFINE_float("discount_factor", 0.95, "Discount factor (gamma)")
flags.DEFINE_float("epsilon_start", 1.0, "Initial exploration rate")
flags.DEFINE_float("epsilon_end", 0.01, "Final exploration rate")
flags.DEFINE_float("epsilon_decay", 0.99, "Epsilon decay rate per episode")
flags.DEFINE_integer("grid_size", 8, "Grid size for state discretization (used by state discretization / action step)")
flags.DEFINE_bool("use_relative_state", True, "Use relative state (beacon - marine) instead of absolute positions")
flags.DEFINE_enum("action_mode", "delta", ["delta", "grid_absolute"], "Action mode: delta=relative move, grid_absolute=move to fixed grid points")
flags.DEFINE_integer("action_step_pixels", 12, "Step size in pixels for delta actions (clamped to screen)")
flags.DEFINE_bool("reward_shaping", True, "Enable distance-based reward shaping")
flags.DEFINE_float("distance_reward_scale", 0.02, "Reward shaping scale for delta distance: r += scale*(prev_dist - curr_dist)")
flags.DEFINE_string("save_path", "models/qlearning_movetobeacon.pkl", "Path to save Q-table")
flags.DEFINE_bool("load_model", False, "Whether to load existing Q-table")

def main(argv):
    """Main training loop"""
    
    print("=" * 80)
    print(f"Q-Learning Agent Training on {FLAGS.map}")
    print("=" * 80)
    print(f"Episodes: {FLAGS.episodes}")
    print(f"Max steps per episode: {FLAGS.max_steps}")
    print(f"Learning rate (α): {FLAGS.learning_rate}")
    print(f"Discount factor (γ): {FLAGS.discount_factor}")
    print(f"Epsilon: {FLAGS.epsilon_start} -> {FLAGS.epsilon_end} (decay: {FLAGS.epsilon_decay})")
    print(f"Grid size: {FLAGS.grid_size}x{FLAGS.grid_size}")
    print(f"Render: {FLAGS.render}")
    print("=" * 80)
    
    # Create agent
    agent = QLearningAgent(
        learning_rate=FLAGS.learning_rate,
        discount_factor=FLAGS.discount_factor,
        epsilon_start=FLAGS.epsilon_start,
        epsilon_end=FLAGS.epsilon_end,
        epsilon_decay=FLAGS.epsilon_decay,
        grid_size=FLAGS.grid_size,
        screen_size=FLAGS.screen_size,
        use_relative_state=FLAGS.use_relative_state,
        action_mode=FLAGS.action_mode,
        action_step_pixels=FLAGS.action_step_pixels,
        reward_shaping=FLAGS.reward_shaping,
        distance_reward_scale=FLAGS.distance_reward_scale
    )
    
    # Load existing Q-table if requested
    if FLAGS.load_model:
        agent.load_q_table(FLAGS.save_path)
    
    # Statistics tracking
    episode_rewards = []          # env reward
    episode_rewards_shaped = []   # shaped reward
    episode_steps = []
    avg_rewards_window = []
    avg_rewards_shaped_window = []
    
    # Create environment
    with sc2_env.SC2Env(
        map_name=FLAGS.map,
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        agent_interface_format=features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(
                screen=FLAGS.screen_size,
                minimap=FLAGS.minimap_size
            ),
            use_feature_units=True
        ),
        step_mul=FLAGS.step_mul,
        game_steps_per_episode=0,
        visualize=FLAGS.render
    ) as env:
        
        # Training loop
        for episode in range(FLAGS.episodes):
            agent.reset()
            
            # Reset environment
            obs = env.reset()
            obs_spec = obs[0]
            
            # Run episode
            step_count = 0
            while step_count < FLAGS.max_steps:
                # Agent takes action
                action = agent.step(obs_spec)
                
                # Step environment
                obs = env.step([action])
                obs_spec = obs[0]
                
                step_count += 1
                
                # Check if episode is done
                if obs_spec.last():
                    # Final update for terminal state
                    current_state = agent._get_state(obs_spec)
                    if agent.previous_state is not None and agent.previous_action is not None:
                        # Reward shaping for terminal transition uses the last stored prev_distance and current terminal distance
                        marine_x, marine_y, beacon_x, beacon_y, marine_found, beacon_found = agent._get_positions(obs_spec)
                        current_distance = None
                        if marine_found and beacon_found:
                            current_distance = float(np.hypot(marine_x - beacon_x, marine_y - beacon_y))

                        shaped_reward = float(obs_spec.reward)
                        if agent.reward_shaping and (agent.prev_distance is not None) and (current_distance is not None):
                            shaped_reward += agent.distance_reward_scale * (agent.prev_distance - current_distance)

                        agent.update_q_value(
                            agent.previous_state,
                            agent.previous_action,
                            shaped_reward,
                            current_state,
                            True
                        )
                    break
            
            # Decay epsilon
            agent.decay_epsilon()
            
            # Record statistics
            episode_rewards.append(agent.total_env_reward)
            episode_rewards_shaped.append(agent.total_shaped_reward)
            episode_steps.append(agent.steps)
            
            # Calculate moving average (last 20 episodes)
            window_size = min(20, len(episode_rewards))
            avg_reward = np.mean(episode_rewards[-window_size:])
            avg_rewards_window.append(avg_reward)
            avg_reward_shaped = np.mean(episode_rewards_shaped[-window_size:])
            avg_rewards_shaped_window.append(avg_reward_shaped)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1:4d}/{FLAGS.episodes} | "
                      f"EnvReward: {agent.total_env_reward:6.1f} | "
                      f"Shaped: {agent.total_shaped_reward:7.2f} | "
                      f"Avg(20): {avg_reward:6.1f} | "
                      f"Steps: {agent.steps:4d} | "
                      f"Epsilon: {agent.epsilon:.4f} | "
                      f"Q-table size: {len(agent.q_table)}")
            
            # Save Q-table periodically
            if (episode + 1) % 100 == 0:
                agent.save_q_table(FLAGS.save_path)
    
    # Final save
    agent.save_q_table(FLAGS.save_path)
    
    # Print final statistics
    print("\n" + "=" * 80)
    print("Training Completed!")
    print("=" * 80)
    print(f"Total episodes: {FLAGS.episodes}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Q-table size: {len(agent.q_table)} state-action pairs")
    print(f"Average reward (last 20 episodes): {np.mean(episode_rewards[-20:]):.2f}")
    print(f"Average shaped reward (last 20 episodes): {np.mean(episode_rewards_shaped[-20:]):.2f}")
    print(f"Average steps (last 20 episodes): {np.mean(episode_steps[-20:]):.2f}")
    print(f"Best episode reward: {max(episode_rewards):.2f}")
    print("=" * 80)
    
    # Save training statistics
    stats_path = FLAGS.save_path.replace('.pkl', '_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump({
            'episode_rewards': episode_rewards,
            'episode_rewards_shaped': episode_rewards_shaped,
            'episode_steps': episode_steps,
            'avg_rewards': avg_rewards_window,
            'avg_rewards_shaped': avg_rewards_shaped_window
        }, f)
    print(f"Training statistics saved to {stats_path}")


if __name__ == "__main__":
    app.run(main)

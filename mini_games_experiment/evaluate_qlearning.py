"""
Evaluation script for trained Q-Learning MoveToBeacon agent
Loads a trained Q-table and evaluates performance with visualization
"""

from absl import app, flags
from qlearning_agent import QLearningAgent
from pysc2.env import sc2_env
from pysc2.lib import actions, features
import numpy as np
import pickle
import os

FLAGS = flags.FLAGS
flags.DEFINE_string("model_path", "models/qlearning_movetobeacon.pkl", "Path to saved Q-table")
flags.DEFINE_integer("episodes", 10, "Number of episodes to evaluate")
flags.DEFINE_bool("render", True, "Whether to render the game")
flags.DEFINE_integer("screen_size", 84, "Screen resolution")
flags.DEFINE_integer("minimap_size", 64, "Minimap resolution")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step")
flags.DEFINE_integer("max_steps", 200, "Maximum steps per episode")
flags.DEFINE_integer("grid_size", 8, "Grid size (must match training)")
flags.DEFINE_float("epsilon", 0.0, "Exploration rate (0.0 for pure exploitation)")


def main(argv):
    """Main evaluation loop"""
    
    print("=" * 80)
    print("Q-Learning MoveToBeacon - Evaluation Mode")
    print("=" * 80)
    print(f"Model path: {FLAGS.model_path}")
    print(f"Episodes: {FLAGS.episodes}")
    print(f"Epsilon: {FLAGS.epsilon} (0.0 = greedy, >0 = exploration)")
    print(f"Render: {FLAGS.render}")
    print("=" * 80 + "\n")
    
    # Check if model exists
    if not os.path.exists(FLAGS.model_path):
        print(f"[FAIL] Error: Model file not found at {FLAGS.model_path}")
        print("\nPlease train a model first:")
        print("  python mini_games_experiment/MoveToBeacon_1_Qlearning.py --episodes=500 --norender")
        return 1
    
    # Create agent
    agent = QLearningAgent(
        learning_rate=0.0,  # No learning during evaluation
        discount_factor=0.95,
        epsilon_start=FLAGS.epsilon,
        epsilon_end=FLAGS.epsilon,
        epsilon_decay=1.0,  # No decay
        grid_size=FLAGS.grid_size,
        screen_size=FLAGS.screen_size,
        # Evaluation focuses on true env reward; shaping is unnecessary.
        reward_shaping=False
    )
    
    # Load Q-table
    agent.load_q_table(FLAGS.model_path)
    
    if len(agent.q_table) == 0:
        print("[FAIL] Error: Loaded Q-table is empty!")
        return 1
    
    print(f"[OK] Loaded Q-table with {len(agent.q_table)} state-action pairs\n")
    
    # Try to load training statistics
    stats_path = FLAGS.model_path.replace('.pkl', '_stats.pkl')
    if os.path.exists(stats_path):
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        print(f"Training history:")
        print(f"  Total episodes trained: {len(stats['episode_rewards'])}")
        print(f"  Final avg reward: {stats['avg_rewards'][-1]:.2f}")
        print(f"  Best episode reward: {max(stats['episode_rewards']):.2f}")
        print()
    
    # Statistics tracking
    episode_rewards = []
    episode_rewards_shaped = []
    episode_steps = []
    episode_scores = []
    
    # Create environment
    with sc2_env.SC2Env(
        map_name="MoveToBeacon",
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
        
        print("Starting evaluation...")
        print("-" * 80)
        
        # Evaluation loop
        for episode in range(FLAGS.episodes):
            agent.reset()
            
            # Reset environment
            obs = env.reset()
            obs_spec = obs[0]
            
            # Run episode
            step_count = 0
            while step_count < FLAGS.max_steps:
                # Agent takes action (greedy if epsilon=0)
                action = agent.step(obs_spec)
                
                # Step environment
                obs = env.step([action])
                obs_spec = obs[0]
                
                step_count += 1
                
                # Check if episode is done
                if obs_spec.last():
                    break
            
            # Record statistics
            episode_rewards.append(agent.total_env_reward)
            episode_rewards_shaped.append(agent.total_shaped_reward)
            episode_steps.append(agent.steps)
            episode_scores.append(obs_spec.observation.score_cumulative[0])
            
            # Print episode results
            print(f"Episode {episode + 1:2d}/{FLAGS.episodes} | "
                  f"EnvReward: {agent.total_env_reward:6.1f} | "
                  f"Shaped: {agent.total_shaped_reward:7.2f} | "
                  f"Steps: {agent.steps:4d} | "
                  f"Score: {obs_spec.observation.score_cumulative[0]:6.1f}")
    
    # Print summary statistics
    print("-" * 80)
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print(f"Episodes evaluated: {FLAGS.episodes}")
    print()
    print("Rewards:")
    print(f"  Average: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"  Min:     {np.min(episode_rewards):.2f}")
    print(f"  Max:     {np.max(episode_rewards):.2f}")
    print()
    print("Shaped Rewards (for debugging):")
    print(f"  Average: {np.mean(episode_rewards_shaped):.2f} +/- {np.std(episode_rewards_shaped):.2f}")
    print()
    print("Steps:")
    print(f"  Average: {np.mean(episode_steps):.2f} +/- {np.std(episode_steps):.2f}")
    print(f"  Min:     {np.min(episode_steps)}")
    print(f"  Max:     {np.max(episode_steps)}")
    print()
    print("Scores:")
    print(f"  Average: {np.mean(episode_scores):.2f} +/- {np.std(episode_scores):.2f}")
    print(f"  Min:     {np.min(episode_scores):.2f}")
    print(f"  Max:     {np.max(episode_scores):.2f}")
    print("=" * 80)
    
    # Performance assessment
    avg_reward = np.mean(episode_rewards)
    print("\nPerformance Assessment:")
    if avg_reward >= 25:
        print("[OK] Excellent! Agent is performing very well.")
    elif avg_reward >= 20:
        print("[OK] Good! Agent has learned an effective policy.")
    elif avg_reward >= 15:
        print("[WARN] Fair. Agent is learning but could improve with more training.")
    else:
        print("[FAIL] Poor. Agent needs more training or hyperparameter tuning.")
    
    print("\nTo improve performance:")
    print("  1. Train for more episodes (e.g., --episodes=1000)")
    print("  2. Adjust learning rate (e.g., --learning_rate=0.15)")
    print("  3. Tune epsilon decay (e.g., --epsilon_decay=0.997)")
    print("  4. Try larger grid size (e.g., --grid_size=12)")
    print()


if __name__ == "__main__":
    app.run(main)

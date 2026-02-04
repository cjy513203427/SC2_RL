"""
Quick test script for Q-Learning MoveToBeacon implementation
Runs a short training session to verify everything works correctly
"""

from absl import app
from qlearning_agent import QLearningAgent
from pysc2.env import sc2_env
from pysc2.lib import actions, features
import numpy as np


def test_agent_components():
    """Test agent initialization and core methods"""
    print("Testing agent components...")
    
    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        grid_size=8,
        screen_size=84,
        use_relative_state=True,
        action_mode="delta",
        action_step_pixels=12,
        reward_shaping=True,
        distance_reward_scale=0.02
    )
    
    # Test action space creation
    assert len(agent.actions) == 9, f"Expected 9 actions for delta mode, got {len(agent.actions)}"
    print(f"[OK] Action space created (delta mode): {agent.num_actions} actions")
    
    # Test epsilon decay
    initial_epsilon = agent.epsilon
    agent.decay_epsilon()
    assert agent.epsilon < initial_epsilon, "Epsilon should decrease"
    print(f"[OK] Epsilon decay works: {initial_epsilon:.4f} -> {agent.epsilon:.4f}")
    
    # Test Q-value methods
    test_state = (3, 4, 5, 6)
    test_action = 3
    agent.q_table[(test_state, test_action)] = 0.5
    
    max_q = agent._get_max_q_value(test_state)
    assert max_q == 0.5, f"Expected max Q=0.5, got {max_q}"
    print("[OK] Q-value retrieval works")
    
    # Test action selection
    action = agent.choose_action(test_state)
    assert 0 <= action < agent.num_actions, f"Invalid action: {action}"
    print("[OK] Action selection works")
    
    # Test Q-value update
    agent.update_q_value(test_state, test_action, 1.0, (4, 5, 5, 6), False)
    new_q = agent.q_table[(test_state, test_action)]
    assert new_q != 0.5, "Q-value should be updated"
    print(f"[OK] Q-value update works: {0.5:.4f} -> {new_q:.4f}")
    
    print("\n[OK] All agent component tests passed!\n")


def test_short_training():
    """Run a short training session to verify environment interaction"""
    print("Running short training test (5 episodes, no rendering)...")
    print("-" * 60)
    
    # Create agent
    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.9,
        grid_size=8,
        screen_size=84,
        use_relative_state=True,
        action_mode="delta",
        action_step_pixels=12,
        reward_shaping=True,
        distance_reward_scale=0.02
    )
    
    # Create environment
    with sc2_env.SC2Env(
        map_name="MoveToBeacon",
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        agent_interface_format=features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(
                screen=84,
                minimap=64
            ),
            use_feature_units=True
        ),
        step_mul=8,
        game_steps_per_episode=0,
        visualize=False
    ) as env:
        
        episode_rewards = []
        
        for episode in range(5):
            agent.reset()
            obs = env.reset()
            obs_spec = obs[0]
            
            step_count = 0
            max_steps = 100
            
            while step_count < max_steps:
                action = agent.step(obs_spec)
                obs = env.step([action])
                obs_spec = obs[0]
                step_count += 1
                
                if obs_spec.last():
                    # Final update
                    current_state = agent._get_state(obs_spec)
                    if agent.previous_state is not None and agent.previous_action is not None:
                        agent.update_q_value(
                            agent.previous_state,
                            agent.previous_action,
                            obs_spec.reward,
                            current_state,
                            True
                        )
                    break
            
            agent.decay_epsilon()
            episode_rewards.append(agent.total_env_reward)
            
            print(f"Episode {episode + 1}/5: "
                  f"EnvReward={agent.total_env_reward:6.1f}, "
                  f"Shaped={agent.total_shaped_reward:7.2f}, "
                  f"Steps={agent.steps:3d}, "
                  f"Epsilon={agent.epsilon:.4f}, "
                  f"Q-table size={len(agent.q_table)}")
    
    print("-" * 60)
    avg_reward = np.mean(episode_rewards)
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Final Q-table size: {len(agent.q_table)} state-action pairs")
    
    # Basic sanity checks
    assert len(agent.q_table) > 0, "Q-table should have entries after training"
    # Note: MoveToBeacon's native reward can be sparse; with reward shaping enabled we should
    # still see some positive shaped return across short runs.
    assert agent.total_shaped_reward != 0, "Shaped reward should be non-zero with reward shaping enabled"
    
    print("\n[OK] Short training test passed!\n")


def main(argv):
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Q-Learning MoveToBeacon - Test Suite")
    print("=" * 60 + "\n")
    
    try:
        # Test 1: Agent components
        test_agent_components()
        
        # Test 2: Short training
        test_short_training()
        
        print("=" * 60)
        print("[OK] All tests passed successfully!")
        print("=" * 60)
        print("\nYou can now run full training with:")
        print("  python mini_games_experiment/MoveToBeacon_1_Qlearning.py --episodes=500 --norender")
        print()
        
    except Exception as e:
        print("\n[FAIL] Test failed!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    app.run(main)

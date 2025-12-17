"""
Simple Random Agent for StarCraft II
This script demonstrates a basic agent that takes random actions in the SC2 environment
"""

import numpy as np
from absl import app, flags
from pysc2.env import sc2_env
from pysc2.lib import actions, features

FLAGS = flags.FLAGS
flags.DEFINE_string("map", "MoveToBeacon", "Name of the SC2 map to use")
flags.DEFINE_integer("episodes", 200, "Number of episodes to run")
flags.DEFINE_bool("render", True, "Whether to render the game")
flags.DEFINE_integer("screen_size", 84, "Screen resolution")
flags.DEFINE_integer("minimap_size", 64, "Minimap resolution")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step")
flags.DEFINE_integer("max_steps", 1000, "Maximum steps per episode")


class SimpleAgent:
    """A simple random agent"""
    
    def __init__(self):
        self.total_reward = 0
        self.episodes = 0
        self.steps = 0
    
    def reset(self):
        """Reset agent state for a new episode"""
        self.total_reward = 0
        self.steps = 0
    
    def step(self, obs):
        """Take a step in the environment
        
        Args:
            obs: Observation from the environment
            
        Returns:
            action: Action to take
        """
        self.steps += 1
        
        # Track reward
        self.total_reward += obs.reward
        
        # Get available actions
        available_actions = obs.observation.available_actions
        
        # For simplicity, use no-op most of the time, and occasionally use select_army + move
        if actions.FUNCTIONS.no_op.id in available_actions:
            action_id = actions.FUNCTIONS.no_op.id
            return actions.FUNCTIONS.no_op()
        
        # If no-op not available, choose the first available action
        action_id = available_actions[0]
        
        # Get action arguments
        args = []
        for arg_type in actions.FUNCTIONS[action_id].args:
            if arg_type.name in ['screen', 'minimap', 'screen2']:
                # Random position
                if 'screen' in arg_type.name:
                    h, w = obs.observation.feature_screen.height_map.shape
                else:
                    h, w = obs.observation.feature_minimap.height_map.shape
                args.append([np.random.randint(0, w), np.random.randint(0, h)])
            elif arg_type.name == 'queued':
                args.append([0])  # Don't queue actions
            elif arg_type.name in ['control_group_act', 'control_group_id', 
                                   'select_point_act', 'select_add', 'select_worker']:
                args.append([0])  # Use first enum value
            elif arg_type.name == 'select_unit_act':
                args.append([0])
            elif arg_type.name == 'build_queue_id':
                args.append([0])
            elif arg_type.name == 'unload_id':
                args.append([0])
            else:
                # For unknown argument types, use 0
                args.append([0])
        
        return actions.FUNCTIONS[action_id](*args)


def main(argv):
    """Main entry point"""
    
    print("=" * 60)
    print(f"Running Simple Agent on {FLAGS.map}")
    print("=" * 60)
    print(f"Episodes: {FLAGS.episodes}")
    print(f"Max steps per episode: {FLAGS.max_steps}")
    print(f"Render: {FLAGS.render}")
    print("=" * 60)
    
    # Create agent
    agent = SimpleAgent()
    
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
        
        # Run episodes
        for episode in range(FLAGS.episodes):
            print(f"\n--- Episode {episode + 1} ---")
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
                    break
            
            # Print episode statistics
            print(f"Episode {episode + 1} finished")
            print(f"  Steps: {agent.steps}")
            print(f"  Total Reward: {agent.total_reward}")
            print(f"  Final Score: {obs_spec.observation.score_cumulative[0]}")
    
    print("\n" + "=" * 60)
    print("Simple Agent Run Completed!")
    print("=" * 60)


if __name__ == "__main__":
    app.run(main)


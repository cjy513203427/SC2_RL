"""
Test script for PySC2 environment
This script tests the basic SC2 RL environment setup
"""

import os
import sys
from pysc2.env import sc2_env
from pysc2.lib import features, actions
from absl import app, flags

# Flags for PySC2
FLAGS = flags.FLAGS
flags.DEFINE_bool("render", True, "Whether to render the game")
flags.DEFINE_integer("screen_size", 84, "Screen resolution")
flags.DEFINE_integer("minimap_size", 64, "Minimap resolution")

def test_environment(argv):
    """Test basic PySC2 environment creation and interaction"""
    
    print("=" * 60)
    print("Testing PySC2 Environment Setup")
    print("=" * 60)
    
    # Check SC2PATH
    sc2_path = os.environ.get('SC2PATH')
    if sc2_path:
        print(f"[OK] SC2PATH is set: {sc2_path}")
    else:
        print("[WARNING] SC2PATH not set, PySC2 will use default path")
    
    try:
        # Create a simple SC2 environment
        print("\nCreating SC2 Environment...")
        print(f"  Map: MoveToBeacon")
        print(f"  Screen size: {FLAGS.screen_size}")
        print(f"  Minimap size: {FLAGS.minimap_size}")
        
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
            step_mul=8,
            game_steps_per_episode=0,
            visualize=FLAGS.render
        ) as env:
            
            print("[OK] Environment created successfully!")
            
            # Reset environment
            print("\nResetting environment...")
            obs = env.reset()
            print(f"[OK] Environment reset successful!")
            print(f"  Number of observations: {len(obs)}")
            
            # Get available actions
            obs_spec = obs[0]
            available_actions = obs_spec.observation.available_actions
            print(f"  Available actions: {len(available_actions)}")
            
            # Take a few random steps
            print("\nTaking 5 test steps...")
            for step in range(5):
                # Use no-op action
                action = [actions.FUNCTIONS.no_op()]
                obs = env.step(action)
                print(f"  Step {step + 1} completed")
            
            print("\n" + "=" * 60)
            print("[SUCCESS] All tests passed! PySC2 environment is working correctly!")
            print("=" * 60)
            
            return True
            
    except Exception as e:
        print(f"\n[ERROR] Error during environment test:")
        print(f"  {type(e).__name__}: {str(e)}")
        print("\nPossible solutions:")
        print("  1. Make sure StarCraft II is installed")
        print("  2. Run setup_sc2_env.py to configure SC2PATH")
        print("  3. Check that you have the required maps")
        return False

def main(argv):
    """Main entry point"""
    try:
        success = test_environment(argv)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(0)

if __name__ == "__main__":
    app.run(main)


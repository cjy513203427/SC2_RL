# StarCraft II Reinforcement Learning Environment

This project sets up a reinforcement learning environment for StarCraft II using PySC2.

## Prerequisites

- StarCraft II installed on Windows (default installation path)
- Conda/Miniconda installed

## Setup Instructions

### 1. Create and Activate Conda Environment

```bash
conda create -n sc2_rl python=3.10 -y
conda activate sc2_rl
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install PySC2 directly:

```bash
pip install pysc2
```

### 3. Configure SC2 Path

Run the setup script to configure the StarCraft II installation path:

```bash
python setup_sc2_env.py
```

This script will:
- Detect your SC2 installation path
- Set the `SC2PATH` environment variable
- Verify the installation

### 4. Test the Environment

Run the test script to verify everything is working:

```bash
python test_sc2_env.py
```

Options:
- `--render`: Enable/disable game rendering (default: True)
- `--screen_size`: Screen resolution (default: 84)
- `--minimap_size`: Minimap resolution (default: 64)

Example:
```bash
python test_sc2_env.py --render --screen_size=84 --minimap_size=64
```

## Project Structure

```
SC2_RL/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup_sc2_env.py         # SC2 path configuration script
├── test_sc2_env.py          # Environment test script
└── story initialization.md   # Project initialization notes
```

## Next Steps

After successful setup, you can:

1. **Explore PySC2 APIs**: Learn about observation space, action space
2. **Implement RL Agents**: Use algorithms like DQN, A3C, PPO
3. **Train on Different Maps**: Test various SC2 mini-games and maps
4. **Integrate RL Frameworks**: Add PyTorch/TensorFlow with Stable-Baselines3

## Common Issues

### SC2 Not Found
If the setup script can't find SC2:
- Verify SC2 is installed at `C:\Program Files (x86)\StarCraft II` or similar
- Manually set environment variable: `set SC2PATH=C:\path\to\StarCraft II`

### Map Not Found
If you get map errors:
- Download required maps from the [SC2 Map Pack](https://github.com/Blizzard/s2client-proto#map-packs)
- Place maps in: `%SC2PATH%\Maps\`

### Version Mismatch
If you get version errors:
- Check your SC2 version matches PySC2 requirements
- Update SC2 through Battle.net launcher

## Resources

- [PySC2 GitHub](https://github.com/google-deepmind/pysc2)
- [PySC2 Documentation](https://github.com/google-deepmind/pysc2/blob/master/docs/environment.md)
- [SC2 API](https://github.com/Blizzard/s2client-proto)
- [SC2LE](https://deepmind.com/blog/article/deepmind-and-blizzard-open-starcraft-ii-ai-research-environment)


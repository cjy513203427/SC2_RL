# StarCraft II RL 环境快速启动指南

## 环境已就绪! ✓

基本的SC2强化学习环境已经配置完成并通过测试。

## 快速开始

### 1. 激活环境
```bash
conda activate sc2_rl
cd D:\Cursor_project\SC2_RL
```

### 2. 运行示例

#### 测试环境 (无渲染)
```bash
python test_sc2_env.py --norender
```

#### 测试环境 (有渲染)
```bash
python test_sc2_env.py
```

#### 运行简单Agent
```bash
# 运行5个episode，带可视化
python simple_agent.py --episodes=5

# 运行10个episode，不带可视化（更快）
python simple_agent.py --episodes=10 --norender --max_steps=200
```

## 可用的Mini-game地图

所有地图都已下载并可用:

1. **MoveToBeacon** - 学习移动到信标 (适合初学者)
2. **CollectMineralShards** - 收集矿物碎片
3. **CollectMineralsAndGas** - 收集资源
4. **DefeatRoaches** - 击败蟑螂
5. **DefeatZerglingsAndBanelings** - 击败小狗和毒爆
6. **FindAndDefeatZerglings** - 找到并击败小狗
7. **BuildMarines** - 建造机枪兵

### 更换地图示例
```bash
python simple_agent.py --map=CollectMineralShards --episodes=3
```

## 项目文件说明

| 文件 | 用途 |
|------|------|
| `setup_sc2_env.py` | 配置SC2安装路径 |
| `download_maps.py` | 下载mini-game地图 |
| `test_sc2_env.py` | 测试环境配置 |
| `simple_agent.py` | 简单随机agent示例 |
| `requirements.txt` | Python依赖列表 |
| `README.md` | 完整项目文档 |
| `环境配置说明.md` | 详细配置说明 |

## 常用命令

### 查看可用地图
```bash
python -m pysc2.bin.map_list
```

### 查看已安装的包
```bash
conda list
```

### 重新安装依赖
```bash
pip install -r requirements.txt
```

## 环境信息

- **Conda环境名**: sc2_rl
- **Python版本**: 3.10.19
- **PySC2版本**: 4.0.0
- **SC2路径**: C:\Program Files (x86)\StarCraft II
- **地图路径**: C:\Program Files (x86)\StarCraft II\Maps\mini_games\

## 下一步开发方向

### 1. 实现简单的学习算法
从简单的表格Q-learning或DQN开始:
- 定义状态空间
- 定义动作空间
- 实现Q表或神经网络
- 添加训练循环

### 2. 尝试不同的mini-games
每个mini-game都有不同的挑战:
- **MoveToBeacon**: 最简单，适合测试基础框架
- **CollectMineralShards**: 需要多单位控制
- **DefeatRoaches**: 需要战斗策略

### 3. 添加监控和日志
- 使用TensorBoard可视化训练过程
- 记录每个episode的得分
- 保存训练曲线

### 4. 优化训练
- 实现经验回放
- 尝试不同的神经网络架构
- 调整超参数

## 示例代码片段

### 获取观察信息
```python
from pysc2.env import sc2_env
from pysc2.lib import features

# 在环境中
obs = env.reset()[0]

# 获取特征层
feature_screen = obs.observation.feature_screen
feature_minimap = obs.observation.feature_minimap

# 获取游戏信息
score = obs.observation.score_cumulative
available_actions = obs.observation.available_actions
reward = obs.reward
```

### 执行动作
```python
from pysc2.lib import actions

# 执行no-op动作
action = actions.FUNCTIONS.no_op()
obs = env.step([action])

# 执行移动动作 (如果可用)
if actions.FUNCTIONS.Move_screen.id in obs.observation.available_actions:
    action = actions.FUNCTIONS.Move_screen("now", [x, y])
    obs = env.step([action])
```

## 需要帮助?

1. 查看 `README.md` 获取完整文档
2. 查看 `环境配置说明.md` 了解配置细节
3. 查看 [PySC2 GitHub](https://github.com/google-deepmind/pysc2) 官方文档

## 测试状态

所有基本功能已测试通过:
- ✓ 环境创建
- ✓ 环境重置
- ✓ 动作执行
- ✓ 观察获取
- ✓ Episode完成

**环境已就绪，可以开始开发RL算法了!** 🚀


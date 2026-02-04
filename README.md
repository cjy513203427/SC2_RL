# 星际争霸2强化学习（PySC2）实验项目

本项目基于 [PySC2](https://github.com/google-deepmind/pysc2) 在 StarCraft II 的 mini-games 上做强化学习实验，目前已完成 **MoveToBeacon 的表格型 Q-learning**（含训练与评估脚本）。

## 快速开始

```bash
conda activate sc2_rl
cd D:\Cursor_project\SC2_RL
python test_sc2_env.py --norender
```

训练 Q-learning（无渲染）：

```bash
python mini_games_experiment/MoveToBeacon_1_Qlearning.py --episodes=500 --norender
```

评估（只跑策略，不学习）：

```bash
python mini_games_experiment/evaluate_qlearning.py --episodes=50 --norender --model_path=models/qlearning_movetobeacon.pkl
```

## 文档（统一放在 docs/）

- **快速开始**：`docs/快速开始.md`
- **环境配置**：`docs/环境配置说明.md`
- **Q-learning 实验说明**：`docs/qlearning/Qlearning说明.md`
- **博客园文章草稿**：`docs/blog/重温星际2强化学习QLearning(一).md`
- **初始化记录**：`docs/初始化记录.md`
- **计划**：`docs/计划.md`

## 代码入口

- **训练**：`mini_games_experiment/MoveToBeacon_1_Qlearning.py`\n  - 会更新 Q 表并保存到 `models/`\n  - 训练阶段会使用探索（epsilon 衰减）与可选的奖励塑形（dense feedback）
- **评估**：`mini_games_experiment/evaluate_qlearning.py`\n  - 只加载 Q 表，不更新（`learning_rate=0`，默认 `epsilon=0` 贪婪策略）\n  - 用于客观测量 EnvReward
- **Agent 实现**：`mini_games_experiment/qlearning_agent.py`\n  - 存放 Q-learning 更新与状态/动作抽象（避免脚本间 flags 冲突）

## 项目结构（概览）

```
SC2_RL/
├── README.md
├── docs/
├── mini_games_experiment/
├── requirements.txt
├── setup_sc2_env.py
├── download_maps.py
├── test_sc2_env.py
└── simple_agent.py
```





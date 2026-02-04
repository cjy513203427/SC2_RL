# MoveToBeacon：Q-learning 实验说明

本文档对应代码：
- 训练脚本：`mini_games_experiment/MoveToBeacon_1_Qlearning.py`
- 评估脚本：`mini_games_experiment/evaluate_qlearning.py`
- Agent 实现：`mini_games_experiment/qlearning_agent.py`

## 1. 任务说明（MoveToBeacon）

- 目标：控制 marine 反复移动到信标（beacon）位置。
- 奖励（EnvReward）：每次碰到 beacon 获得 +1（其余时间为 0）。
- 训练输出里你会看到 EnvReward（真实环境奖励）与 Shaped（塑形奖励，训练辅助信号）。

## 2. Q-learning 核心公式

\[
Q(s,a) \leftarrow Q(s,a) + \alpha\Bigl(r + \gamma \max_{a'}Q(s',a') - Q(s,a)\Bigr)
\]

- \(\alpha\)：学习率（`learning_rate`）
- \(\gamma\)：折扣因子（`discount_factor`）
- \(r\)：即时奖励（这里用的是“塑形后”的 `shaped_reward`）
- \(\max_{a'}Q(s',a')\)：下一状态的最优动作价值估计

代码实现位置：`mini_games_experiment/qlearning_agent.py` 的 `update_q_value()`。

## 3. 状态与动作设计（当前实现）

### 3.1 状态（默认：相对状态）

默认 `use_relative_state=True`，把状态定义为 beacon 相对 marine 的离散网格偏移：

- `grid_size=8`，把 84×84 screen 离散成 8×8 网格
- 状态：`(dx, dy)`，其中 `dx,dy ∈ [-7,7]`

这样状态空间更小、泛化更好（同样相对位置共享同一策略）。

### 3.2 动作（默认：delta 相对位移）

默认 `action_mode="delta"`，动作空间为 9 个相对位移：
- 上/下/左/右
- 四个对角
- 不动

由 `action_step_pixels` 控制每步位移像素（默认 12）。

## 4. 奖励塑形（Reward Shaping）

默认 `reward_shaping=True`，引入距离变化的密集反馈：

```
shaped_reward = env_reward + distance_reward_scale * (prev_distance - current_distance)
```

直觉：如果这一步更接近 beacon，就给一点正反馈；远离则给一点负反馈。

注意：
- **评估时**（`evaluate_qlearning.py`）默认关闭塑形（只看 EnvReward）。
- **训练时**塑形帮助学习更快，但最终成绩仍以 EnvReward 为准。

## 5. 训练与评估

### 5.1 训练

```bash
conda activate sc2_rl
cd D:\Cursor_project\SC2_RL
python mini_games_experiment/MoveToBeacon_1_Qlearning.py --episodes=500 --norender
```

输出文件：
- `models/qlearning_movetobeacon.pkl`：Q 表
- `models/qlearning_movetobeacon_stats.pkl`：训练统计（奖励曲线等）

### 5.2 评估（不学习）

```bash
python mini_games_experiment/evaluate_qlearning.py --episodes=50 --norender --model_path=models/qlearning_movetobeacon.pkl
```

## 6. 指标怎么看（简版）

- **EnvReward**：真实奖励，越大越好（200步内“吃到 beacon 的次数”）。
- **Shaped**：塑形后的累计奖励，仅辅助训练参考。
- **Steps**：当前 episode 步数（通常会到达 `max_steps`）。
- **Epsilon**：探索率（训练初期高，后期低）。
- **Q-table size**：已访问的 (state, action) 条目数（相对状态通常在千级）。


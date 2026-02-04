# 重温星际2强化学习QLearning(一)

> 本文记录我在 StarCraft II（PySC2）mini-game **MoveToBeacon** 上重温表格型 Q-learning 的过程：从问题抽象到代码落地，再到训练指标的理解。  
> 项目源码目录：`D:\Cursor_project\SC2_RL`

## 1. 为什么从 MoveToBeacon 开始

如果一上来就做完整对战，你会同时遇到：
- 超大状态空间（多单位、多视角、长期规划）
- 超大动作空间（大量可用指令、复杂参数）
- 稀疏奖励与延迟信用分配

mini-games 把问题简化得非常“可控”。其中 **MoveToBeacon** 只需要学习一件事：**控制 marine 移动到 beacon**，每次碰到 beacon 都有 +1 奖励，非常适合用来复习 Q-learning 的闭环。

## 2. 环境准备（最短路径）

详细环境配置见：`docs/环境配置说明.md`。这里给最短命令：

```bash
conda activate sc2_rl
cd D:\Cursor_project\SC2_RL
python test_sc2_env.py --norender
```

## 3. 问题抽象：状态、动作、奖励

为了让“表格”可行，必须离散化。

### 3.1 状态（State）

当前实现默认使用**相对状态**（beacon 相对 marine 的位置），把 84×84 screen 划分为 `grid_size×grid_size` 网格（默认 `8×8`），状态定义为：

- $s = (dx, dy)$
- 其中 $dx,dy$ 是 beacon 网格坐标减去 marine 网格坐标

这样做的好处：状态空间显著变小，也更容易泛化（“在左上角/右下角”并不重要，“在左边/右边”更重要）。

### 3.2 动作（Action）

默认动作模式为 `delta`：9 个相对位移（含不动），由 `action_step_pixels` 控制每一步移动的像素距离。

直觉上，这更接近人类控制：不断朝 beacon 方向挪动。

### 3.3 奖励（Reward）

MoveToBeacon 的环境奖励很稀疏：
- 未碰到 beacon：$r=0$
- 碰到 beacon：$r=1$

为了更快学习，本项目在训练时引入了可选的**奖励塑形**（reward shaping），用“距离变近”提供密集反馈；评估时关闭塑形，只看真实奖励。

## 4. Q-learning 的核心公式（贝尔曼更新）

Q-learning 的更新规则可以写成：

$$
Q(s,a) \leftarrow Q(s,a) + \alpha\Bigl(r + \gamma \max_{a'}Q(s',a') - Q(s,a)\Bigr)
$$

其中：
- $\alpha$：学习率（`learning_rate`），控制“这次经验影响多大”
- $\gamma$：折扣因子（`discount_factor`），控制“未来收益的权重”
- $r$：即时奖励
- $\max_{a'}Q(s',a')$：下一状态下最优动作的价值估计

一句话理解：**新的认知 = 旧的认知 + 学习率 ×（目标 - 旧的认知）**。

> 代码实现位置：`mini_games_experiment/qlearning_agent.py` 的 `update_q_value()`。

## 5. 代码结构：训练 / 评估 / Agent

本项目把可复用逻辑放在一个文件里，避免脚本间 flags 冲突。

- **Agent（核心逻辑）**：`mini_games_experiment/qlearning_agent.py`\n  - 状态提取、动作选择、Q 值更新、epsilon 衰减、保存/加载 Q 表
- **训练脚本**：`mini_games_experiment/MoveToBeacon_1_Qlearning.py`\n  - 创建环境\n  - 循环训练 episodes\n  - 定期保存 Q 表与统计数据
- **评估脚本**：`mini_games_experiment/evaluate_qlearning.py`\n  - 加载训练好的 Q 表\n  - 关闭学习（`learning_rate=0`）\n  - 统计多局平均 EnvReward

## 6. 训练与评估命令

训练（500 回合，无渲染）：

```bash
python mini_games_experiment/MoveToBeacon_1_Qlearning.py --episodes=500 --norender
```

评估（50 回合，无渲染，更稳定）：

```bash
python mini_games_experiment/evaluate_qlearning.py --episodes=50 --norender --model_path=models/qlearning_movetobeacon.pkl
```

想看效果就开渲染（少跑几局即可）：

```bash
python mini_games_experiment/evaluate_qlearning.py --episodes=10 --render --model_path=models/qlearning_movetobeacon.pkl
```

## 7. 指标怎么看（简版）

训练日志里你会看到：
- **EnvReward**：真实环境奖励（最重要），在 200 步里吃到 beacon 的次数
- **Shaped**：塑形后的累计奖励，仅用于训练参考
- **Epsilon**：探索率（从大到小衰减）
- **Q-table size**：已访问的 (state, action) 数量（相对状态通常千级）

## 8. 关键矩阵 / 数据结构的维度（对应当前实现）

下面这些“矩阵/数组/表”是你在实现和调参时最常关心的维度信息（默认 `screen_size=84`、`minimap_size=64`、`grid_size=8`、`action_mode=delta`）。

### 8.1 观测（Observation）

- **`obs.observation.feature_screen.player_relative`**：形状 **(84, 84)** 的 2D 数组  
  - 取值含义（常用）：`1`=己方（marine），`3`=中立（beacon）
- **`obs.observation.feature_minimap.*`**：形状 **(64, 64)**（本实验未使用）
- **`obs.reward`**：标量（float），MoveToBeacon 一般为 **0 或 1**

### 8.2 状态（State）

状态不是矩阵，是用来作为 Q 表 key 的 tuple：

- **相对状态（默认）**：$s=(dx,dy)$，长度 **2**  
  - $dx,dy \in [-7,7]$（网格偏移）
- **绝对状态（可选）**：$(mx,my,bx,by)$，长度 **4**  
  - 每个坐标 $\in [0,7]$

### 8.3 动作（Action）

动作表 `agent.actions`：

- **delta 模式（默认）**：长度 **9**，每个元素是 `(dx_pixels, dy_pixels)`  
  - `dx_pixels, dy_pixels ∈ {-step,0,+step}`，默认 `step=12`
- **grid_absolute 模式**：长度 **grid_size²**（例如 `8²=64`），每个元素是 `(target_x,target_y)`（像素坐标）

### 8.4 Q 表（Q-table）

Q 表是一个字典（严格来说是 `defaultdict(float)`），不是二维矩阵：

- **Key**：`(state, action)`  
  - `state`：长度 2 或 4 的 tuple  
  - `action`：动作索引 `int`（delta 模式下为 0~8）
- **Value**：`Q(s,a)`，标量 float

当你在某个状态下取所有动作的 Q 值时，会形成一个长度为 `num_actions` 的列表：

- `q_values = [Q(s,0), Q(s,1), ..., Q(s,num_actions-1)]`  
  - delta 模式下维度就是 **(9, )**

## 8. 小结与下一篇

这一篇完成了 Q-learning 的“从公式到代码闭环”。下一篇打算做两件事：
- 用更合理的状态表示（例如加入速度/方向，或更细粒度网格）
- 对比 DQN 等方法在 MoveToBeacon 上的表现与工程成本

项目文档索引见根目录 `README.md`，以及 `docs/qlearning/Qlearning说明.md`。


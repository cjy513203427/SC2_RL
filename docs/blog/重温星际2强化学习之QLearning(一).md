# 重温星际2强化学习QLearning(一)

> 本文记录我在 StarCraft II（PySC2）mini-game **MoveToBeacon** 上重温表格型 Q-learning 的过程：从问题抽象到代码落地，再到训练指标的理解。  
> 项目源码：[https://github.com/cjy513203427/SC2_RL](https://github.com/cjy513203427/SC2_RL)

## 1. 为什么从 MoveToBeacon 开始

如果一上来就做完整对战，你会同时遇到：
- 超大状态空间（多单位、多视角、长期规划）
- 超大动作空间（大量可用指令、复杂参数）
- 稀疏奖励与延迟信用分配

mini-games 把问题简化得非常“可控”。其中 **MoveToBeacon** 只需要学习一件事：**控制 marine 移动到 beacon**，每次碰到 beacon 都有 +1 奖励，非常适合用来复习 Q-learning 的闭环。

## 2. 快速开始 (Quick Start)

详细环境配置见：`docs/环境配置说明.md`。这里提供最简运行命令：

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

- **Agent（核心逻辑）**：`mini_games_experiment/qlearning_agent.py`
  - 状态提取、动作选择、Q 值更新、epsilon 衰减、保存/加载 Q 表
- **训练脚本**：`mini_games_experiment/MoveToBeacon_1_Qlearning.py`
  - 创建环境
  - 循环训练 episodes
  - 定期保存 Q 表与统计数据
- **评估脚本**：`mini_games_experiment/evaluate_qlearning.py`
  - 加载训练好的 Q 表
  - 关闭学习（`learning_rate=0`）
  - 统计多局平均 EnvReward

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

## 7. 训练指标详细解读

训练日志会打印类似这一行（每 10 个 episode 打印一次）：

```
Episode  500/500 | EnvReward:    4.0 | Shaped:    5.00 | Avg(20):    6.9 | Steps:  200 | Epsilon: 0.0100 | Q-table size: 1413
```

下面逐项解释，并说明**这些数到底怎么来的**、怎么判断“训练变好了”。

### 7.1 EnvReward（真实环境奖励，最重要）

**它是什么**：PySC2 环境返回的真实奖励（MoveToBeacon 中通常是碰到 beacon 得 +1）。  
**它怎么统计**：每一步的 `obs.reward` 累加成一局的总和：

$$
\text{EnvReward}_\text{episode}=\sum_{t=1}^{T} r_t
$$

在代码里对应 `qlearning_agent.py`：`env_reward = float(obs.reward)`，然后 `self.total_env_reward += env_reward`。

**怎么解读**：
- **越大越好**：表示一局（最多 200 步）里吃到 beacon 的次数更多。
- **评估以它为准**：训练时即使开了奖励塑形（Shaped），最后能力是否真的提升，要看 `EnvReward`（尤其是 `evaluate_qlearning.py` 的统计）。

### 7.2 Shaped（塑形奖励，用于“更快学”，不用于对外评估）

**它是什么**：在 `EnvReward` 基础上加了“距离变近就给一点正反馈”的密集奖励，形式是：

$$
\text{Shaped}_t = r_t + \lambda \cdot (d_{t-1}-d_t)
$$

- $d_t$：marine 与 beacon 的欧氏距离
- $\lambda$：`distance_reward_scale`（默认 0.02）

代码里对应 `qlearning_agent.py`：
- `shaped_reward = env_reward`
- 若开启 `reward_shaping=True` 且能算出距离，则 `shaped_reward += scale*(prev_dist - curr_dist)`
- `self.total_shaped_reward += shaped_reward`

**怎么解读**：
- **只作为训练过程参考**：不同塑形系数/不同状态表示下，`Shaped` 的绝对值不可横比。
- 你可以看它是否大体随训练上升、是否比 `EnvReward` 更平滑（通常会更平滑）。

### 7.3 Avg(20) / Average reward（最近 20 局的滑动平均）

你日志里的 `Avg(20)` 指的是：**最近 20 局（不足 20 局则用已有的局数）的 EnvReward 平均值**。

训练中计算方式（`MoveToBeacon_1_Qlearning.py`）是：
- `window_size = min(20, len(episode_rewards))`
- `avg_reward = mean(episode_rewards[-window_size:])`

用公式写就是：

$$
\text{Avg}(20)_k=\frac{1}{W}\sum_{i=k-W+1}^{k}\text{EnvReward}_i,\quad W=\min(20,k)
$$

训练结束时打印的：
- `Average reward (last 20 episodes)`：固定取最后 20 局 `EnvReward` 的均值（当总局数 ≥ 20 时）

**怎么解读**：
- **看趋势，不看单点**：单局 EnvReward 波动很大，`Avg(20)` 更能反映策略是否真的变好。
- **有用的判断标准**：`Avg(20)` 持续上升并趋于稳定，基本说明学习有效；如果长期贴近 0，通常是状态/动作抽象或奖励设计有问题。

### 7.4 Steps / Average steps（每局步数）

**它是什么**：一局走了多少 step（你的脚本里每次 `agent.step()` 会 `self.steps += 1`）。  
**为什么经常是 200**：`MoveToBeacon_1_Qlearning.py` 里有 `max_steps`（默认 200），即便 agent 表现很好，通常也会跑满时间上限，所以 `Steps` 很常见一直是 200。

**怎么解读**：
- 在这个任务里 **Steps 不是关键指标**，更应该看 `EnvReward` / `Avg(20)`。
- 如果你改了逻辑让“吃到 beacon 就提前结束”，那 Steps 才会有更明显意义（更少步数代表更快完成）。

### 7.5 Epsilon（探索率）

**它是什么**：epsilon-greedy 的探索概率。每一步：
- 以概率 $\epsilon$ 随机选动作
- 以概率 $(1-\epsilon)$ 选当前 Q 值最大的动作

脚本里每个 episode 结束后衰减一次：

$$
\epsilon \leftarrow \max(\epsilon_\text{end},\epsilon\cdot \epsilon_\text{decay})
$$

**怎么解读**：
- 训练前期 $\epsilon$ 大：多探索，Q 表填得快但回报波动大
- 后期 $\epsilon$ 小：更像“按学到的策略执行”，`Avg(20)` 往往更稳定

### 7.6 Q-table size（Q 表规模）

**它是什么**：`q_table` 里已访问过的 `(state, action)` 键值对数量（字典条目数）。  
**它怎么来的**：每次更新/访问都会让某些 `(s,a)` 出现在表里，所以它随探索增长。

**怎么解读**：
- **相对状态**（默认 \(s=(dx,dy)\)）会把状态空间压得很小，所以规模通常是**千级**（你示例里 1413）。
- 如果你换成**绝对状态**或更细网格，Q 表可能迅速变成**万级/十万级**，学习会慢很多、也更容易稀疏。

### 7.7 Best episode reward（单局最好成绩）

**它是什么**：所有 episode 里最大的 `EnvReward`。  
**怎么解读**：它能说明“最好能达到多高”，但容易是偶然性；通常更关注 `Avg(20)` 和评估集平均值。

> 实战建议：判断“是否真的学会了”，优先跑 `evaluate_qlearning.py`（关闭学习与塑形），看多局平均 `EnvReward`，比训练日志更可靠。

## 8. 核心数据结构与维度说明

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

## 9. 小结与后续计划

这一篇完成了 Q-learning 的“从公式到代码闭环”。下一篇打算做两件事：
- 用更合理的状态表示（例如加入速度/方向，或更细粒度网格）
- 对比 DQN 等方法在 MoveToBeacon 上的表现与工程成本

项目文档索引见根目录 `README.md`，以及 `docs/qlearning/Qlearning说明.md`。


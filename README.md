<div align="center">
  <h1 align="center">Legged Robot Walkthrough</h1>
</div>

<p align="center">
    <strong>这是一个基于Unitree Go2的强化学习示例仓库。</strong> 
</p>

---

## 安装配置

安装和配置步骤请参考 [setup.md](/doc/setup_zh.md)

## 流程说明

强化学习实现运动控制的基本流程为：

`Train` → `Play` → `Sim2Sim` → `Sim2Real`

- **Train**: 通过 Gym 仿真环境，让机器人与环境互动，找到最满足奖励设计的策略。通常不推荐实时查看效果，以免降低训练效率。
- **Play**: 通过 Play 命令查看训练后的策略效果，确保策略符合预期。
- **Sim2Sim**: 将 Gym 训练完成的策略部署到其他仿真器，避免策略小众于 Gym 特性。
- **Sim2Real**: 将策略部署到实物机器人，实现运动控制。

本仓库将主要基于前三步进行开发。

## 🛠️ 使用指南

### 1. 训练

运行以下命令进行训练：

```bash
python legged_gym/scripts/train.py
```

#### ⚙️  参数说明
- `--headless`: 默认启动图形界面，设为 true 时不渲染图形界面（效率更高）
- `--resume`: 从日志中选择 checkpoint 继续训练
- `--experiment_name`: 运行/加载的 experiment 名称
- `--run_name`: 运行/加载的 run 名称
- `--load_run`: 加载运行的名称，默认加载最后一次运行
- `--checkpoint`: checkpoint 编号，默认加载最新一次文件
- `--num_envs`: 并行训练的环境个数
- `--seed`: 随机种子
- `--max_iterations`: 训练的最大迭代次数
- `--sim_device`: 仿真计算设备，指定 CPU 为 `--sim_device=cpu`
- `--rl_device`: 强化学习计算设备，指定 CPU 为 `--rl_device=cpu`

**默认保存训练结果**：`logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`

---

### 2. Play

如果想要在 Gym 中查看训练效果，可以运行以下命令：

```bash
python legged_gym/scripts/play.py
```

**说明**：

- Play 启动参数与 Train 相同。
- 默认加载实验文件夹上次运行的最后一个模型。
- 可通过 `load_run` 和 `checkpoint` 指定其他模型。

#### 💾 导出网络

Play 会导出 Actor 网络，保存于 `logs/{experiment_name}/exported/policies` 中：
- 普通网络（MLP）导出为 `policy_1.pt`
- RNN 网络，导出为 `policy_lstm_1.pt`

### 3. Sim2Sim (Mujoco)

支持在 Mujoco 仿真器中运行 Sim2Sim：

```bash
python deploy/deploy_mujoco/deploy_mujoco.py go2.yaml
```

#### 参数说明
- `config_name`: 配置文件，默认查询路径为 `deploy/deploy_mujoco/configs/`

#### ➡️  替换网络模型

默认模型位于 `deploy/pre_train/{robot}/motion.pt`；自己训练模型保存于`logs/go2/exported/policies/policy_1.pt`，只需替换 yaml 配置文件中 `policy_path`。

## 🎉  致谢

本仓库开发离不开以下开源项目的支持与贡献，特此感谢：

- [legged\_gym](https://github.com/leggedrobotics/legged_gym): 构建训练与运行代码的基础。
- [rsl\_rl](https://github.com/leggedrobotics/rsl_rl.git): 强化学习算法实现。
- [mujoco](https://github.com/google-deepmind/mujoco.git): 提供强大仿真功能。
- [unitree\_rl\_gym](https://github.com/unitreerobotics/unitree_rl_gym)]: 提供代码基础及机器人模型。
- [walk\_these\_ways](https://github.com/Improbable-AI/walk-these-ways): 提供代码基础及算法实现。

---

## 🔖  许可证

本项目根据 [BSD 3-Clause License](./LICENSE) 授权：
1. 必须保留原始版权声明。
2. 禁止以项目名或组织名作举。
3. 声明所有修改内容。

详情请阅读完整 [LICENSE 文件](./LICENSE)。


## TODO

- [ ] Find out how gait training works in the code, and how to manually change the gait
- [ ] For privileged observation, the walk-these-ways method is slightly different with the original one. Find out the difference.
- [ ] Actuator net (`compute_torques`, `init_buffer`)

## Commands

`lin_vel_x`, `lin_vel_y`, `ang_vel_yaw`, `body_height`, `gait_frequency`, `gait_phase`, `gait_offset`, `gait_bounds`, `gait_duration`, `footswing_height`, `body_pitch`, `body_roll`, `stance_width`, `stance_length`, `aux_reward_coef`
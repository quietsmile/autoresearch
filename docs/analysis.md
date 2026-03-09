# Autoresearch 分析：改进方向 / 模型规模 / 多机多卡

> 分析时间：2026-03-08
> 代码路径：/newcpfs/user/yuchen/agent/autoresearch/

---

## 一、当前模型规模

从 run.log 和 results.tsv 最新状态（768-dim 配置）：

| 参数分组              | 数量     |
|----------------------|----------|
| wte (embedding)      | 6.3M     |
| value_embeds         | 25.2M    |
| lm_head              | 6.3M     |
| transformer matrices | 56.6M    |
| scalars              | 16       |
| **total**            | **~94M** |

架构：8层 Transformer，768 维，6 头，vocab=8192，seq_len=2048，窗口模式 SSSL
VRAM 占用：约 64GB（results.tsv 中 keep 项均为 63.9GB）
当前最优 val_bpb = **0.978553**（QK-norm before RoPE，commit d0d85b2）

---

## 二、可以进一步改进的方向

### A. 改进 autoresearch 框架本身（更高杠杆）

1. **Agent 间知识共享**
   - 现状：8 个 GPU worker 完全独立，互不感知对方实验结果
   - 改进：共享一个全局 results.tsv（合并表），Agent 开始新实验前先读取所有 GPU 历史
   - 效果：避免重复实验（如多个 GPU 同时尝试 GQA），加速探索收敛

2. **更强的 program.md**
   - 加入"已知无效方向"列表（从 results.tsv discard 项提取），减少重复踩坑
   - 加入文献参考（Chinchilla scaling, Muon 原始论文, NorMuon, ResFormer 等）
   - 加入实验设计策略（grid search / ablation / 组合先验改进的优先级）

3. **自动化启动 Agent（真正无人值守）**
   - 现状：需要人工在每个 tmux session 里手动打开 Claude Code
   - 改进：用 `claude -p "有空看一下 program.md，开始新一轮实验" --dangerously-skip-permissions` 非交互式启动
   - 可配合 launch.sh 改造成全自动流程，人不在场也能启动

4. **GPU 间分工**
   - 给每个 GPU 的 program.md 指定不同探索方向（超参 / 架构 / 优化器 / 数据）
   - 避免 8 个 Agent 都集中在超参调整，忽视架构创新

### B. 改进训练代码本身（Agent 可直接做的）

1. **更大模型空间探索**
   - 当前最优 768-dim，1024-dim 区域未充分探索
   - DEPTH × ASPECT_RATIO 的 Pareto 前沿尚有空间

2. **注意力机制替换**
   - Linear attention / RetNet 风格
   - ALiBi 位置编码替换 RoPE

3. **优化器参数精调**
   - Muon ns_steps（当前5）搜索空间
   - beta1/beta2 联合网格搜索

4. **数据增强**
   - 增加训练 shards（当前仅 10 个，总共有 6542 个可用）
   - 数据混合比例调整

---

## 三、改成多机多卡的复杂度

**结论：中等到高复杂度，有 3 个核心阻塞点。**

### 阻塞点 1：MuonAdamW 明确标注"single GPU only"

```python
# train.py:293
# Optimizer (MuonAdamW, single GPU only)
```

根源：`torch.stack([p.grad for p in params])` 把同形状参数堆叠成大张量做批量正交化（Polar Express）。DDP 模式下梯度由 reducer 管理，无法直接 stack；FSDP 模式下参数被分片，更不可行。

修复路径：
- 单机多卡 DDP：`optimizer.step()` 前手动 `allreduce` 梯度，再做正交化 → **约 100 行，可行**
- 多机多卡 FSDP：重写 Muon 为 ZeRO-style sharded optimizer → **约 300 行，较复杂**

### 阻塞点 2：Dataloader 无 rank 感知

`prepare.py` 的 `make_dataloader` 没有 `rank/world_size` 参数，所有进程会读取相同数据。

修复：为 `_document_batches()` 增加 `rank, world_size` 参数，按 rank 对 parquet 文件列表做 stride 分片。**约 20-30 行，相对简单。**

### 阻塞点 3：训练循环无分布式初始化

需加入：
```python
dist.init_process_group(backend="nccl")
model = DDP(model, device_ids=[local_rank])
```
**约 40-50 行，机械性改动。**

### 工作量汇总

| 目标                             | 改动量       | 难度 |
|----------------------------------|-------------|------|
| 单机 8 卡 DDP（真正数据并行）      | ~200 行     | 中等 |
| 多机多卡 DDP（如 4机×8卡=32卡）   | ~250行+集群配置 | 中高 |
| 多机 FSDP（更大模型+显存分片）     | ~400 行     | 高   |

### 关键判断：多卡对 autoresearch 框架本身价值有限

当前 `launch.sh` 已实现"8 倍实验并发"（每卡独立跑实验），对于 autoresearch 的设计哲学（**固定 5 分钟比较架构**）已经足够好。

真正多卡训练的收益：
- ✅ 可训练更大模型（需 FSDP）
- ✅ 相同时间内处理更多 token，提升单次实验质量
- ❌ 破坏 5 分钟基准的可比性
- ❌ 实现成本较高

**更高性价比的改进方向是优化 program.md 和 Agent 协作策略，而不是做多卡改造。**

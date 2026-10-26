# Low latency dispatch 性能测试指南（EP16 / EP32）

本文面向 `kernel/ll-ndr-hns1825` 分支的 Ascend 950 NDR 实现。它提供：

- 单机 16 卡（EP16）检查和测量；
- 两机 32 卡（EP32，每机 16 卡）完整测量；
- 未插桩算子性能，以及带阶段计时的独立诊断；
- 生成可归档的原始数据、性能摘要和分段耗时 CSV。

## 1. 测试范围与前置条件

- 目标系统需为本分支支持的 Ascend 950 + HNS_1825 环境，Linux、Python ≥3.11、CANN、匹配版本的 `torch`/`torch_npu`、C++17、CMake ≥3.20，以及 Make 或 Ninja。
- 原生 LL 的合法本机组是 16 个连续 rank。单机 16 卡是 EP16；跨机测试必须为 EP32，两机各 16 卡。跨机 RDMA 需要目标机 provider、NIC/驱动和拓扑配置正确。
- 所有节点使用相同 Git 提交、SHMEM SDK、构建开关、shape 和运行参数。LL 入口显式选择本分支的混合后端；单机 EP16 走机内 MTE，不代表跨机 RDMA 已验收。
- 正式性能与阶段诊断分开采集：正式基线使用 timer OFF；阶段分解使用 timer ON + `observe`，有插桩开销，不能当作正式性能数值。

## 2. 获取代码与构建

在每台测试机执行。两机必须检出同一提交；不要在测试期间一台机器更新代码或 SDK。

```bash
git clone --branch kernel/ll-ndr-hns1825 --single-branch \
  https://github.com/TangYunfei147/DeepEP-LL.git DeepEP-LL-ndr
cd DeepEP-LL-ndr
git submodule update --init --recursive --depth 1
git rev-parse HEAD

# 加载本机 CANN；也可以先 export CANN_ENV=/实际路径/set_env.sh
source tests/ll_rdma/env.sh
python3 -m venv --system-site-packages .venv-ll-rdma
source .venv-ll-rdma/bin/activate
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"

unset SHMEM_ROOT CATLASS_ROOT
python scripts/build_shmem.py
export DEEPEP_NPU_ARCH=Ascend950
python -m pip install -r requirements-build.txt
```

`tests/ll_rdma/env.sh` 会加载 `scripts/ll_timer_env.sh`，为 CANN/SHMEM 设置本仓库运行环境。首次准备匹配的 SDK 后，按下面的模式编译并安装 wheel。修改 kernel 或切换 ON/OFF 后都要重新安装；多机必须在每台机器安装同一构建模式的产物。

阶段诊断构建：

```bash
export DEEPEP_LL_STAGE_TIMER=ON
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

正式性能构建：

```bash
export DEEPEP_LL_STAGE_TIMER=OFF
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

完整 wheel 会编译 Host 扩展和 Device kernel。对同一份代码先后测 OFF 和 ON 时，每次切换开关都重新执行安装命令；不要用 ON wheel 的结果替代 OFF 性能基线。

## 3. 单机 EP16

单机测试需要 16 张已分配的卡。先运行 CPU 计划检查，再执行设备测试。示例使用全部可见卡 `0..15`，请按实际分配情况改 `DEVICES`；端口需空闲。

```bash
source .venv-ll-rdma/bin/activate
source tests/ll_rdma/env.sh
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE
DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

python tests/ll_stage_timer/run.py --devices "$DEVICES" \
  --config tests/ll_rdma/target16.json --node-rank 0 --plan
```

**阶段诊断**（本机 16 卡）：

```bash
export DEEPEP_LL_STAGE_TIMER=ON
# 确认安装的是 timer ON wheel
RUN=ep16-timer-001
python tests/ll_stage_timer/run.py --devices "$DEVICES" \
  --num-nodes 1 --node-rank 0 --master-addr 127.0.0.1 \
  --master-port 29671 --shmem-port 29672 \
  --num-tokens 32 --num-max-tokens-per-rank 256 \
  --hidden 7168 --num-topk 6 --num-experts 128 --dtype bf16 --do-cpu-sync 0 \
  --warmup 10 --num-iters 100 --ll-stage-timer --ll-timer-iters 5 \
  --ll-timer-mode observe --skip-perf-test --dispatch-only \
  --output-dir results/ll-dispatch --run-name "$RUN"
```

此模式先预热并执行稳定调用，然后独立采 5 轮 dispatch 阶段；`--skip-perf-test` 表示不运行正式 profiler。只测 dispatch，因此不会把 combine 混进本次诊断。

分析并导出阶段性能文件：

```bash
python tests/ll_stage_timer/analyze.py "results/ll-dispatch/$RUN" \
  --op dispatch --trace-iteration 0
```

分析目录默认为 run 下的 `stage-analysis-dispatch/`，主要文件：

- `stages.csv`：每个 rank / AIV 核 / 阶段的统计值（mean、min、max、P50、P90、P99），是分段耗时表；
- `summary.json`：完整配置、样本范围、阶段统计、慢 rank/核及诊断摘要；
- `trace.json`：所选采样轮次的时间线，可在 Perfetto 或 Chrome trace viewer 打开；
- `bottleneck_summary.md`：可读的瓶颈线索和阶段解释；
- `bandwidth.csv`：计时器记录的通信字节与有效带宽估算。该估算不是 NIC 线速。

原始 `sample_*.json` 保存在 `results/ll-dispatch/$RUN/case_*/normal/ll_stage_timer/dispatch/rank_*/`。保留原始样本，方便使用更新后的分析器重新生成报告。

## 4. 两机 EP32：阶段诊断

每台机器都按第 2 节安装相同 timer ON wheel。每机选 16 张卡，节点 0 有 global rank 0..15，节点 1 有 global rank 16..31。两机都使用节点 0 的可达 IPv4 地址、相同端口和相同 `RUN`；在两台主机上**同时**启动实际命令。

```bash
source .venv-ll-rdma/bin/activate
source tests/ll_rdma/env.sh
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE
DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
MASTER_IP=192.0.2.10   # 替换为节点 0 的真实、节点 1 可达的 IPv4
RUN=ep32-timer-001     # 两机相同；每次新实验使用新名字
NODE_RANK=0            # 节点 0 设 0；节点 1 设 1

export DEEPEP_LL_STAGE_TIMER=ON
python tests/ll_stage_timer/run.py --backend rdma --devices "$DEVICES" \
  --num-processes 16 --num-nodes 2 --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_IP" --master-port 29500 --shmem-port 29501 \
  --num-tokens 8 --num-max-tokens-per-rank 256 \
  --hidden 3584 --num-topk 8 --num-experts 384 --dtype bf16 --do-cpu-sync 0 \
  --warmup 10 --num-iters 100 --ll-stage-timer --ll-timer-iters 5 \
  --ll-timer-mode observe --ll-timer-frequency-hz 1000000000 \
  --skip-perf-test --dispatch-only \
  --output-dir results/ll-dispatch --run-name "$RUN"
```

**执行时的设置：** 节点 0 设置 `NODE_RANK=0`，节点 1 设置 `NODE_RANK=1`；其他参数必须相同。节点 0/1 的 rank 数据分别写入本机工程的同名 run 目录。跨机模式先做 RDMA preflight；失败时应先解决环境问题，不要改成 MTE 后把结果记成跨机性能。

每台机器的原始文件都包含本机 16 个 rank，可分别执行分析器，得到各自 `stage-analysis-dispatch/`。分析器按 global rank 保留 EP32 身份，默认按本机 16 rank 做报告。若需汇总全 32 rank，把两机**同一 run** 的 rank 原始文件按原有相对目录合并，保留 run manifest，再使用 `--require-all-ranks` 分析；不要覆盖同名 rank 文件或混合不同实验。

```bash
# 每个节点在本机运行：只分析该节点完整的 16-rank 组
python tests/ll_stage_timer/analyze.py "results/ll-dispatch/$RUN" \
  --op dispatch --trace-iteration 4

# 若已将两机 raw rank 文件合并到同一个 RUN 目录：要求完整 EP32
python tests/ll_stage_timer/analyze.py /path/to/merged-run \
  --op dispatch --require-all-ranks --trace-iteration 4 \
  --output-dir /path/to/merged-run/stage-analysis-dispatch-ep32
```

## 5. 两机 EP32：正式 dispatch 性能

保持硬件、拓扑、shape 和路由与 ON 诊断相同；以下用 `tests/ll_rdma/one_click.sh` 同时执行 timer OFF 构建、正确性与正式性能采样。该入口会为当前 mode 设置 `DEEPEP_LL_STAGE_TIMER=OFF`、安装并核验 OFF wheel；无需先手动安装 OFF wheel。它默认每机设备 `0..15`、10 次预热、100 次采集、尾部 50 个样本统计，并执行 RDMA preflight 和两机参数/提交核对。两边要同时运行，run 名和端口一致。

```bash
# 节点 0
bash tests/ll_rdma/one_click.sh 0 192.0.2.10 ep32-perf-001 --mode perf
# 节点 1（与节点 0 同时启动）
bash tests/ll_rdma/one_click.sh 1 192.0.2.10 ep32-perf-001 --mode perf
```

`192.0.2.10` 要替换为节点 0 的真实可达 IPv4。性能结果位于 `results/ll-ndr/ep32-perf-001-perf/`，rank 0 输出性能汇总，各节点保留本地 profiler 原始数据。这个基准默认 shape 是 B=8、H=3584、top-k=8、E=384、BF16 和 FP8；正式比较要固定参数，或通过 `--tokens`、`--dtype`、`--capacity`、`--hidden`、`--topk`、`--experts` 显式指定同一组目标参数。诊断 shape 应与正式性能 shape 一致。

单机正式基线可用 `tests/ll_stage_timer/run.py` 在 16 卡 EP16 上运行同一 shape，移除 `--ll-stage-timer` 和 `--skip-perf-test`，并确保 OFF wheel 已安装。EP16 和 EP32 的数据分别记录，单机结果不代表跨机扩展性能。

## 6. 如何读阶段计时

- 计时默认 1 GHz，分析器按 `ticks / 1000` 换算为微秒；原始 tick 保留在样本中。
- `observe` 不增加阶段间流水屏障；MTE/RDMA 阶段的提交耗时不一定等于传输完成时间。尤其跨机 `send_issue` 含暂存、RDMA put 和 quiet，不能解释为纯网络耗时。
- 阶段可能嵌套或在不同 AIV 核上并行。不要把 `stages.csv` 所有行相加当成 dispatch 总延迟；查看 `summary.json` 的每 rank 最大核统计和时间线，区分并行阶段、慢 rank 与阶段内部等待。
- `timer ON` 会引入设备计数器读取和记录写回开销。dispatch 总性能以 timer OFF 的 profiler/task 采样为准，ON 报告用于定位阶段。
- 对外报告请同时记录：Git SHA、SDK/CANN/驱动版本、机器与 NIC 拓扑、rank 到机器/设备映射、wheel timer 开关、shape/dtype/路由、warmup/采样次数、完整命令和 preflight 结果。

更完整的阶段字段定义与统计限制见 [LL 阶段打点说明](../ll_stage_timer/README.md)；两机一键流程见 [ONE_CLICK.md](ONE_CLICK.md)。

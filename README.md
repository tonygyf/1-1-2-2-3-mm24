# ExtremeOCR

极端成像条件下的场景文本检测与识别研究项目，按 CCF-B 工作量推进。主线是**局部高光/饱和下的端到端 text spotting**，主 baseline 为 LRANet++。资源为 4 × RTX 6000D（约 83 GiB/卡）。

已在 Docker 内下载正式 Total-Text 和作者权重，完成现代 CUDA 移植、真实推理、单卡/双卡 30 步训练与断点恢复诊断。300 张官方测试图的上游扫描实测为 **检测 90.11%、无词典端到端 84.80%**；权重训练 profile 尚未核清，**尚不判定论文精度复现通过**。软件、运行环境和方法有效性分别验收，详见 [当前状态](docs/STATUS.md) 和 [初始化报告](outputs/INITIALIZATION_REPORT.md)。

## Docker 开始

宿主机只负责 Docker CLI。Python、下载器、依赖安装、扩展编译、数据处理和实验均在容器内进行；无需在宿主机安装 Python 包。现代镜像从固定 NGC digest、固定源码 archive 和校验过的补丁重新构建。

```bash
# 在项目根目录；构建的 RUN 步骤均在容器内执行。
docker build --memory 48g --memory-swap 48g --shm-size 8g \
  --build-arg LOCAL_UID=$(id -u) --build-arg LOCAL_GID=$(id -g) \
  -f docker/Dockerfile.modern -t extremeocr:lranet-modern .
export EXTREMEOCR_IMAGE_ID=$(docker image inspect --format '{{.Id}}' extremeocr:lranet-modern)
LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) docker compose up -d --no-build
docker compose exec research python scripts/preflight.py
docker compose exec research python scripts/fetch_baseline.py
docker compose exec research python scripts/verify_local.py
```

Compose 默认只在 Python 中使用 GPU 0，CPU RAM 上限 48 GiB、无额外 swap、12 核 CPU、8 GiB shm；不要占用他人的 GPU 任务。多卡诊断需显式设置空闲卡，例如 `EXTREMEOCR_GPUS=0,1 docker compose up -d --no-build`，训练命令对应 `--gpus 2`。当前服务器的四卡同步训练尚未验证。

## 数据与初始化权重

```bash
docker compose exec research python scripts/download_assets.py
docker compose exec research python scripts/split_training.py \
  --images data/raw/totaltext/training \
  --annotations data/raw/totaltext/totaltext_train.json \
  --output data/derived/totaltext_split --seed 42
docker compose exec research python scripts/prepare_checkpoint.py
docker compose exec research python scripts/server_gate.py \
  --checkpoint checkpoints/upstream/lranet_pp/totaltext_runtime.pth \
  --output work/runs/server-lifecycle-new.json
```

数据下载器支持断点续传、固定 SHA256、ZIP CRC、图像解码/尺寸和 1,255/300 数量检查。保留作者 JSON 顺序；训练图像根目录为 `training`，测试根目录为 `totaltext`。现有 split、suite、run 和运行权重不会被自动覆盖；已有资源直接复用，创建新 run 时换名。

官方下载权重与发布代码存在旧键名/冗余层差异。`prepare_checkpoint.py` 仅处理固定 SHA256 的这一份作者权重：保留全部匹配 tensor，按官方非严格加载行为补入 TPS 常量、移除不被当前代码使用的键。原始权重保持原样；转换记录、原始/运行权重哈希均保留。未知权重不能套用该策略。

## 诊断与短程训练

```bash
docker compose exec research python -m extremeocr.cli build \
  --images data/raw/totaltext/training \
  --annotations data/derived/totaltext_split/totaltext_dev.json \
  --output data/derived/totaltext_dev --config configs/diagnostic.json

docker compose exec research python scripts/lranet_run.py eval \
  --images data/derived/totaltext_dev/clean/images \
  --annotations data/derived/totaltext_dev/clean/totaltext_dev.json \
  --checkpoint checkpoints/upstream/lranet_pp/totaltext_runtime.pth \
  --split dev --protocol fixed --score-probe --run dev-clean-new --execute

docker compose exec research python scripts/lranet_run.py train \
  --images data/raw/totaltext/training \
  --annotations data/derived/totaltext_split/totaltext_train.json \
  --checkpoint checkpoints/upstream/lranet_pp/totaltext_runtime.pth \
  --module none --stage probe --gpus 1 --seed 42 --run baseline-probe-new --execute
```

`probe` 为 30 步，在第 15/30 步保存。恢复训练使用 `--checkpoint <前次training/iter_15.pth> --resume`，并保持 module/levels 一致；它恢复迭代、optimizer 和 EMA。断点恢复是生命周期诊断，不承诺 DataLoader/RNG 或跨 world-size 的逐步轨迹一致。五组等预算研究对照和正式三 seed SOP 见 [EXPERIMENTS.md](docs/EXPERIMENTS.md)。

本轮按用户确认止于初始化和诊断，不自动启动 500/10,000 步方法实验。不要用官方测试集选研究阈值或模块；普通微调和容量对照仍是方法有效性所必需的实验。

## 项目 skills

`.agents/skills/` 中 ponytail、autoresearch、grill-me 和 grilling 均已读取并用于本轮：最小完整实现、有界实测迭代、范围/研究边界确认。grill-me 是 grilling 的别名。ponytail 固定版本与其他 skill 来源在 `configs/upstream.lock.json`；未执行第三方安装 hooks。新增 skill 可在下一轮自动发现，本轮已显式读取应用。

## 目录约定

| 位置 | 职责 |
| --- | --- |
| `src/extremeocr/` | 退化、数据协议、模型插件 |
| `configs/` | 固定协议、资源来源/哈希、移植和 checkpoint 兼容清单 |
| `docker/`、`compose.yaml` | 可重建镜像、依赖、补丁与容器资源配置 |
| `scripts/`、`tests/` | 下载、准备、实验、审计及必要回归验证 |
| `data/`、`checkpoints/` | 数据和权重，实际文件不进 Git |
| `work/upstream/`、`work/ports/` | 官方快照和单独兼容移植源码，不进 Git |
| `work/runs/<run>/` | 运行快照、日志、预测、临时训练 checkpoint |
| `outputs/` | 初始化报告、正式表格、可交付结果 |
| `.agents/skills/` | 项目级 skills |

研究背景见 [PROJECT_BRIEF.md](outputs/PROJECT_BRIEF.md)，固定门槛见 `configs/quality_gates.json`。正式资源许可及权重数据暴露见 [data](data/README.md)、[checkpoints](checkpoints/README.md)。

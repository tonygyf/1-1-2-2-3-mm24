# Docker 与资源约束

更新：2026-10-22，Asia/Shanghai。Docker daemon 已可用；本轮安装、下载、编译和实验均在新建的本项目容器或 Docker build 的 RUN 步骤中完成，没有向宿主机安装依赖，也未修改其他项目容器。

## 当前验证

- 实测硬件：4 × NVIDIA RTX 6000D，compute capability 12.0，单卡 89,173,131,264 bytes（约83.1 GiB）。GPU 2 有其他任务，本轮使用 0/1/3。
- 基础镜像：`nvcr.io/nvidia/pytorch@sha256:b5541f591b87e68d3883040bfdbf89707bd6f80e2e38b95411c9ff3e1cb63eeb`；Python3.12.3、torch2.10.0a0 NVIDIA25.12、CUDA13.1。
- MMCV1.7.2 源码编译 sm_120，MMDetection2.15.1；保留旧 API，不能换成 MMCV2/MMDet3。
- 实测 DCNv2/custom sampler CUDA 前后向、原 FPN 与零初始化 adapter 恒等、真实图片推理、单卡/双卡30步、optimizer/EMA/iteration 恢复通过。四卡同步尚未验证。
- `docker/Dockerfile.reference` 保留为历史环境参考，CUDA11.1 不支持本卡的 sm_120，不用于当前实验。

## 重建与启动

```bash
docker build --memory 48g --memory-swap 48g --shm-size 8g \
  --build-arg LOCAL_UID=$(id -u) --build-arg LOCAL_GID=$(id -g) \
  -f docker/Dockerfile.modern -t extremeocr:lranet-modern .
export EXTREMEOCR_IMAGE_ID=$(docker image inspect --format '{{.Id}}' extremeocr:lranet-modern)
docker compose up -d --no-build
docker compose exec research bash
```

进入 shell 后运行项目 Python 命令。没有 buildx 的主机可直接用上述 legacy build；不要为了下载/编译而在宿主机安装工具。镜像构建由固定 archive + `docker/patches/` 重放源码移植，`prepare_modern_port.py` 核验原包及补丁 SHA。构建上下文排除数据、权重和运行产物，只允许配置、准备脚本、补丁及历史镜像使用的官方源码快照；容器运行通过 `/workspace` 挂载项目。

镜像内官方模型代码为 `/opt/vendor/lranet`，MMDetection 为 `/opt/vendor/mmdetection`，MMCV 为安装的 CUDA wheel。`EXTREMEOCR_SOURCE_VARIANT=modern` 使源码审计检查显式的 `configs/modern_port.json`；没有登记的改动仍会被拒绝。原始快照在 `work/upstream/LRANet-PP`，`fetch_baseline.py` 始终对它执行 official 字节检查。原 evaluator 文件未改算法。

兼容变化包括 NumPy 已移除类型名、rapidfuzz 等价 API、版本声明、现代 C++ 头/API、sampler 构建包路径、CUDA stream 的 torch.device 参数。OpenCV contrib 需最后安装，提供作者训练标注生成所需 TPS 接口。checkpoint release mismatch 由单独固定策略处理，见 `checkpoints/README.md`。

## 运行限制

Compose 默认 CPU RAM 48 GiB、无额外 swap、12 核、1024 PID、8 GiB独立 shm，drop capabilities，非 root 用户，`--init` 回收子进程。训练/eval 由原有 resource_guard 管理，提前 RAM 阈值40 GiB、最长2小时。最初开发容器的 watchdog 自动按48 GiB硬上限将默认阈值钳到42 GiB；不要把默认配置中的128 GiB误作实际硬上限。

Compose 的 preparation 容器有网络以便下载。进入正式离线实验可使用 `docker run --network none` 和相同资源/挂载/环境（宿主仅调用 Docker CLI）；项目 `scripts/docker_run.py` 保留为可测试的命令生成器，但本轮不在宿主运行 Python。

GPU 默认 `CUDA_VISIBLE_DEVICES=0`。多卡时显式选择空闲 GPU，并令 `lranet_run.py --gpus N` 与可见卡数一致；GPU 显存无通用硬配额。不要按全局进程名字 kill；watchdog 只清理其拥有的进程组。

坏样本错误风暴的实际训练故障已触发 watchdog 并结束本组；单元测试另覆盖超时、SIGTERM、遗留 worker 等。真实超内存/全部故障注入演练尚未全部完成，不能写完整 G05 通过。torch/NCCL 的退出警告也不等同于已经实测所有清理场景。

## 来源证据

每次 run 固定 project/src/scripts/configs 快照，记录数据/初始化权重/配置哈希、seed、训练预算、pip freeze、扩展二进制哈希和资源轨迹。最终镜像 ID、构建日志及镜像内环境验证在初始化报告中汇总；首次开发容器的分数另标其直接调用环境，不能伪造镜像 ID。

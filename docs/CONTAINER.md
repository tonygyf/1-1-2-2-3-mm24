# 容器与进程资源约束

## 当前可用状态

本机Docker客户端存在，但daemon未运行；本轮没有构建镜像或运行容器。`docker/Dockerfile.reference` 是待服务器验证的历史环境候选，不能称为可直接支持RTX6000D的已验证镜像。实际GPU架构、驱动和主机RAM未知；83GB/卡不意味着CPU内存也是这个数。

该候选锁定作者README的Python3.7、torch1.8.1/cu111、mmcv-full1.3.9、mmdet2.15.1，并从本地已校验快照构建。运行时vendor指向镜像里的 `/opt/lranet`，因此自定义grid sampler的相对包导入能找到同目录编译产物；不要只用一个未编译的源码挂载覆盖它。

支持依赖不是作者完整lockfile：上游requirements未完整固定版本，其中tests要求的Levenshtein0.23.0不兼容README的Python3.7。本参考候选记录为0.12.2的运行时实现；rapidfuzz固定1.9.1以保留旧代码使用的string_metric，yapf固定0.40.1。服务器构建后必须保存pip freeze并跑matcher fixture。若有差异，先修环境，不把它隐瞒成完全相同的软件栈。

## 服务器准备

在具备NVIDIA Container Toolkit的Linux服务器操作。本轮不执行这些命令。

```bash
python scripts/fetch_baseline.py
docker build --progress=plain -f docker/Dockerfile.reference -t extremeocr:lranet-reference .
docker image inspect extremeocr:lranet-reference > work/reference-image-inspect.json
python scripts/docker_run.py --execute -- python scripts/preflight.py
```

构建只复制第三方源码与docker依赖说明，不复制数据、checkpoint或实验输出。构建默认 `MAX_JOBS=4`；源码中CUDA扩展保留原样。参考CUDA arch列表仅7.0/7.5/8.0/8.6，不能通过随意填写未来架构来让旧CUDA获得支持。

若实际GPU不支持该工具链，保留 reference image/tag 作为协议证据，在 `work/` 建单独移植源码/补丁和新镜像tag。先做到kernel前后向、原模型输出一致性、同一官方checkpoint的clean精度，再允许研究模块训练。当前没有预先声称某个现代torch/mmcv组合可用。

## 默认资源与可调整配置

`configs/resource_limits.json` 是默认值；复制它到 `work/server_limits.json` 按服务器实际资源修改，用 `docker_run.py --limits work/server_limits.json` 传入。整个有效配置都会传到watchdog，不仅仅是RAM阈值。

| 资源 | 默认值 | 实际行为 |
| --- | --- | --- |
| CPU RAM | 128 GiB | Docker cgroup硬上限；不是GPU显存上限 |
| 提前终止阈值 | 112 GiB | 每2秒查容器聚合内存，连续3次超限终止本次session |
| swap | 额外0 | `--memory-swap`等于`--memory` |
| CPU | 24核配额 | 避免占满共享服务器 |
| PID | 512 | 限制DataLoader/DDP进程膨胀 |
| shared memory | 16 GiB | 独立shm；不用host IPC，计入容器总内存预算 |
| 单实验最长 | 48小时 | 超时停止，不无限重启 |
| 无日志进展 | 30分钟 | 判断潜在卡死；首轮编译不在训练watchdog中 |
| 坏数据重采样 | 累计100条`data error` | 停止并要求修复数据链，防上游无限换样本 |
| 退出宽限 | 30秒 | 先SIGTERM，仍存活再SIGKILL本次session进程组 |

watchdog测的是容器cgroup内存，包含page cache等，不等同于Python对象RSS；更换阈值要据实际峰值和数据加载方式调整，不能为了绕过OOM直接去掉限制。硬上限可能先触发kernel OOM，故不保证每次都来得及优雅退出；Docker和watchdog记录用于复盘。

默认启动含 `--init --rm --restart=no --cap-drop=ALL --security-opt=no-new-privileges --network=none`。实验所需图像/权重/源码必须预先准备好。宿主项目目录挂载到 `/workspace`；无host PID、host IPC或privileged权限；不连接外部服务。

```bash
python scripts/docker_run.py --limits work/server_limits.json -- python scripts/preflight.py
```

不加 `--execute` 只打印命令。Linux实际运行用同一命令加 `--execute`。在容器内运行lranet_run仍须显式给 `--execute` 才启动模型：宿主层开关启动容器，内部开关启动实验。

## 自动清理的边界

watchdog使用 `start_new_session=True`，只给本次session进程组发信号，完成、失败或中断后都清理遗留活进程。`--init` 回收被托管的孤儿进程和zombie；真正zombie已结束执行，不能靠kill让它消失，需要父进程wait或init接管。

不会扫描整台机器按名字kill python，也不自动杀别人的GPU任务。Docker没有通用的每进程VRAM硬限额；本方案CPU RAM硬限制不能保证GPU不OOM。CUDA OOM/训练异常应以非零退出结束该组，其余worker由guard清理。未来只有在明确GPU进程归属映射后才增加显存watchdog，不能用全局nvidia-smi列表误杀。

容器故障演练属于G05，尚未在本机执行：包括内存阈值、walltime、SIGTERM、主进程退出留worker、坏数据风暴和正常退出。通过前不写“僵尸清理已实测”。

先在服务器运行 `python -m unittest discover -s tests -p test_resource_guard.py -v` 做不分配大内存的进程测试，再在受限容器中验证实际cgroup与init回收。unit测试用模拟memory计数触发清理，不能替代真实Docker硬上限测试。

# 实验 SOP：先确认链路，再确认方法

更新：2026-10-19。当前阶段为**代码与本地软件验收**，不代表 baseline 精度复现。先读 [研究方案](RESEARCH_PLAN.md)、[上游审计](BASELINE_ALIGNMENT.md) 和 [容器说明](CONTAINER.md)。所有命令从项目根目录执行。

## 0. 每次运行必须留下什么

每个实验独占 `work/runs/<run>`：`config.py`、`run.json`、`console.log`、`watchdog.json`、`checkpoint_keys_rank*.json`；有 adapter 的训练另有 `adapter_health_rank*.json`。训练权重在该 run 的 `training/`；评测预测在 `predictions.pkl`。不复用旧 run，不把 prepared 当 completed。

`run.json` 在执行前记录 seed、训练预算、上游版本、基底/配置/初始化权重 SHA256、每张输入图像和标注的 SHA256、环境与 Git 状态。收集分数前重新核对输入哈希。缺图、错 split、官方测试顺序改变、权重 key/shape 错误均拒绝运行。正式成果归档时还应保存代码 commit/工作区补丁、镜像 digest、`pip freeze`、扩展二进制哈希和原始数据来源许可；不要只保存分数 CSV。

研究checkpoint本身嵌入module、levels和源配置SHA256，加载时与sidecar核对，并拒绝非有限普通权重。请保留完整run和 `training/checkpoint_contract.json`；不要随手给不同条件模式的权重套同一份sidecar。

完整合格线和前置依赖在 `configs/quality_gates.json`。这些科研门槛需按证据逐项验收；当前运行器自动检查的是源码、split、资源、权重和模块健康，**不会根据文字门槛自动批准大训练**。

实际执行会在run中冻结 `project/src`、`project/scripts`、`project/configs`，worker从该快照加载项目代码，避免工作区后续编辑影响DDP不同rank。启动器固定本地镜像ID，worker记录 `pip-freeze.txt`、`extension_hashes.json`，watchdog写 `resource_trace.jsonl` 供内存增长复盘。轨迹增长本身不等于内存泄漏，需区别page cache、预取与活对象。

## 1. 本地只验软件

```bash
python -m pip install -e .
python -m pip install --target work/qa_deps --no-deps -r scripts/requirements.qa.txt
python scripts/fetch_baseline.py
python scripts/verify_local.py
```

QA 依赖只进入忽略目录，避免替换训练环境的旧依赖。普通数据工具需要 Python >=3.9；历史 Python3.7 容器通过 `PYTHONPATH` 使用兼容模块，不安装本项目 pyproject。验收命令调用官方 matcher 的软件 fixture，**不加载 OCR 模型、不训练**。若官方快照未获取，先运行 fetch；有源码修改则先检查，程序不替你覆盖。

## 2. 服务器环境门槛 G02/G03

按 CONTAINER.md 建参考镜像并记录实际硬件。旧 CUDA 未必支持新显卡；先跑 preflight，再显式跑 kernel/lifecycle gate，失败时先做独立移植镜像，不能跳到训练。

```bash
python scripts/docker_run.py --execute -- python scripts/preflight.py
python scripts/docker_run.py --execute -- python scripts/server_gate.py --checkpoint checkpoints/upstream/lranet_pp/totaltext.pth --output work/runs/server-lifecycle.json
```

`server_gate.py` 会构建真实模型、加载完整官方权重、比较 adapter 零初始化前后的 FPN 输出，并执行 DCNv2 和自定义 sampler 的 CUDA 前向/反向；它不替代真实图片推理、DDP 或全模型反向。运行时可再用 `resource_guard.py --run-dir work/runs/gate -- ...` 包住 gate；Docker 硬内存上限始终有效。

## 3. E000：只复现官方协议

先按 `data/README.md`、`checkpoints/README.md` 放入官方准备的完整 Total-Text JSON、图像及权重。不要随意重排 JSON。对应上游 GitHub commit 的全部文件与基底需保持一致。

```bash
python scripts/docker_run.py --execute -- python scripts/lranet_run.py eval --images data/raw/totaltext --annotations data/raw/totaltext/totaltext_test.json --official-annotations data/raw/totaltext/totaltext_test.json --checkpoint checkpoints/upstream/lranet_pp/totaltext.pth --protocol official-sweep --run E000-official --execute
```

这保留原模型和原 evaluator 扫阈值；本项目只加资源隔离、绝对资源路径及严格权重审计。实际 resize 为外层 `(1800,1000)` 的 keep-ratio 限制，不能按内层 `(1280,800)` 解释。

论文目标见 `configs/paper_targets.json`，存在两种外部训练数据版本；下载权重后先核对它属于哪个版本。标准版论文目标为检测 88.1%、无词典 e2e 82.2%，扩展训练版为 90.8%/84.6%。均为**论文引用值**，本项目尚无实测值；对应 profile 未核清前不判“复现通过”。两项绝对差目标不超过 0.5 个百分点，超差先查权重、图像顺序、resize、GT 和运行环境。固定阈值表不与这个扫阈值上限混用。

## 4. 在原图级划分 train/dev

```bash
python scripts/split_training.py --images data/raw/totaltext/training --annotations data/raw/totaltext/totaltext_train.json --output data/derived/totaltext_split --seed 42
```

默认按独立源图分组，约 15% 的组为 dev；相同字节的重复图强制同组。真实照片或连续帧应提供 `--groups <json>`，JSON 是文件名到地点/对象/拍摄序列组的映射。工具不能自动发现不同编码的近重复或同场景，必须人工审阅分组。

**初始化暴露要单独记录**：官方 Total-Text 已微调 checkpoint 通常已见过原训练集，包括后划出的 clean dev 源图。它支持公平的二次适配对照，但这不是“从未见过源图”的 dev。若要严格隔离初始训练，用官方 Synth150K/MLT pretrain 权重给所有研究对照相同初始化，并另跑 train-only clean baseline；不要混用两种初始化后直接解释模块收益。用 `--initialization-note` 写清本次选择。最终 test/真实 heldout 始终不参与训练和选模块。

## 5. 分别构建 train 与 dev 退化

```bash
extremeocr build --images data/raw/totaltext/training --annotations data/derived/totaltext_split/totaltext_train.json --output data/derived/totaltext_train --config configs/screen.json
extremeocr build --images data/raw/totaltext/training --annotations data/derived/totaltext_split/totaltext_dev.json --output data/derived/totaltext_dev --config configs/screen.json
python scripts/mix_training.py --suite data/derived/totaltext_train --annotation-name totaltext_train.json --variants clean glare/s1 glare/s2 glare/s3 --output data/derived/totaltext_train/totaltext_mixed.json
```

所有变体包含 clean 都用 PNG，保持实例标注、顺序与 split 元数据，manifest 记录像素/标注/配置哈希。先在 dev 比较 glare/exposure/bloom 及 blur/rain/snow；不要在测试集挑退化档位或阈值。初始固定置信阈值 0.3；若预先登记 clean-dev 选择规则，只能执行一次，之后所有方法/退化冻结。

clean 加三档 glare 的混合样本比为 1:1:1:1，即 clean 占 25%，不是 50%。所有模块用相同混合标注，避免把采样比例误当模块收益。

## 6. 探针：先验证观测可信

零样本 dev 评测同一个初始化，保存 pre/post recognition confidence：

```bash
python scripts/docker_run.py --execute -- python scripts/lranet_run.py eval --split dev --images data/derived/totaltext_dev/glare/s2/images --annotations data/derived/totaltext_dev/glare/s2/totaltext_dev.json --checkpoint checkpoints/upstream/lranet_pp/totaltext.pth --score-probe --run E001-dev-glare-s2 --execute
```

dev 使用 `predict_dev.py` **显式关闭上游默认的评测开关**，然后用独立 GT zip 和顺序映射调用原 matcher；不触碰官方 GT 包。项目 dev 的序列化按原整数坐标、0.1 预过滤、三位分数舍入、阈值及 winding 规则处理。Windows 下原官方序列化存在路径分隔符假设，项目 dev 使用可移植序列化；服务器仍须在同一组预测上对照原序列化，确认数值一致。

`score_probe_summary.json` 给出识别置信乘积导致多少候选跨过阈值，**不是漏检精度**。官方 detection/e2e care 集合不同；不从两种 Hmean 的差推导识别错误率。

```bash
python scripts/probe_instances.py --images data/derived/totaltext_dev/glare/s2/images --annotations data/derived/totaltext_dev/glare/s2/totaltext_dev.json --output work/runs/probe-crops-glare-s2
python scripts/score_crops.py --instances work/runs/probe-crops-glare-s2/instances.jsonl --predictions work/runs/probe-crops-glare-s2/recognizer_predictions.jsonl
```

第一条导出每个词 polygon 内饱和率、原图身份和轴对齐 crop；第二条输入是另一个**固定识别器真实输出**的 JSONL，每行 `{image_id, annotation_id, text}`，缺预测或重号报错。当前没有自动安装/运行 OpenOCR，也没有实现 LRANet 的 GT-grid oracle；矩形 crop 不等同于曲线校正、更不能单独证明采样失配。clean/degraded 使用同一识别器、同一 crop 身份；人工额外标记可读性，只作分层报告，不改最终官方 GT。

## 7. 30 步 probe：单卡再四卡

```bash
python scripts/docker_run.py --gpus all --execute -- python scripts/lranet_run.py train --images data/derived/totaltext_train --annotations data/derived/totaltext_train/totaltext_mixed.json --checkpoint checkpoints/upstream/lranet_pp/totaltext.pth --module saturation --stage probe --gpus 1 --run P001-adapter-1gpu --execute
python scripts/docker_run.py --execute -- python scripts/lranet_run.py train --images data/derived/totaltext_train --annotations data/derived/totaltext_train/totaltext_mixed.json --checkpoint checkpoints/upstream/lranet_pp/totaltext.pth --module saturation --stage probe --gpus 4 --run P002-adapter-4gpu --execute
```

`--gpus` 在 docker_run 是可见设备选择，在 lranet_run 是 DDP worker 数。训练默认 probe，只有显式 `--stage full` 才是大训练。检查各 rank 梯度有限、非零更新、loss、吞吐、内存，随后加载保存的 `training/iter_30.pth` 做 dev 推理。30 步不要求 Hmean 提升。

研究训练统一 `--no-validate`，所以继承配置中的 test-val 不会运行。不会自动拿测试集选 checkpoint。上游 EMA hook 保留；推理保留 checkpoint 原普通参数键，与上游 test 口径一致，仅移除名称映射和 shape 都核验过的 `ema_*` 备份。不能把普通末次参数随意称为“末次 EMA”。

## 8. 500 步 pilot：只在 dev 排序

同样命令改为 `--stage pilot`，分别跑下表五个 module，seed42、相同初始化、batch=4/卡、4卡、AdamW lr2e-5、weight_decay1e-4、1280训练画布，500步时 LR 在250步衰减。正式10000步在5000步衰减。两种预算不能混在同一公平比较表。

| 方法 | `--module` | 用途 |
| --- | --- | --- |
| 普通混合微调 | none | 必须赢过的数据增强基线 |
| 容量匹配残差 | residual | 两通道条件恒零，网络结构/参数量完全相同 |
| 饱和条件 adapter | saturation | 任意通道/全通道近饱和的空间统计 |
| 全局饱和条件 | global | 空间均值，检验是否真需要位置 |
| 翻转饱和条件 | flipped | 固定翻转位置，检验条件错位 |

`--levels 0 1 2 3` 为全部四级；可消融 `--levels 0`、`--levels 1 2 3`。不启用的层保留参数但不走残差，报告总参数与实际训练参数的区别。P2仅识别路径使用，P3/P4由检测识别共享，P5主要检测使用；不能把“按层”消融直接叫“按任务解耦”。

评测已训练权重必须使用相同 module/levels，并保留权重同目录的 `checkpoint_contract.json`：

```bash
python scripts/docker_run.py --execute -- python scripts/lranet_run.py eval --split dev --images data/derived/totaltext_dev/clean/images --annotations data/derived/totaltext_dev/clean/totaltext_dev.json --checkpoint work/runs/E004-pilot/training/iter_500.pth --module saturation --run E004-pilot-dev-clean --execute
python scripts/inspect_run.py --run-dir work/runs/E004-pilot-dev-clean
python scripts/collect_metrics.py --run-dir work/runs/E004-pilot-dev-clean --output work/runs/dev_metrics.csv
```

路径示例要求先用对应 run 名完成训练。对 clean 与主退化三档分别评末次 checkpoint；容量/位置对照不胜，优先查归因，不堆新模块。

## 9. 正式训练、冻结、最终测试

pilot 通过后冻结 configs、数据、初始化、阈值、模块和选择规则。主要三组 none/residual/saturation 用 `--stage full` 各做 seed42/43/44，固定10000步末次 checkpoint；global/flipped和按层消融至少seed42，若贡献核心依赖这些差异则补三seed。另补 clean-only 微调、通用增强微调、曝光校正前处理对照；这些是独立数据/前处理分支，当前不冒充已经实现的一键方法。

大训练启动的预登记目标：dev 三档 glare 平均 e2e 比普通微调和容量对照均提高至少1.0个百分点，clean不掉超过0.5点，最严重档不倒退超过0.5点。不是保证收益；不满足就停止该模块大训练/撤回相应主张，保存负结果。不要改测试阈值来过线。

冻结后才构建完整 official test 退化集，执行 `--split test --official-annotations <原始300图JSON>` 的固定阈值评测；真实heldout需单独适配matcher/GT，不把它伪装成Total-Text。尚未接 CTW1500/第二架构运行器；完成对应GT、图像顺序、词典和checkpoint门槛后再扩展。

最终报告每档 detection/e2e、平均/最差档、绝对掉点、保留率、每seed分数、均值/标准差、训练失败和完整失败图。图像为单位的配对bootstrap需重算聚合TP/FP/FN，不能平均每图Hmean代替数据集Hmean；当前统计工具未自动实现该bootstrap。最终测试失败用于如实报告，不回流选模块。若因错误必须重跑，登记具体bug、影响范围和新的协议版本。

## 10. 故障演练与复盘

在容器内专门的演练 run 验证：超内存、超时、SIGTERM、父进程提前退出遗留 worker、坏样本重采样。watchdog超限只杀自己创建的session进程组；Docker `--init` 回收孤儿僵尸。真正的 zombie 不能靠SIGKILL消除，需要父进程wait/init回收。

每次失败填 [复盘模板](RUN_REVIEW_TEMPLATE.md)：触发条件、退出码、最后loss、资源峰值、哪一级失败、修复内容、重跑范围。软件测试、环境通过、baseline复现、方法有效和论文创新分别记状态。首项目的SOP也应保留失败路径。

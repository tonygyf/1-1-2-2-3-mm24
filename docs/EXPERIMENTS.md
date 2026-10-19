# 实验手册

## 软件与计算环境

数据生成工具使用 Python >=3.9、NumPy、Pillow。`adapter.py` 和 LRANet 插件通过 PYTHONPATH 加载，尽量保持旧 PyTorch API 兼容；完整模型集成尚未实测。

上游公开环境为 Python 3.7 / PyTorch 1.8.1 / CUDA 11.1 / mmcv-full 1.3.9 / mmdet 2.15.1，并有 DCNv2 和自定义 grid sampler。这是作者测试环境的记录，**不能当作新显卡服务器上已经验证的环境**。先在服务器运行 `scripts/preflight.py`，按实际 GPU 架构选择可运行的 PyTorch/CUDA，再编译扩展。现代环境移植应单独记录补丁和结果，别与原论文复现混为一谈。

公开安装步骤和依赖详见 [上游 README](https://github.com/ychensu/LRANet-PP#environment)。本机仅安装了 CPU 版 torch 2.2.2，因此本轮完成的是软件 smoke。

## 1. 准备 baseline 与数据

```bash
python scripts/fetch_baseline.py
python scripts/preflight.py
```

按 data/README.md 和 checkpoints/README.md 放置正式资源。检查 JSON 中 `file_name` 能由图像根目录正确解析。复用作者提供的 `gt_totaltext.zip` 评测包，当前源码快照已包含。

上游训练/验证配置存在相对路径，所以在 Linux 上将 baseline 的 `data` 链接到本项目 `data/raw`：

```bash
cd work/upstream/LRANet-PP
ln -s ../../../data/raw data
cd ../../..
```

已有同名目录时先检查，不覆盖。生成工具和完整 baseline 环境可分开使用。

## 2. E000：复现原论文协议

在 baseline 环境中，从 upstream 工作目录运行作者原命令；该命令会扫描多个置信阈值，返回最优检测与 e2e 分数，后续鲁棒性表不使用这一扫描协议。

```bash
cd work/upstream/LRANet-PP
python tools/test.py configs/lranet_pp/lranet_pp_totaltext.py ../../../checkpoints/upstream/lranet_pp/totaltext.pth --eval hmean-e2e
cd ../../..
```

记录环境、权重 SHA256、数据版本、resize、词典设置与数字差异。不把官网表格抄到本地“实测”列。

## 3. E001：固定协议诊断

```bash
extremeocr build --images data/raw/totaltext --annotations data/raw/totaltext/totaltext_test.json --output data/derived/totaltext_test --config configs/screen.json
python scripts/lranet_run.py eval --images data/derived/totaltext_test/clean/images --annotations data/derived/totaltext_test/clean/totaltext_test.json --checkpoint checkpoints/upstream/lranet_pp/totaltext.pth --run E001-clean --score-threshold 0.3 --execute
python scripts/lranet_run.py eval --images data/derived/totaltext_test/glare/s2/images --annotations data/derived/totaltext_test/glare/s2/totaltext_test.json --checkpoint checkpoints/upstream/lranet_pp/totaltext.pth --run E001-glare-s2 --score-threshold 0.3 --execute
python scripts/collect_metrics.py --run-dir work/runs/E001-clean --output work/runs/metrics.csv
```

不加 `--execute` 时只生成配置和命令。更换 run 名再实际运行，旧 run 不覆盖。运行器当前仅支持 Total-Text 官方整理 JSON 的 full split；文件名包含 `totaltext` 以触发上游正确 evaluator。IC15/CTW 与真实数据评测须另接，尚未实现。

阈值初值 0.3；可以只在 clean validation 上选一次，再冻结用于所有退化/方法。上游还含 score >0.1 的预过滤，不能把阈值设得更低。`lranet_protocol.py` 在内存中将上游阈值循环改为单值，保留其匹配与词过滤规则，不改外部源码。此集成尚待完整环境核验。

所有变体均输出 PNG，含 clean，因此不引入 JPEG 编码差异。改变文件后缀时仅同步更新 `images.file_name`；实例 polygon、bbox、transcription 与次序保持不变。光斑位置由文件名和 seed 决定，严重度变化不换光斑中心。manifest 保存每张图生成参数、seed、前后哈希和饱和率。

## 4. E002–E004：等预算微调

先在训练 split 生成独立 suite，不能拿 test suite 训练。用 `mix_training.py` 将 clean 与训练退化合并；生成混合标注只引用已有 PNG，不重复复制图像。

```bash
extremeocr build --images data/raw/totaltext --annotations data/raw/totaltext/totaltext_train.json --output data/derived/totaltext_train --config configs/screen.json
python scripts/mix_training.py --suite data/derived/totaltext_train --annotation-name totaltext_train.json --variants clean glare/s1 glare/s2 glare/s3 --output data/derived/totaltext_train/totaltext_mixed.json
python scripts/lranet_run.py train --images data/derived/totaltext_train --annotations data/derived/totaltext_train/totaltext_mixed.json --checkpoint checkpoints/upstream/lranet_pp/totaltext.pth --run E002-seed42 --gpus 4 --execute
python scripts/lranet_run.py train --images data/derived/totaltext_train --annotations data/derived/totaltext_train/totaltext_mixed.json --checkpoint checkpoints/upstream/lranet_pp/totaltext.pth --run E004-seed42 --gpus 4 --adapter --execute
```

相同初始化、数据、迭代、学习率、resize、batch、seed。baseline 默认训练 10k iter、每卡 batch 4、AdamW lr 2e-5；第一轮可统一缩短到 2k iter 以快速筛选，正式实验再冻结预算并补 3 seeds。验证仍指向 clean 官方数据，用同一验证选择规则，禁止在最终真实测试集挑 checkpoint。

目前 adapter 接在 4 个 FPN 输出后，两路连续近饱和统计分别反映任意通道与全通道剪裁；残差末层零初始化。模型注册与完整训练未验证。E003 无条件残差容量对照待实现，别用“无模块”替代“相同参数量对照”。

## 5. 论文证据表

报告 clean 和每个退化档的 detection/e2e Hmean，跨严重度均值与最差档，绝对掉点及性能保留率。识别诊断增加 GT crop recognizer accuracy / CER，避免直接用两个任务 Hmean 的差推断“识别掉得更多”。GT crop 与真实集转录评测入口待接。

最小对照：零样本、普通混合微调、通用增强微调、容量匹配残差、adapter、曝光校正+原模型。若真实测试与两个数据集上的收益稳定，再加第二架构/机制辅助损失。TextDS 待代码发布后才做同检测协议对照，不能编造复现分数。

发布用结果必须来自日志、完整配置和固定数据；`collect_metrics.py` 只接受已完成 eval 的真实输出。软件 smoke 图和单元测试数字不进入论文准确率表。

# ExtremeOCR

极端成像条件下的场景文本检测与识别研究项目。目标：形成适合 CCF-B 投稿的完整实证工作。

当前主线：**局部高光/饱和下的端到端 text spotting**。主 baseline 为 LRANet++（2025 arXiv，官方仓库现标注 TPAMI），第二研究入口为 SemiETS（CVPR 2025）。训练资源：用户提供的 4 × RTX 6000D，单卡约 83GB；不以轻量化作为选题限制。

当前已完成官方560文件字节对齐、模块化FPN对照、独立dev评测和有资源限制的实验入口。尚未下载正式数据及权重，尚无真实OCR指标；Docker镜像和服务器GPU环境待验证。

## 开始

```powershell
python -m pip install -e .
python -m pip install --target work/qa_deps --no-deps -r scripts/requirements.qa.txt
python scripts/fetch_baseline.py
python scripts/verify_local.py
python scripts/preflight.py
```

`smoke.py` 不需要 GPU；生成六类退化，实际运行数据集构建、检查标注继承及适配器前向/反向，输出预览和状态报告。NumPy/Pillow 由基础安装提供；adapter 检查还需要 PyTorch（已有训练环境可复用）。确定性等协议检查由 unittest 覆盖。

正式数据到位后：

```powershell
extremeocr build --images data/raw/totaltext --annotations data/raw/totaltext/totaltext_test.json --output data/derived/totaltext_test --config configs/screen.json
```

先读 [本轮研究方案](docs/RESEARCH_PLAN.md)、[baseline对齐审计](docs/BASELINE_ALIGNMENT.md)、[容器限制](docs/CONTAINER.md)，然后按 [实验SOP](docs/EXPERIMENTS.md) 执行。每个模块合格线在 `configs/quality_gates.json`，各run用 `scripts/inspect_run.py` 检查实际跑通的链路。选题来源见 [项目简报](outputs/PROJECT_BRIEF.md)。

## 目录约定

| 位置 | 唯一职责 |
| --- | --- |
| `.agents/skills/` | 项目级 skills，来源版本见 `configs/upstream.lock.json` |
| `configs/` | 退化协议、实验矩阵、上游版本锁 |
| `src/extremeocr/` | 退化生成、数据协议、模型插件 |
| `scripts/` | 获取 baseline、运行实验、采集结果、环境检查 |
| `tests/` | 防止实验协议和指标处理错误的必要检查 |
| `docs/` | 实验操作及持续推进清单 |
| `data/` | 原始/派生数据，占位与来源说明；实际数据不进 Git |
| `checkpoints/` | 官方/实验权重，占位与来源说明；权重不进 Git |
| `work/` | 外部源码、运行日志、预测和中间产物；不进 Git |
| `outputs/` | 可交付简报、图表、最终表格 |

不另建 `results`、`runs`、`assets`、`datasets`、`weights` 等同义顶层目录。实验运行统一放 `work/runs/<experiment>`，成果统一进入 `outputs/`。

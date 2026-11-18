# 初始化权重

2026-10-22 已在 Docker 内下载 [LRANet++ 官方 Total-Text 权重](https://github.com/ychensu/LRANet-PP#trained-model)。

- 原始文件：`checkpoints/upstream/lranet_pp/totaltext.pth`，作者名 `totaltext_final.pth`；SHA256 `520c3285d46fc77cf43bfe8f7f2854254e691e5c28265fd911939f101e8d624e`。
- 运行文件：`totaltext_runtime.pth`。`scripts/prepare_checkpoint.py` 保留 550 个匹配 tensor，不修改其数值；补入当前代码按定义生成的 8 个 TPS 常量，略去当前代码不用的 24 个旧键，匹配官方非严格加载行为。原文件不改。策略只覆盖固定原始 SHA，来源与运行文件 SHA 在 `.source.json` 中。
- 原始 release 只有 state_dict，没有训练元数据；标准版/扩展训练版 profile 未核清。现有实测分数不能反推训练数据并据此宣布复现通过。
- 所有研究对照须从同一初始化开始并记录 dev 源图暴露。Synth/MLT pretrain 尚未下载；严格训练来源隔离仍需它或其他明确的初始化。

容器内运行 `python scripts/download_assets.py --only checkpoint`，然后 `python scripts/prepare_checkpoint.py`。运行权重不自动覆盖。实际权重、训练临时 checkpoint 不进 Git；后者在各 `work/runs/<run>/training/`。原模型/数据使用条款适用。

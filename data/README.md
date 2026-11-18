# 数据

2026-10-22 已在 Docker 中下载 LRANet++ 作者整理的 Total-Text，完成 ZIP CRC、图像解码、尺寸、1255/300 数量及 SHA256 校验。资源入口与固定哈希为 `configs/asset_sources.json`；本地来源说明/逐图哈希为 `data/raw/totaltext/SOURCE.md`、`manifest.json`，下载收据伴随各标注和 archive。

```text
data/raw/totaltext/totaltext_train.json
data/raw/totaltext/totaltext_test.json
data/raw/totaltext/training/...  # train JSON file_name 不含 training/
data/raw/totaltext/test/...      # test JSON file_name 含 test/
data/derived/totaltext_split/    # seed42: train1067/dev188
data/derived/totaltext_train/    # clean + glare/s1,s2,s3；PNG 1:1:1:1
data/derived/totaltext_dev/      # 188×4，固定阈值诊断；不生成测试腐蚀集
```

在容器内运行 `python scripts/download_assets.py --only data`。校验收据匹配时复用；已完成 split/suite 不自动覆盖。场景/近重复分组尚需人工复查；官方微调权重可能见过 dev 源图，不能把它当作训练来源从未见过的独立泛化集。

来源：[LRANet++ 官方说明](https://github.com/ychensu/LRANet-PP#dataset)、[原 Total-Text](https://github.com/cs-chan/Total-Text-Dataset)。保留作者标注和图像顺序。原始数据/分发条款适用，未声称新的分发授权；实际数据不进 Git。

真实强光采集、可读性审核、场景分组、第二数据集仍待完成。饱和后人也无法辨认的实例需另记 unreadable；合成 glare 掉点不直接构成真实场景结论。

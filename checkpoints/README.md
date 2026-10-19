# 权重放置

目前没有下载任何正式 OCR 权重。

后续按需创建：

```text
checkpoints/
  upstream/lranet_pp/totaltext.pth
  upstream/lranet_pp/pretrain.pth
  experiments/E004-seed42/best.pth
```

- 从 [LRANet++ 官方 Trained Model](https://github.com/ychensu/LRANet-PP#trained-model) 获取 Total-Text 已训练权重，优先直接做零样本诊断；不从大规模预训练开始烧时间。
- 普通微调、模块、容量对照从同一权重起步；预训练数据保持一致。
- 每个权重旁放 `SOURCE.md`：原始 URL、SHA256、训练数据、版本、用途、许可/使用限制。下载页能打开不代表文件已可用。
- 最终论文用到的权重才从 `work/runs` 晋升到这里；临时 checkpoint 留在对应 run 中。

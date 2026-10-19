# 数据放置

实际图像与标注尚未下载。只在这里记录资源位置，不创建空数据集伪装成可用数据。

后续按需创建：

```text
data/
  raw/totaltext/totaltext_train.json
  raw/totaltext/totaltext_test.json
  raw/totaltext/training/...
  raw/totaltext/test/...
  raw/real_highlight/...            # 真实测试图，单独分组
  derived/totaltext_test/...        # CLI 自动生成，不手动编辑
  derived/totaltext_train/...
```

1. 第一阶段使用 LRANet++ 作者整理的 Total-Text 图像与 JSON，保持原始 `images` 顺序、file_name 路径及 train/test 划分。[官方获取说明](https://github.com/ychensu/LRANet-PP#dataset)。作者的测试 `img_prefix` 是 totaltext 根目录，file_name 可能包含 `test/`，以 JSON 内容为准。
2. [Total-Text 官方项目](https://github.com/cs-chan/Total-Text-Dataset)用于核对原始数据/协议。其他格式的 COCO 标注不一定兼容作者评测顺序。
3. ICDAR2015 作为第二验证集候选；当前运行器专门接 Total-Text，不把不兼容的 IC15 评测说成已实现。
4. 真实强光图计划先 50 张试采集，最终 200–500 张。保留原始拍摄记录/许可、设备、场景组、光照标签和转录。按地点/视频/标牌分组划分；同一对象不同曝光不得跨训练和测试。
5. 饱和后人工也无法辨认的文字另标 `unreadable`。主测试保留可读文字，完整压力测试单列。不要把完全被覆盖的文字直接当可读 GT 评价。
6. [Robust-Text](https://proceedings.mlr.press/v189/cao23a.html)包含真实复杂环境，但以行级检测为主，不直接当作已有 spotting 转录数据。先核对能否下载及标注。

数据下载后在 `data/raw/<name>/SOURCE.md` 记录：来源链接、下载时间、许可、版本、数量、标注哈希、划分及已知重叠。服务器可将 data 指向已有共享盘，不复制多份。

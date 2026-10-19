# 工作区

外部源码统一在 `work/upstream`，完整实验记录在 `work/runs/<id>`，软件 smoke 中间文件在 `work/smoke`。

LRANet++ 当前快照来自官方固定 commit，使用 ZIP 获取而非嵌套 Git，具体版本在 `configs/upstream.lock.json`。可通过 `python scripts/fetch_baseline.py` 重建。

这里不是成果存储位置。目录内容默认不进 Git。根目录重整前仅有空 work、outputs，没有需要搬迁或删除的旧项目文件。

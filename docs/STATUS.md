# 状态与下一步

更新：2026-10-19，Asia/Shanghai。

## 已做

- 确认 CCF-B 定位与用户服务器资源。
- 初步核实 LRANet++、SemiETS、SwinTextSpotter v2、TextDS 与已有鲁棒性工作。
- 下载 LRANet++ 固定源码快照；检查真实配置、模型插入点及评测器。
- 安装项目级 grill-me、依赖 grilling、autoresearch，锁定来源 commit。
- 搭好退化协议、数据清单、adapter 与运行/结果采集入口。
- 软件 MVP 已具备数据构建、标注继承和 adapter 前向/反向的验证入口；结果见 outputs/smoke_status.json。
- 5 项单元检查通过，19 个数据变体构建通过，Python wheel 构建通过；Git main 仓库已初始化。
- Codex provider 选项位置与 HTTP/代理配置已修复，新启动进程实际请求验证完成，详情见 outputs/CODEX_CONNECTION.md。

## 尚未做

- 未连接训练服务器；未验证服务器 GPU 上的 DCN / 自定义 grid sampler 编译。
- 未下载正式数据及 checkpoint；未完成 baseline 复现或掉点测试。
- adapter 只属于待实验实现；MMOCR 注册和全模型训练需要在真实依赖环境验证。
- 近期基金仅找到相关官方项目样本，未获得申请书正文。不能据此复述所谓获批创新点。
- 论文检索属于定向首轮，不能宣称穷尽或保证“首次”。

## 后续范围

按用户最新要求，本轮停在最小 MVP 与待办清单，不启动服务器训练。完整待办见 outputs/TODOLIST.md。

## 下一轮的一个主要交付

在服务器用官方 Total-Text checkpoint 完成 clean + glare/exposure/bloom 三档固定阈值推理，得到每档 detection/e2e Hmean、原始预测和失败图；同批次保留 blur/rain/snow 快速筛查。首轮不训练新模型。

之后：用相同预算跑普通退化微调，再跑 adapter 与无条件残差容量对照。真实图采样同时进行，不让数据收集阻塞零样本诊断。

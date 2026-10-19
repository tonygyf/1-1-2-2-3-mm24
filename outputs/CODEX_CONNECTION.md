# Codex 连接配置修复记录

日期：2026-10-19；客户端：Codex Desktop；本地 CLI：0.163.0-alpha.10.2。

## 原因与实际修改

用户配置文件为 `C:\Users\15634\.codex\config.toml`。原本的 `model_provider = "chatgpt-http"` 位于 `[model_providers.chatgpt-http]` 内，TOML 将它解析成子表字段，顶层 provider 没有被选中。修复后它位于所有表之前。模型及推理强度保持原值。

本机历史日志记录了 Responses WebSocket 连续超时重试到 5/5，后续还出现 SSE idle timeout 与 response body decoding error。因此截图中的重连计数不能一概等同于“正在从 WebSocket 回退 HTTP”；HTTP 流自身也可能中断。

现在的连接相关配置：

```toml
model_provider = "chatgpt-http"

[model_providers.chatgpt-http]
name = "ChatGPT HTTP"
base_url = "https://chatgpt.com/backend-api/codex"
wire_api = "responses"
requires_openai_auth = true
supports_websockets = false
request_max_retries = 2
stream_max_retries = 1
stream_idle_timeout_ms = 120000

[features]
respect_system_proxy = true
```

这是相关字段的摘录，不要用它覆盖整个配置文件。完整配置已在修改前备份到项目 `work/config-backups/`，其他原有设置保留。

`supports_websockets=false` 让新请求直接走 HTTP/SSE；流中断重试改成 1 次。120 秒是**没有流事件时的空闲超时**，不是整轮运行上限；它缩短坏连接等待，也可能让长时间无事件的推理更早报错。

Windows 当前系统代理为 `127.0.0.1:10808`，该端口有 xray 进程监听。Codex 原先 `respect_system_proxy` 为 false；现已启用，让新进程使用系统代理。此开关是当前本机版本支持的开发中功能，CLI 会提示这一状态，提示没有被隐藏。没有修改 Windows 全局代理、认证文件或代理节点。

## 已验证

- TOML 解析通过，顶层 provider 为 chatgpt-http，子表中不存在误放的 model_provider。
- `codex --strict-config doctor` 加载配置成功。
- Doctor 确认活动 provider 不启用 Responses WebSocket，HTTP inference/route 探测成功。
- 同配置加 `respect_system_proxy` 的 Doctor 检查通过，显示系统代理已启用。
- 修正 provider 后一次真实、无工具模型请求成功返回 `HTTP_OK`，耗时 21.42 秒。
- 启用系统代理后第二次真实请求成功返回 `HTTP_OK`，耗时 17.31 秒，exit code 0，收到 turn.completed。两次不是严格测速实验，不据此承诺固定提速比例。

底层原始验证记录保存在 `work/http-smoke-result.json`、`work/http-proxy-smoke-result.json` 与 Doctor JSON。没有把账户令牌写入报告。

当前已经运行的桌面进程可能保留旧配置。**结束当前工作后重启 Codex**，确保后续桌面会话加载新设置；本轮没有强制结束正在工作的应用。已验证的是新启动进程的配置和请求，不是尚未重启的桌面进程。

这消除了已发现的 provider 选项错误与 WebSocket 试错等待；网络或上游服务仍可能偶发断流，不能保证永久零重连。

## 官方依据

- [配置参考](https://learn.chatgpt.com/docs/config-file/config-reference)：provider 选择、WebSocket 支持、流重试与空闲超时字段；默认流重试为 5 次、空闲超时 300000ms。
- [网关配置说明](https://learn.chatgpt.com/docs/enterprise/connect-to-a-gateway)：model_provider 等顶层字段应放在第一个 TOML 表之前。
- [HTTP/SSE 传输说明](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations)：supports_websockets=false 选择 HTTP。

系统代理开关的依据为当前安装版 `codex features list` 及实际 Doctor/请求测试，官方配置网页本轮未检索到该开关的说明。

# WordCovenant

> 凡口头所言，皆立为契约，有据可查，事事落单。

WordCovenant 是一款 macOS 优先、本地优先的桌面工作台，用于可见的对话采集、带时间戳的文字记录、匿名说话人簇，以及由用户主动触发的 Agent 动作。

## 隐私约定

- 音频、转写文本、说话人数据和上下文默认留在设备本地。
- 默认拒绝网络出站。只有用户可见地启用仅本次会话有效的出网总开关，并分别批准具名工具、其 HTTPS 源站和数据范围后，才允许出网。
- 转写文本、模型、技能、钩子或外部响应本身都不能授予权限。
- Agent 计划是强类型数据，而不是 shell 命令，不能绕过 Rust 的策略检查。
- 当前版本不会通过声纹自动识别真实身份，也不宣称具备法律上的不可否认性。

## 当前 M0 能力

- WordCovenant 的 macOS 应用品牌、受限的 Tauri capability 集合，以及不包含通配网络源的 CSP。
- 本地录音工作台，始终可见的本地优先/录音状态，以及按时间顺序排列的转写时间线。
- 用于采集会话、可修订转写片段、匿名说话人簇、计划与工具调用的 Rust 领域契约。
- 默认拒绝的 HTTPS 出站策略，包含可见的仅本次会话总开关，以及精确源站、数据范围、到期和撤销检查。
- 通过 SQLite 在本地持久化的仅追加 SHA-256 审计链。
- 有界音频采集契约、采样时钟时间换算、明确的音频间隙类型，以及可重复的测试采集源。
- 用于会话生命周期、时间线查询、出网批准与撤销、本地动作提议和网络策略预检的强类型 Tauri 命令。

为了能在浏览器中直接查看，M0 UI 会在 Tauri 之外使用本地合成数据。界面会明确标注为 `synthetic`；这不是麦克风或 ASR 的降级替代方案。

## 后续里程碑

M0 实现计划见 [docs/plans/2026-08-07-word-covenant-local-first-mvp.md](docs/plans/2026-08-07-word-covenant-local-first-mvp.md)。完整的产品路线图、里程碑、验收门槛、风险和架构决策见 [docs/plans/2026-08-07-word-covenant-roadmap.md](docs/plans/2026-08-07-word-covenant-roadmap.md)。

1. M0.1 新增可见的仅本次会话出网总开关。即使配置已获批准，在用户确认该总开关前仍会被拦截；应用重启后开关恢复关闭。
2. M1 新增 CoreAudio 采集、真实会话时钟，以及可见的音量与音频间隙事件。
3. M2 新增显式导入的本地 VAD 与 ASR 适配器。模型绝不会自动下载。
4. M3 新增匿名说话人聚类，以及由用户控制的重命名和合并修正。
5. M4 新增用户触发的 Agent 规划、本地 TTS、批准界面和具名的出站 HTTP 配置。
6. M5 新增声明式技能，随后引入经过签名且受约束的 WASM 钩子。
7. M6 新增由 Keychain 支持的加密、保留控制、导出、公证和发布流程。

## 开发

前置条件：较新的 Rust 工具链、pnpm、Xcode 命令行工具，以及常规的 Tauri macOS 开发环境。

```sh
pnpm install
pnpm test --run
pnpm type-check
pnpm check
pnpm tauri dev
```

可在浏览器中单独运行预览，地址为 `http://127.0.0.1:1420`：

```sh
pnpm vite:dev
```

## 验证

```sh
pnpm test --run
pnpm type-check
pnpm build
cargo test --manifest-path src-tauri/Cargo.toml
cargo check --manifest-path src-tauri/Cargo.toml
```

测试覆盖默认拒绝出网、授权匹配、过期、撤销、转写时间戳边界、审计链篡改检测、本地采集队列行为，以及可见的本地优先 UI 状态。

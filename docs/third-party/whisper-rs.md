# whisper-rs / whisper.cpp

- Rust crate: `whisper-rs` `0.16.0`
- Rust FFI crate: `whisper-rs-sys` `0.15.0`
- Source: <https://codeberg.org/tazz4843/whisper-rs>
- Upstream runtime: <https://github.com/ggml-org/whisper.cpp>
- `whisper-rs` and `whisper-rs-sys` license: Unlicense
- Bundled whisper.cpp / ggml license: MIT
- macOS build feature: `metal`

## 包内默认模型与兼容性

macOS 发布包内置经审查的多语种 Whisper `large-v3-turbo` Q5_0 模型
`ggml-large-v3-turbo-q5_0.bin`，并将其作为首次打开时的
默认本地 ASR。正常录音无需用户下载、导入，或填写 SHA-256、模型卡和许可证元数据。
这些权重是多语种的，但当前产品将默认解码语言固定为中文，并使用中文主导的中英混合提示；
不启用自动语言检测，也不承诺任意语言组合的混说准确率。

该工件使用以下唯一的兼容性声明：

```text
whisper.cpp-ggml
```

`whisper-rs` 0.16 随附的 whisper.cpp 加载器会校验 GGML 文件魔数；本集成不加载 GGUF，
文件扩展名也不是格式证明。

当前受审查工件来自 `ggerganov/whisper.cpp` 的固定修订
`5359861c739e955e79d9a303bcbc70fb988958b1`：多语种
`ggml-large-v3-turbo-q5_0.bin`，大小为 574,041,195 bytes，SHA-256 为
`394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2`。模型卡标识为
`openai/whisper-large-v3-turbo`，发布记录中的许可证标识为 `MIT`。

已安装客户端启动时只核对包内 `manifest.json`、编译进可执行文件的审查锁、常规文件属性
和字节数，不遍历模型计算运行时 SHA-256。模型摘要由发布暂存与 DMG 校验步骤核对，正式
产品以 macOS 应用签名保护包内资源。资源缺失、清单/大小不匹配或 Whisper 加载失败会禁用
内置 ASR；客户端不会下载、更新、恢复模型，也不会回退到云端或系统识别。

原生文件导入仍可用于可选的“高级本地模型”覆盖。此路径只读取用户选择的本机文件，且仍需
`whisper.cpp-ggml`、预期 SHA-256、模型卡和许可证确认；覆盖只对当前进程有效，重启后恢复
通过验证的内置默认模型。

## Runtime boundary

适配器只在原生 Rust 内存中接收 16 kHz 单声道 PCM，使用 beam search 5，将每个 WebRTC
VAD 语句作为无历史文本、单 segment 的独立请求，禁用翻译和自动语言检测，并以模型来源
写入最终转写记录。客户端没有 HTTP 模型下载器、模型 URL、云端回退或 WebView PCM 传输。

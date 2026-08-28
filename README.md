# 余波\~ \( Aftermath \) v1\.0\.2

- v1.0.2：修复反复提示“检测到潜在事件，等待 10 秒后开始记录...”但是没有触发截图的 bug。其他忘了。

> 当你打完游戏或做完某个大项目(或更多场景)时，bot 会发送符合你设置的人设的句子并回应你！
> 监控 CPU/GPU/自定义进程 占用和屏幕变化，当占用超过阈值且屏幕有明显变化时截图，占用下降后通过模型生成复盘并主动发送到指定会话。


---

## ✨ 主要特性

- **全平台监控**：支持 Windows /macOS/ Linux，自动检测 CPU 占用，并尝试适配 NVIDIA / AMD / Intel GPU（需安装对应库）。

- **屏幕变化检测**：基于像素差异比例判断 “明显变化”，避免细微抖动误触发。

- **事件录制**：触发后按设定间隔截图，条件恢复后自动结束并清理临时文件。

- **智能触发与冷却**：可配置 CPU/GPU 阈值区间、屏幕变化阈值、冷却时间，防止频繁打扰。

- **多模态描述**：调用 AstrBot 配置的 LLM（支持多模态）生成自然语言描述，结合人设输出个性化文案。

- **主动推送**：将截图与描述发送到指定目标会话（UMO）。

- **手动测试**：提供 `/am_start` 等指令，方便快速验证功能。

- **灵活配置**：支持按进程名监控、截图画质 / 分辨率选择、图片单独发送等高级选项。

---

## 📦 先决条件

- **图形环境**：运行 AstrBot 的主机必须具有可访问的显示（Windows 桌面 /macOS/ Linux \+ X11 或 Wayland）。

- **Python**：3\.8\+（与宿主 AstrBot 兼容）。

- **依赖**：`psutil`、`mss`、`Pillow`、`numpy`（pip 安装）。

---


（可选）GPU 监控依赖：

- NVIDIA：`pip install nvidia-ml-py`

- AMD：`pip install pyamdsmi`

- Intel：`pip install pyzes`（并设置环境变量 `ZES_ENABLE_SYSMAN=1`）

3. 在 AstrBot 插件管理页重载或重启 AstrBot。

---

## ⚙️ 配置说明（插件 UI）

在 AstrBot 的插件配置界面中可配置以下字段（JSON Schema 已内置条件显示逻辑）：

|配置项|类型|默认值|说明|
|---|---|---|---|
|cpu\_threshold|string|`"50~100"`|CPU 占用率区间，仅当 CPU 占用落在区间内才视为高负载（支持～、\-、, 分隔）|
|enable\_gpu|bool|true|是否启用 GPU 监控（需安装对应库，否则自动忽略）|
|gpu\_threshold|string|`"50~100"`|GPU 占用率区间（条件：enable\_gpu=true）|
|screen\_change\_threshold|float|5\.0|屏幕像素差异百分比阈值，超过视为明显变化|
|cooldown|float|300\.0|事件冷却时间（秒），结束后需等待此时间才能再次触发|
|check\_interval|float|2\.0|资源检查间隔（秒），数值越小响应越快但耗资源|
|screenshot\_interval|float|5\.0|事件期间截图间隔（秒）|
|process\_name|list|\[\]|指定监控进程名（如 cs2\.exe），填写后仅监控这些进程的启停，忽略 CPU/GPU 阈值。留空则使用资源监控|
|low\_load\_duration|int|15|负载低于阈值后持续该秒数才结束事件，防止波动误触发|
|process\_start\_duration|int|10|检测到进程启动或负载变化后等待该秒数再开始截图（避免拍到桌面）|
|process\_end\_duration|int|3|进程退出后等待该秒数再处理截图，确保进程完全关闭|
|target\_umo|string|`""`|目标会话 UMO（通过 /umo 获取），留空则只记录不发送|
|persona|string|`""`|人设 ID（AstrBot 人格），留空则使用默认描述模板|
|prompt\_template|text|见下|LLM 提示词模板，支持 `{persona}` 占位符|
|provider\_id|string|`""`|LLM Provider ID，留空使用全局默认模型|
|ollama\_api\_url|string|`"http://127.0.0.1:11434"`|Ollama API 地址，仅当多模态不可用时直连使用|
|screenshot\_quality|float|100|截图画质（0‑100），100 存为 PNG，否则 JPEG|
|screenshot\_resolution|string|`"原始"`|截图分辨率预设（原始、4K、2K、1080p、720p、480p），超过则等比缩放|
|send\_each\_image\_separately|bool|true|是否每张图片单独生成描述并立即发送（实时反馈）|
|send\_images|bool|true|是否将截图随消息一起发送|
|max\_images|int|3|最多发送的截图数量（条件：send\_images=true）|
|storage\_dir|string|`""`|截图保存目录（绝对路径），留空使用插件数据目录|

> 默认 prompt\_template：
> 
> ```Plain Text
> 请你以观察者的视角（用户正在操作电脑，你需要观察用户，而不是想象自己在操控电脑！），仔细观察这些连续截图，用符合以下人设的语气简要地说出来，不得长篇大论。
> 【人设】{persona}
> ```
> 
> 

---

## 🛠 指令（Commands）

- `/am_help` — 获取当前插件的所有命令和其介绍。

- `/umo` — 获取当前会话的 UMO，用于配置 target\_umo。

- `/am_status` — 查看当前监控状态、启用的 GPU 类型、已记录截图数。

- `/am_test` — 手动触发一次录制（每 5 秒截一张，共 4 张，然后发送）。

- `/am_clear` — 清除当前会话的记忆，防止旧话题干扰。

- `/am_start` — 手动触发一次录制，直到输入 `/am_stop` 停止。

- `/am_stop` — 停止录制，将截图发送给 LLM 并返回消息。

- （请将命令发送给你的 bot ，而不是发送给 Astrbot ）

---

## 📖 使用示例

1. 在插件 UI 中配置触发阈值、人设、目标 UMO（可先用 `/umo` 获取）。

2. 启动 / 重载插件，开始监控。

3. 当 CPU/GPU 高负载且屏幕有明显变化时，插件自动截图并调用 LLM 生成描述，发送到目标会话。

4. 负载下降后自动结束，等待冷却时间后再次待命。

---

## 🧯 故障排查

- **截图失败**：确保主机有图形环境（Windows 桌面、macOS、Linux \+ X11/Wayland）。

- **GPU 未检测**：检查是否安装对应库，并确认权限允许访问监控接口（Intel 的 pyzes 部分系统不可用）。

- **LLM 描述失败**：确认 AstrBot 已配置有效聊天模型，且模型支持图片输入；不支持时可关闭 send\_images。

- **消息未发送**：检查 target\_umo 是否正确（需完整格式如 `aiocqhttp:GroupMessage:123456789`）。

- **日志**：查看 AstrBot 插件日志获取详细错误。

---


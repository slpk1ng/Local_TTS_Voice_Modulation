- v1.0.2：修复反复提示“检测到潜在事件，等待 10 秒后开始记录...”但是没有触发截图的 bug。其他忘了。


\- v1.0.1：更新 `本地 LLM 深度思考功能` ，仅使用本地 ollama 模型时生效，thinking会在日志中输出，不会发送到最终回复中。



增加 `过滤无意义截图` 、 `进程未检测到时回退到CPU/GPU监控` 等功能。其他懒得写了



将原命令 /am\_start 的功能替换为 /am\_test （功能不变）；将命令 /am\_start的效果更新为：开始按照截图时间间隔截图，直到输入 /am\_stop 停止截图，将截图发送至 LLM 并返回消息；增加 /am_help 命令。





\- v1.0.0：正式版。修复已知 bug，新增进程监控、低负载持续时间、进程结束等待时间、截图画质 / 分辨率选择、每张图片单独生成描述、是否随消息发送截图等功能。





\- v0.1.0：测试版首发。



### 我的其他插件：

[https://github.com/slpk1ng/Local_TTS_Voice_Modulation](https://github.com/slpk1ng/Local_TTS_Voice_Modulation)
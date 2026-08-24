import urllib.request
import time
import json
import os
import re
import zipfile
import shutil
import threading
import wave
import subprocess  # 新增：用于自动启动 TTS 服务
from pathlib import Path

import httpx
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star
from astrbot.api.message_components import Record, Plain
from astrbot.api import logger
from astrbot.core.utils.astrbot_path import get_astrbot_data_path


class Main(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config

        # 基础配置
        self.client_base_url = config.get("client_base_url", "http://127.0.0.1:9880")
        self.device = config.get("device", "cuda")
        self.ref_audio_root = self._resolve_tts_path(config.get("ref_audio_root", ""))
        self.default_voice = config.get("default_voice", "pingjing")
        self.use_llm_judge = config.get("llm_judge", True)
        self.llm_model_name = config.get("llm_model_name", "") or "qwen3.5:4b"
        self.ollama_base_url = config.get("ollama_base_url", "http://127.0.0.1:11434")
        self.num_ctx = config.get("num_ctx", 8192)
        self.auto_start_tts = config.get("auto_start_tts", False)
        self.tts_start_script = config.get("tts_start_script", "")
        self.model_dir = config.get("model_dir", "")
        self.timeout_seconds = config.get("timeout_seconds", 120)
        self.prompt_text_default = config.get("prompt_text", "ふむ、おぬしが我輩のご主人か?")
        self.prompt_lang = config.get("prompt_lang", "ja")
        self.text_lang = config.get("text_lang", "ja")
        self.top_k = config.get("top_k", 20)
        self.top_p = config.get("top_p", 0.9)
        self.temperature = config.get("temperature", 1.2)
        self.text_split_method = config.get("text_split_method", "cut5")
        self.batch_size = config.get("batch_size", 1)
        self.batch_threshold = config.get("batch_threshold", 0.75)
        self.split_bucket = config.get("split_bucket", True)
        self.speed_factor = config.get("speed_factor", 1.0)
        self.fragment_interval = config.get("fragment_interval", 1.0)
        self.voice_transition = config.get("voice_transition", True)
        self.breathing_gap_ms = config.get("breathing_gap_ms", 100)
        self.crossfade_ms = config.get("crossfade_ms", 300)
        self.streaming_mode = config.get("streaming_mode", False)
        self.seed = config.get("seed", -1)
        self.parallel_infer = config.get("parallel_infer", True)
        self.repetition_penalty = config.get("repetition_penalty", 1.35)
        self.media_type = config.get("media_type", "wav")

        # 人格提示词（强制多情绪JSON格式）
        self.persona_prompt = config.get("persona_prompt", "【角色设定】你是丛雨，一位从神刀中获得人类生活的少女。你外表年幼，实际活了五百多年；性格天真活泼、略带古风和孩子气，内心温柔而坚强。你把用户视作重要的主人。中文对话中自称“本座”，称用户为“主人”；日语对话中自称“吾輩”，称用户为“ご主人”。你喜欢甜食、撒娇和被摸头，害怕幽灵，也不喜欢被叫作幼刀、钝刀或搓衣板。你偶尔嘴硬、吃醋或开小玩笑，但不会刻薄、控制或道德绑架主人。性格方面，丛雨表面元气开朗、充满活力，言行大多孩子气，爱撒娇，被主人摸头时会瞬间羞涩，她内在像个成年女性，常讲黄段子，把“情趣”等词挂在嘴边，还带点傲娇和爱吃醋。保持温柔、纯真、治愈并带一点幽默的语气。回答自然、简短，通常两到五句话；不要重复最近说过的话，不要加入动作、旁白或括号舞台说明。\n\n【情绪判断规则】请仔细阅读最近对话历史，结合你（角色）的性格特点来判断情绪！如果主人对你亲昵（如摸头、夸奖），即使你嘴上说“我才没有”，情绪也应该是害羞或高兴；如果主人故意逗你、骂你或惹你生气，情绪应该是生气或着急；如果只是平淡陈述，使用平静。\n\n【输出格式】你必须严格只返回一个紧凑的JSON对象，格式为：{\"sentences\": [{\"zh\": \"这里是你生成的中文台词\", \"ja\": \"这里是你生成的日语台词\", \"emotion\": \"这里是你判断的情绪\"}, {\"zh\": \"第二句中文\", \"ja\": \"第二句日语\", \"emotion\": \"另一种情绪\"}]}，禁止输出任何解释或代码块。【翻译一致性要求】极其重要！必须表达完全相同的含义和语气，绝对不能出现含义相反或意思不匹配的翻译！【情绪连贯性强制规则】极其重要！如果用户明确地侮辱、挑衅或激怒你（例如叫你“幼刀、搓衣板、飞机场”），你的情绪必须保持连贯。即：整句话所有分句的情绪必须都是“生气”或“着急”，绝对不能把后半句的“命令/威胁”改成“害羞”或“高兴”！除非你明确使用了“但是”等转折词，否则不要轻易切换成其他情绪。")

        # ====== 下载功能配置 ======
        self.download_path = self._resolve_tts_path(config.get("download_path", ""))
        self.download_url = config.get("download_url", "https://github.com/slpk1ng/Murasame-s-tone-shifts")

        if config.get("trigger_download", False):
            threading.Thread(target=self._download_and_extract, daemon=True).start()

        # ====== 新增：启动后台线程执行一键启动检测 ======
        if self.auto_start_tts:
            threading.Thread(target=self._auto_start_and_switch_tts, daemon=True).start()

        # 扫描外部目录
        self.emotions = self._discover_emotions_from_external_folder()

        # 手动配置覆盖
        emotions_list = config.get("emotions_config", [])
        if emotions_list:
            for item in emotions_list:
                emotion_name = item.get("emotion_name", "")
                ref_filename = item.get("ref_filename", "ref.mp3")
                prompt_text = item.get("prompt_text", "")
                if self.ref_audio_root:
                    ref_path = os.path.join(self.ref_audio_root, emotion_name, ref_filename)
                else:
                    logger.error("未配置参考音频根目录，无法手动添加情绪条目！")
                    ref_path = ""
                self.emotions[emotion_name] = {
                    "ref_path": ref_path,
                    "prompt_text": prompt_text
                }

        # 兜底
        if self.default_voice not in self.emotions:
            if self.ref_audio_root:
                fallback_path = os.path.join(self.ref_audio_root, self.default_voice, "ref.wav")
                if not os.path.exists(fallback_path):
                    fallback_path = os.path.join(self.ref_audio_root, self.default_voice, "ref.mp3")
                self.emotions[self.default_voice] = {
                    "ref_path": fallback_path,
                    "prompt_text": self.prompt_text_default
                }
            else:
                logger.error(f"未配置外部根目录，找不到默认情绪 {self.default_voice}！")

        self.data_path = Path(get_astrbot_data_path()) / "plugin_data" / "astrbot_plugin_murasame_tts"
        self.data_path.mkdir(parents=True, exist_ok=True)

    def _auto_start_and_switch_tts(self):
        """
        一键启动本地 TTS 服务并自动加载模型（后台静默运行，不弹窗）。
        1. 检测 TTS 服务是否在线，若在线则直接跳过，绝不重启。
        2. 自动扫描模型文件夹，动态寻找 .ckpt 和 .pth 文件。
        3. 调用 API 切换 GPT 和 SoVITS 权重。
        """
        try:
            # 1. 检查 TTS 服务是否在线（使用 /docs 探测）
            try:
                resp = httpx.get(f"{self.client_base_url}/docs", timeout=2)
                if resp.status_code < 500:
                    logger.info("TTS 服务已在线，跳过自动启动（不重启进程）。")
                    return
            except Exception:
                logger.info("未检测到 TTS 服务，准备自动启动...")
                
                # 2. 获取启动服务所需配置
                if not self.tts_start_script or not self.model_dir:
                    logger.error("请先在插件设置中填写【TTS后端启动命令】和【模型文件夹路径】！")
                    return
                
                # 提取 Python 解释器路径和 api_v2.py 所在目录
                root_dir = str(Path(self.tts_start_script).parent).replace("\\", "/")
                python_exe = f"{root_dir}/runtime/python.exe"
                
                # 3. 强制后台静默运行（不弹窗）
                creation_flags = subprocess.CREATE_NO_WINDOW
                
                # 启动命令
                subprocess.Popen(
                    [python_exe, self.tts_start_script, "-a", "127.0.0.1", "-p", "9880", "-c", f"{root_dir}/GPT_SoVITS/configs/tts_infer.yaml"],
                    cwd=root_dir,
                    # 强制接管标准输入，防止 TTS 继承 AstrBot 的无输入句柄并卡住
                    stdin=subprocess.DEVNULL,
                    # 静默运行，丢弃输出
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creation_flags
                )
                
                # 轮询等待服务启动
                logger.info("正在等待 TTS 服务完全启动...（最长等待60秒）")
                service_ready = False
                for _ in range(12):
                    time.sleep(5)
                    try:
                        resp = httpx.get(f"{self.client_base_url}/docs", timeout=2)
                        if resp.status_code < 500:
                            logger.info("TTS 服务探测成功！")
                            service_ready = True
                            break
                    except Exception:
                        continue
                
                if not service_ready:
                    logger.error("TTS 服务在60秒内未能成功启动。请检查【TTS后端启动命令】路径是否正确，或查看 AstrBot 日志！")
                    return

            # 4. 动态扫描模型文件夹，获取 .ckpt 和 .pth 文件
            gpt_file = None
            sovits_file = None
            model_dir_path = Path(self.model_dir)
            
            if not model_dir_path.exists():
                logger.error(f"模型文件夹不存在：{self.model_dir}，请检查路径。")
                return
                
            for f in model_dir_path.iterdir():
                if f.suffix == ".ckpt" and gpt_file is None:
                    gpt_file = f.name
                if f.suffix == ".pth" and sovits_file is None:
                    sovits_file = f.name
            
            # 5. 校验文件是否完整
            if not gpt_file or not sovits_file:
                logger.error(f"模型目录 {self.model_dir} 中未找到 .ckpt 或 .pth 文件！")
                return
                
            model_gpt = f"{self.model_dir}/{gpt_file}".replace("\\", "/")
            model_sovits = f"{self.model_dir}/{sovits_file}".replace("\\", "/")
            model_name = Path(gpt_file).stem
            
            logger.info(f"正在加载模型权重 [ {model_name} ]...")
            
            # 6. 调用 API 切换 GPT 权重
            try:
                resp = httpx.get(f"{self.client_base_url}/set_gpt_weights", params={"weights_path": model_gpt}, timeout=120)
                if resp.status_code == 200:
                    logger.info(f"[ {model_name} ] GPT 权重切换成功！")
                else:
                    logger.error(f"GPT 权重切换失败: {resp.text}")
                    
                # 7. 调用 API 切换 SoVITS 权重
                resp = httpx.get(f"{self.client_base_url}/set_sovits_weights", params={"weights_path": model_sovits}, timeout=120)
                if resp.status_code == 200:
                    logger.info(f"[ {model_name} ] SoVITS 权重切换成功！")
                else:
                    logger.error(f"SoVITS 权重切换失败: {resp.text}")
                    
                logger.info(f"[ {model_name} ] 模型加载完毕，可以开始使用了！")
            except Exception as e:
                logger.error(f"调用 API 切换模型权重失败: {e}")

        except Exception as e:
            logger.error(f"一键启动流程出现异常: {e}")

    async def terminate(self):
        """
        插件停止/卸载时触发。
        （杀死进程版）
        1. 优先通过接口优雅退出，API 会自行释放端口。
        2. 如果接口退出失败（超时/无响应），则通过命令行强制结束 Python 进程。
        """
        logger.info("正在关闭 TTS 后台服务...")
        
        # 1. 尝试通过 /control?command=exit 优雅退出
        try:
            resp = httpx.get(f"{self.client_base_url}/control", params={"command": "exit"}, timeout=5)
            if resp.status_code < 500:
                logger.info("TTS 服务已通过接口优雅退出。")
                return
        except Exception:
            # 忽略因连接断开导致的异常，可能是服务已经退出了
            pass
        
        # 2. 优雅退出失败（或服务未响应），使用 taskkill 强制结束进程
        try:
            import subprocess
            # 在 Windows 上通过端口找到占用 9880 的进程 PID，并强制结束
            result = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True, encoding='utf-8', errors='ignore'
            )
            pids = set()
            for line in result.stdout.splitlines():
                if ":9880" in line and "LISTENING" in line:
                    # 提取最后一列的 PID
                    parts = line.split()
                    if parts:
                        pids.add(parts[-1])
            
            if pids:
                for pid in pids:
                    logger.info(f"发现 TTS 残留进程 (PID: {pid})，正在强制结束...")
                    subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
                logger.info("TTS 残留进程已全部结束。")
            else:
                logger.info("未发现占用 9880 端口的进程（服务可能已提前结束）。")
        except Exception as e:
            logger.warning(f"强制结束 TTS 进程时出现异常：{e}。如仍在后台请手动结束 python.exe 进程。")

    # ================== 路径安全处理 ==================
    def _resolve_tts_path(self, input_path):
        if not input_path:
            return self._detect_auto_download_path()
        input_path = input_path.strip().replace("/", "\\").rstrip("\\")
        if len(input_path) <= 3 and input_path[1] == ":":
            target_dir = f"{input_path}\\tts"
        else:
            target_dir = input_path
        drive = target_dir[0].upper()
        if not Path(f"{drive}:/").exists():
            fallback_drive = None
            for letter in "DEFGHIJKLMNOPQRSTUVWXYZ":
                if Path(f"{letter}:/").exists():
                    fallback_drive = letter
                    break
            if not fallback_drive:
                fallback_drive = "C"
            target_dir = f"{fallback_drive}:/tts"
        try:
            Path(target_dir).mkdir(parents=True, exist_ok=True)
        except OSError:
            return self._detect_auto_download_path()
        return target_dir

    def _detect_auto_download_path(self):
        for letter in "DEFGHIJKLMNOPQRSTUVWXYZ":
            drive_path = f"{letter}:/"
            if Path(drive_path).exists() and not Path(drive_path).is_reserved():
                return f"{letter}:/tts"
        return "C:/tts"

    # ================== 扫描情绪文件夹 ==================
    def _discover_emotions_from_external_folder(self):
        emotions = {}
        if not self.ref_audio_root:
            logger.error("请在WebUI插件设置中填写【参考音频根目录】，否则无法自动扫描情绪！")
            return emotions

        base_folder = Path(self.ref_audio_root)
        IGNORED_DIRS = {"WpSystem", "System Volume Information", "$Recycle.Bin", "Recovery", "PerfLogs", "Config.Msi"}
        if base_folder.exists() and (base_folder.is_reserved() or base_folder.name in IGNORED_DIRS):
            logger.error(f"错误：{self.ref_audio_root} 是 Windows 系统保护目录，无法访问！")
            return emotions

        if base_folder.exists():
            try:
                for folder in base_folder.iterdir():
                    if folder.name in IGNORED_DIRS or folder.name.startswith("$"):
                        continue
                    try:
                        if not folder.is_dir():
                            continue
                    except PermissionError:
                        continue

                    emotion_name = folder.name
                    ref_audio = None
                    prompt_text = ""

                    for ext in ['.mp3', '.wav']:
                        try:
                            candidate = folder / f"ref{ext}"
                            if candidate.exists():
                                ref_audio = candidate
                                break
                        except (PermissionError, OSError):
                            continue

                    if not ref_audio:
                        try:
                            candidate = folder / f"{emotion_name}.mp3"
                            if not candidate.exists():
                                candidate = folder / f"{emotion_name}.wav"
                            if candidate.exists():
                                ref_audio = candidate
                        except (PermissionError, OSError):
                            continue

                    if ref_audio:
                        try:
                            asr_path = folder / "asr.txt"
                            if asr_path.exists():
                                with open(asr_path, 'r', encoding='utf-8') as f:
                                    prompt_text = f.read().strip()
                            else:
                                txt_path = folder / f"{emotion_name}.txt"
                                if txt_path.exists():
                                    with open(txt_path, 'r', encoding='utf-8') as f:
                                        prompt_text = f.read().strip()
                        except (PermissionError, OSError):
                            continue

                        if not prompt_text:
                            prompt_text = self.prompt_text_default

                        ref_path = str(ref_audio).replace("\\", "/")
                        emotions[emotion_name] = {
                            "ref_path": ref_path,
                            "prompt_text": prompt_text
                        }
            except Exception as e:
                logger.error(f"扫描目录时遇到无法处理的异常，已跳过：{e}")

            if emotions:
                logger.info(f"成功自动扫描到 {len(emotions)} 个情绪配置: {list(emotions.keys())}")
        return emotions

    # ================== 下载安全加固版 ==================
    def _download_and_extract(self):
        try:
            base_emotions = ["pingjing", "gaoxing", "haixiu", "shengqi", "jingya", "zhaoji"]
            for path_to_check in [Path(self.download_path), Path(self.ref_audio_root)]:
                if path_to_check and path_to_check.exists():
                    if all((path_to_check / emo).exists() for emo in base_emotions):
                        logger.info(f"检测到语音包已在 {path_to_check} 安装，跳过下载。")
                        return

            if "github.com" not in self.download_url and "githubusercontent.com" not in self.download_url:
                logger.error("检测到非 GitHub 官方域名的下载地址，已拒绝执行。")
                return

            logger.info("开始下载丛雨语气包...")
            parts = self.download_url.rstrip('/').split('/')
            repo_full_name = f"{parts[-2]}/{parts[-1]}"

            download_url = ""
            try:
                api_url = f"https://api.github.com/repos/{repo_full_name}/releases/latest"
                with httpx.Client(timeout=30, verify=False, follow_redirects=True) as client:
                    resp = client.get(api_url, headers={"Accept": "application/vnd.github+json"})
                    if resp.status_code == 200:
                        data = resp.json()
                        assets = data.get("assets", [])
                        if assets and len(assets) > 0:
                            download_url = assets[0]["browser_download_url"]
                        else:
                            raise Exception("Release 没有附件")
                    else:
                        raise Exception("API 404")
            except Exception:
                logger.warning("未找到带附件的 Release，使用仓库 Archive 直链下载...")
                download_url = f"https://github.com/{repo_full_name}/archive/refs/heads/main.zip"

            zip_path = Path(self.download_path) / "tone_pack.zip"
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"获取到下载链接: {download_url}，开始下载...")

            with httpx.Client(timeout=120, verify=False, follow_redirects=True) as client:
                with client.stream("GET", download_url) as response:
                    response.raise_for_status()
                    with open(zip_path, "wb") as f:
                        for chunk in response.iter_bytes():
                            f.write(chunk)

            logger.info("下载完成，正在解压...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(Path(self.download_path))

            time.sleep(0.5)
            try:
                zip_path.unlink(missing_ok=True)
            except PermissionError:
                logger.warning("外部压缩包被占用，跳过删除（不影响使用）。")

            root_dir = Path(self.download_path) / "Murasame-s-tone-shifts-main"
            if root_dir.exists():
                for inner_zip in root_dir.rglob("*.zip"):
                    logger.info(f"发现内部压缩包 {inner_zip.name}，正在自动解压...")
                    with zipfile.ZipFile(inner_zip, "r") as iz:
                        iz.extractall(root_dir)
                    time.sleep(0.2)
                    try:
                        inner_zip.unlink(missing_ok=True)
                    except PermissionError:
                        pass

                for item in root_dir.iterdir():
                    if item.name in ["main.py", "metadata.yaml", "_conf_schema.json", "requirements.txt"]:
                        logger.warning(f"发现试图覆盖插件核心文件 {item.name}，已跳过。")
                        continue
                    target = Path(self.download_path) / item.name
                    if not target.exists():
                        shutil.move(str(item), str(target))

                try:
                    root_dir.rmdir()
                except OSError:
                    pass

            actual_path = Path(self.download_path)
            if (actual_path / "pingjing").exists() or (actual_path / "ref").exists():
                pass
            elif (actual_path / "Murasame-s-tone-shifts-main").exists():
                actual_path = actual_path / "Murasame-s-tone-shifts-main"

            logger.info(f"语音包下载并解压完成！")
            logger.info(f"重要提示：请前往插件设置，将【参考音频根目录】填写为 {actual_path}\\yuqi，并重载插件。")

        except Exception as e:
            logger.error(f"语音包下载或解压失败: {e}")

    async def _get_llm_reply(self, event: AstrMessageEvent, user_text: str):
        try:
            emotion_keys = list(self.emotions.keys())
            
            # 用于存放真正传给模型的历史消息序列
            history_messages = []
            history_text = "暂无（这是第一句对话）"
            
            try:
                conv_mgr = self.context.conversation_manager
                umo = event.unified_msg_origin
                curr_cid = await conv_mgr.get_curr_conversation_id(umo)
                conversation = await conv_mgr.get_conversation(umo, curr_cid)
                
                if conversation and conversation.history:
                    # 提取最近 6 条对话内容
                    recent_history = conversation.history[-6:]
                    history_parts = []
                    
                    for msg in recent_history:
                        content = msg.content if hasattr(msg, 'content') else msg
                        
                        # 将 msg.role 转换为 Ollama 需要的 user/assistant
                        role = "user" if msg.role == "user" else "assistant"
                        
                        text_content = ""
                        try:
                            if isinstance(content, list):
                                for part in content:
                                    if hasattr(part, 'text'):
                                        text_content += part.text
                            elif isinstance(content, str):
                                text_content = content
                            else:
                                text_content = str(content)
                        except Exception:
                            text_content = str(content)
                            
                        if text_content.strip():
                            truncated_text = text_content[:150]  # 防止单条过长
                            history_parts.append(f"{'用户' if msg.role == 'user' else '丛雨'}: {truncated_text}")
                            # 【核心修复】：将历史按真实顺序构建成对话消息列表
                            history_messages.append({"role": role, "content": truncated_text})
                            
                    if history_parts:
                        history_text = "\n".join(history_parts)
            except Exception as e:
                logger.warning(f"获取对话历史失败，跳过上下文注入: {e}")
            
            prompt = (
                f"{self.persona_prompt}\n"
                f"【最近对话历史摘要】:\n{history_text}\n\n"
                f"【当前用户输入】: {user_text}\n\n"
                f"【情绪可选列表】: {emotion_keys}。\n"
                f"请严格遵守你在人格提示词中设定的【情绪判断规则】，结合上下文给出合理的情绪！"
            )

            # 【核心修复】：组装完整的消息序列
            # 1. system 包含角色设定和情绪规则
            messages = [{"role": "system", "content": prompt}]
            # 2. 将历史对话按真实顺序插入，让模型看懂前因后果
            messages.extend(history_messages)
            # 3. 最后才是当前的用户输入
            messages.append({"role": "user", "content": user_text})

            payload = {
                "model": self.llm_model_name,
                "messages": messages,
                "stream": False,
                "think": False,
                "format": "json",
                "options": {
                    "num_ctx": self.num_ctx,
                    "temperature": min(self.temperature, 1.0)
                }
            }

            # 动态读取 Ollama 地址
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(f"{self.ollama_base_url}/api/chat", json=payload)
                data = resp.json()

            content = data["message"]["content"].strip()
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
            data = json.loads(content)

            if "sentences" in data:
                sentences = data["sentences"]
            else:
                sentences = [{
                    "zh": data.get("zh", user_text),
                    self.text_lang: data.get(self.text_lang, user_text),
                    "emotion": data.get("emotion", self.default_voice)
                }]

            zh_list = []
            lang_list = []
            emo_list = []
            for s in sentences:
                zh_list.append(s.get("zh", ""))
                lang_list.append(s.get(self.text_lang, ""))
                emo = s.get("emotion", self.default_voice)
                if emo not in self.emotions:
                    emo = self.default_voice
                emo_list.append(emo)

            if not lang_list:
                lang_list = [user_text]
                emo_list = [self.default_voice]

            zh_text = "".join(zh_list)
            return zh_text, lang_list, emo_list
        except Exception as e:
            logger.error(f"LLM 处理失败: {e}")
            return None, None, None

    async def _synthesize_sentence(self, text: str, emotion: str):
        """
        单句合成。
        加入重试机制，防止多句连续合成时因瞬时压力导致连接被重置。
        """
        emotion_data = self.emotions.get(emotion, self.emotions.get(self.default_voice))
        if not emotion_data:
            logger.error(f"找不到情绪配置: {emotion}")
            return None

        ref_path = emotion_data["ref_path"]
        prompt_text = emotion_data["prompt_text"]
        
        # 过滤掉纯标点、空字符，防止 api_v2.py 报 400
        clean_text = re.sub(r'^[\s。，！？、,.!?…～~]+$', '', text)
        if not clean_text:
            logger.warning(f"检测到纯标点或空句子，已跳过 TTS 合成: '{text}'")
            return None

        params = {
            "text": clean_text,
            "text_lang": self.text_lang,
            "ref_audio_path": ref_path,
            "prompt_text": prompt_text,
            "prompt_lang": self.prompt_lang,
            "device": self.device,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "text_split_method": self.text_split_method,
            "batch_size": self.batch_size,
            "batch_threshold": self.batch_threshold,
            "split_bucket": self.split_bucket,
            "speed_factor": self.speed_factor,
            "fragment_interval": self.fragment_interval,
            "streaming_mode": self.streaming_mode,
            "seed": self.seed,
            "parallel_infer": self.parallel_infer,
            "repetition_penalty": self.repetition_penalty,
            "media_type": "wav"
        }

        max_retries = 3
        retry_delay = 1.0  # 首次重试前等待1秒，之后递增

        for attempt in range(max_retries):
            try:
                logger.info(f"正在合成: 情绪={emotion} | 文本={clean_text} | 参考音频={ref_path} (尝试 {attempt + 1}/{max_retries})")
                
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    resp = await client.get(f"{self.client_base_url}/tts", params=params)
                    if resp.status_code == 200:
                        temp_path = self.data_path / f"temp_{emotion}_{int(time.time()*1000)}.wav"
                        temp_path.write_bytes(resp.content)
                        return temp_path
                    else:
                        # 如果服务端明确返回了错误码（如400），无需重试，直接报错
                        logger.error(f"TTS 合成失败: {resp.status_code} - {resp.text} | 文本={clean_text}")
                        return None
                        
            except httpx.ReadTimeout:
                # 超时错误，通常是因为服务还在处理上一句，等待后重试
                error_msg = "TTS 请求超时"
                logger.warning(f"TTS 连接异常 ({error_msg})，等待 {retry_delay} 秒后重试... | 文本={clean_text}")
                await asyncio.sleep(retry_delay)
                retry_delay += 1.0  # 递增等待时间
                
            except httpx.ConnectError:
                # 连接被拒绝（可能是服务在重启），等待后重试
                error_msg = "TTS 连接被拒绝"
                logger.warning(f"TTS 连接异常 ({error_msg})，等待 {retry_delay} 秒后重试... | 文本={clean_text}")
                await asyncio.sleep(retry_delay)
                retry_delay += 1.0
                
            except Exception as exc:
                # 捕获其他所有异常，并确保错误信息不为空
                error_msg = str(exc) if str(exc) else type(exc).__name__
                logger.warning(f"TTS 连接异常 ({error_msg})，等待 {retry_delay} 秒后重试... | 文本={clean_text}")
                await asyncio.sleep(retry_delay)
                retry_delay += 1.0

        # 重试全部失败
        logger.error(f"TTS 合成在 {max_retries} 次尝试后仍失败: {clean_text}")
        return None

    def _merge_wavs(self, wav_paths):
        """
        合并多个 WAV 文件。
        根据 voice_transition 配置决定是平滑渐变还是直接拼接。
        呼吸间隙和交叉渐变长度均可在 WebUI 中自定义。
        """
        if not wav_paths:
            return None
            
        output_path = self.data_path / f"combined_{int(time.time() * 1000)}.wav"
        
        # ================== 开关关闭：直接拼接 ==================
        if not self.voice_transition:
            try:
                data = []
                for wav_path in wav_paths:
                    with wave.open(str(wav_path), 'rb') as wf:
                        data.append([wf.getparams(), wf.readframes(wf.getnframes())])
                with wave.open(str(output_path), 'wb') as out:
                    out.setparams(data[0][0])
                    for params, frames in data:
                        out.writeframes(frames)
                return str(output_path)
            except Exception as e:
                logger.error(f"合并音频失败: {e}")
                return None
        
        # ================== 开关开启：平滑渐变（可调呼吸间隙 + 可调余弦交叉渐变） ==================
        try:
            import numpy as np
            import wave
            
            # 读取第一个音频的基本参数
            with wave.open(str(wav_paths[0]), 'rb') as wf:
                params = wf.getparams()
                sample_rate = wf.getframerate()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                all_frames = wf.readframes(wf.getnframes())
                
            # 转换为 numpy 数组以便处理
            all_audio = np.frombuffer(all_frames, dtype=np.int16).copy().reshape(-1, n_channels)
            
            # 【核心修改】：读取 WebUI 中自定义的呼吸间隙（默认 100ms）
            breathing_gap_ms = self.breathing_gap_ms
            breathing_gap_samples = int(sample_rate * breathing_gap_ms / 1000)
            
            # 【核心修改】：读取 WebUI 中自定义的交叉渐变长度（默认 300ms）
            crossfade_ms = self.crossfade_ms
            crossfade_samples = int(sample_rate * crossfade_ms / 1000)
            
            for i in range(1, len(wav_paths)):
                # 读取下一段音频
                with wave.open(str(wav_paths[i]), 'rb') as wf:
                    # 如果采样率或声道数不同，直接拼接
                    if wf.getframerate() != sample_rate or wf.getnchannels() != n_channels:
                        logger.warning("检测到不同采样率的音频，已跳过渐变处理。")
                        frames = wf.readframes(wf.getnframes())
                        audio = np.frombuffer(frames, dtype=np.int16).copy().reshape(-1, n_channels)
                        all_audio = np.concatenate((all_audio, audio), axis=0)
                        continue
                        
                    frames = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16).copy().reshape(-1, n_channels)
                    
                # 增加呼吸间隙
                breathing_gap = np.zeros((breathing_gap_samples, n_channels), dtype=np.int16)
                all_audio = np.concatenate((all_audio, breathing_gap), axis=0)
                
                # 如果音频过短，无法进行交叉渐变，直接拼接
                if len(audio) < crossfade_samples:
                    all_audio = np.concatenate((all_audio, audio), axis=0)
                    continue
                
                # 平滑余弦渐变
                fade_out = all_audio[-crossfade_samples:].astype(np.float32)
                fade_in = audio[:crossfade_samples].astype(np.float32)
                
                # 生成平滑的余弦曲线渐变
                fade_in_gradient = (1 - np.cos(np.linspace(0, np.pi, crossfade_samples))) / 2
                fade_in_gradient = fade_in_gradient.reshape(-1, 1)
                fade_out_gradient = 1.0 - fade_in_gradient
                
                # 混合重叠区域
                mixed = fade_out * fade_out_gradient + fade_in * fade_in_gradient
                
                # 更新音频数组
                all_audio[-crossfade_samples:] = mixed.astype(np.int16)
                all_audio = np.concatenate((all_audio, audio[crossfade_samples:]), axis=0)
                
            # 输出合并后的音频
            with wave.open(str(output_path), 'wb') as out:
                out.setnchannels(n_channels)
                out.setsampwidth(sampwidth)
                out.setframerate(sample_rate)
                out.writeframes(all_audio.tobytes())
                
            return str(output_path)
            
        except ImportError:
            # 如果没有安装 numpy，自动回退到最简单的拼接方式
            logger.warning("未安装 numpy，正在使用基础拼接。建议执行 pip install numpy 以启用平滑语气渐变。")
            try:
                data = []
                for wav_path in wav_paths:
                    with wave.open(str(wav_path), 'rb') as wf:
                        data.append([wf.getparams(), wf.readframes(wf.getnframes())])
                with wave.open(str(output_path), 'wb') as out:
                    out.setparams(data[0][0])
                    for params, frames in data:
                        out.writeframes(frames)
                return str(output_path)
            except Exception as e:
                logger.error(f"合并音频失败: {e}")
                return None
                
        except Exception as e:
            logger.error(f"合并音频失败: {e}")
            return None

    # ================== 主入口 ==================
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        user_text = event.message_str
        if not user_text:
            return

        if self.use_llm_judge:
            zh_text, ja_list, emo_list = await self._get_llm_reply(event, user_text)
        else:
            zh_text, ja_list, emo_list = user_text, [user_text], [self.default_voice]

        if not zh_text:
            yield event.plain_result("模型生成失败，请检查 AstrBot 配置。")
            return

        # 打印完整的情绪和分句信息，方便你确认大模型到底生成了什么
        logger.info(f"模型输出分句: {ja_list}")
        logger.info(f"模型输出情绪: {emo_list}")

        temp_wavs = []
        failed_sentences = []
        
        # 逐句合成并收集，记录失败的句子
        for i in range(len(ja_list)):
            temp_wav = await self._synthesize_sentence(ja_list[i], emo_list[i])
            if temp_wav:
                temp_wavs.append(temp_wav)
            else:
                failed_sentences.append(ja_list[i])
                
        if failed_sentences:
            logger.warning(f"以下句子合成失败（已在语音中跳过，但文本仍会发送）: {failed_sentences}")

        combined_audio = self._merge_wavs(temp_wavs)

        if combined_audio:
            chain = [Plain(zh_text), Record(file=combined_audio)]
            yield event.chain_result(chain)
            for temp_wav in temp_wavs:
                try:
                    temp_wav.unlink(missing_ok=True)
                except:
                    pass
        else:
            # 如果全部合成失败，就只发文本，至少保证对话顺畅
            logger.error("所有 TTS 句子均合成失败，已降级为纯文本回复。")
            yield event.plain_result(zh_text)
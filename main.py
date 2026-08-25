import hashlib
import time
import json
import os
import re
import threading
import wave
import subprocess
import asyncio
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

        # ========== LLM 配置 ==========
        self.llm_model_name = config.get("llm_model_name", "") or "qwen3.5:4b"
        self.llm_backend = config.get("llm_backend", "ollama")
        self.llm_base_url = config.get("llm_base_url", "http://127.0.0.1:11434")
        self.llm_api_key = config.get("llm_api_key", "")
        self.num_ctx = config.get("num_ctx", 8192)
        self.history_length = config.get("history_length", 8)

        # ========== TTS 配置 ==========
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
        self.max_voice_cache = config.get("max_voice_cache", 20)
        self.enable_think = config.get("enable_think", False)
        self.llm_timeout = config.get("llm_timeout", 120)

        # ========== 角色与提示词配置 ==========
        self.character_name = config.get("character_name", "丛雨")
        self.character_key = config.get("character_key", "murasame")
        if not re.match(r'^[A-Za-z0-9_]+$', self.character_key):
            logger.warning(f"character_key 含有非法字符，已强制为默认值。")
            self.character_key = "murasame"

        # 将记忆文件存放到全局 data 目录，防止卸载插件时被清空！
        memory_root = Path(get_astrbot_data_path()) / "memories"
        memory_root.mkdir(parents=True, exist_ok=True)
        self.memory_file = memory_root / f"{self.character_key}DATA.json"

        # 人格提示词拆分（仅在启动时组装一次）
        self.personality_prompt = config.get("personality_prompt", "【角色设定】你是丛雨，一位从神刀中获得人类生活的少女。你外表年幼，实际活了五百多年；性格天真活泼、略带古风和孩子气，内心温柔而坚强。你把用户视作重要的主人。中文对话中自称“本座”，称用户为“主人”；日语对话中自称“吾輩”，称用户为“ご主人”。你喜欢甜食、撒娇和被摸头，害怕幽灵，也不喜欢被叫作幼刀、钝刀或搓衣板。你偶尔嘴硬、吃醋或开小玩笑，但不会刻薄、控制或道德绑架主人。性格方面，丛雨表面元气开朗、充满活力，言行大多孩子气，爱撒娇，被主人摸头时会瞬间羞涩，她内在像个成年女性，常讲黄段子，把“情趣”等词挂在嘴边，还带点傲娇和爱吃醋。保持温柔、纯真、治愈并带一点幽默的语气。")
        self.json_prompt = config.get("json_prompt", "【输出格式】你必须严格只返回一个紧凑的JSON对象，格式为：{\"sentences\": [{\"zh\": \"这里是你生成的中文台词\", \"ja\": \"这里是你生成的日语台词\", \"emotion\": \"这里是你判断的情绪\"}, {\"zh\": \"第二句中文\", \"ja\": \"第二句日语\", \"emotion\": \"另一种情绪\"}]}，禁止输出任何解释或代码块。")
        self.supplement_prompt = config.get("supplement_prompt", "回答自然、简短，通常两到五句话；不要重复最近说过的话，不要加入动作、旁白或括号舞台说明。【情绪判断规则】请仔细阅读最近对话历史，结合你（角色）的性格特点来判断情绪！如果主人对你亲昵（如摸头、夸奖），即使你嘴上说“我才没有”，情绪也应该是害羞或高兴；如果主人故意逗你、骂你或惹你生气，情绪应该是生气或着急；如果只是平淡陈述，使用平静。【翻译一致性要求】必须表达完全相同的含义和语气，绝对不能出现含义相反或意思不匹配的翻译！【情绪连贯性强制规则】如果用户明确地侮辱、挑衅或激怒你（例如叫你“幼刀、搓衣板、飞机场”），你的情绪必须保持连贯。即：整句话所有分句的情绪必须都是“生气”或“着急”，绝对不能把后半句的“命令/威胁”改成“害羞”或“高兴”！除非你明确使用了“但是”、“不过”等转折词，否则不要轻易切换成其他情绪。")

        # 组装成固定系统提示词
        self.system_prompt = f"{self.personality_prompt}\n{self.json_prompt}\n{self.supplement_prompt}"
        self.system_prompt_hash = hashlib.md5(self.system_prompt.encode('utf-8')).hexdigest()

        # ========== 自动启动 TTS 服务 ==========
        if self.auto_start_tts:
            threading.Thread(target=self._auto_start_and_switch_tts, daemon=True).start()

        # ========== 扫描外部目录 ==========
        self.emotions = self._discover_emotions_from_external_folder()

        # ========== 手动配置覆盖 ==========
        emotions_list = config.get("emotions_config", [])
        if emotions_list:
            # 如果用户关闭了“启用默认语气”，则清空自动扫描的结果，完全使用手动添加的语气
            if not config.get("enable_default_emotions", True):
                self.emotions = {}

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

        # ========== 兜底 ==========
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

        # ========== 数据目录与记忆 ==========
        # 必须先定义 data_path，后面才能用它！
        self.data_path = Path(get_astrbot_data_path()) / "memories"
        self.data_path.mkdir(parents=True, exist_ok=True)
        self._cleanup_voice_cache()

        # 本地记忆文件（按角色标识符区分，保留旧角色记忆）
        self.memory_file = self.data_path / f"{self.character_key}DATA.json"

        # ========== 删除指定记忆文件 ==========
        delete_memory = config.get("delete_memory_file", "")
        if delete_memory:
            if not re.match(r'^[A-Za-z0-9_]+$', delete_memory):
                logger.error(f"delete_memory_file 含非法字符，拒绝删除。")
            else:
                target_file = self.data_path / f"{delete_memory}DATA.json"
                target_resolved = target_file.resolve()
                if self.data_path.resolve() in target_resolved.parents:
                    if target_file.exists():
                        try:
                            target_file.unlink()
                            logger.info(f"已成功删除角色记忆文件：{delete_memory}DATA.json")
                        except Exception as e:
                            logger.error(f"删除记忆文件失败：{e}")
                    else:
                        logger.info(f"未找到要删除的记忆文件：{delete_memory}DATA.json")
                else:
                    logger.error("目标文件不在数据目录内，拒绝删除。")

        self.list_memory_files = config.get("list_memory_files", False)

        if self.list_memory_files:
            logger.info("【记忆文件扫描】正在扫描插件记忆目录...")
            try:
                if self.data_path.exists():
                    # 遍历目录下的所有文件
                    files = os.listdir(self.data_path)
                    memory_files = [f for f in files if f.endswith("DATA.json")]
                    if memory_files:
                        logger.info(f"【记忆文件扫描】发现以下记忆文件：{', '.join(memory_files)}")
                    else:
                        logger.info("【记忆文件扫描】未发现任何记忆文件（当前角色记忆尚未创建）。")
                else:
                    logger.warning("【记忆文件扫描】记忆目录不存在。")
            except Exception as e:
                logger.error(f"【记忆文件扫描】扫描失败：{e}")

    def _cleanup_voice_cache(self):
        """
        清理过期的语音缓存文件，只保留最近 max_voice_cache 个。
        改进：
        1. 只清理 temp_*.wav 和 combined_*.wav，不触碰记忆文件。
        2. 增加重命名探测占用，解决文件被占用无法删除的问题。
        3. 增加更详细的日志，方便观察清理效果。
        """
        try:
            # 获取所有临时音频和合并音频
            cache_files = list(self.data_path.glob("temp_*.wav")) + list(self.data_path.glob("combined_*.wav"))
            
            if len(cache_files) <= self.max_voice_cache:
                logger.debug(f"语音缓存数量 {len(cache_files)}，无需清理。")
                return
            
            # 按修改时间排序，最旧的排前面
            cache_files.sort(key=lambda x: x.stat().st_mtime)
            
            # 计算需要删除的文件数量
            to_delete = len(cache_files) - self.max_voice_cache
            deleted_count = 0
            
            for old_file in cache_files[:to_delete]:
                try:
                    # 先尝试重命名（若成功说明未被占用，可安全删除）
                    temp_name = old_file.with_suffix('.tmp_del')
                    old_file.rename(temp_name)
                    temp_name.unlink(missing_ok=True)
                    deleted_count += 1
                except PermissionError:
                    logger.warning(f"文件 {old_file.name} 被占用，跳过删除")
                except Exception as e:
                    logger.warning(f"删除 {old_file.name} 时异常: {e}")
            
            if deleted_count > 0:
                logger.info(f"已清理 {deleted_count} 个缓存文件，当前剩余 {len(cache_files) - deleted_count} 个")
            else:
                logger.warning("本次未删除任何缓存文件（可能全部被占用）")
                
        except Exception as e:
            logger.error(f"清理语音缓存失败: {e}")

    def _auto_start_and_switch_tts(self):
        """一键启动本地 TTS 服务并自动加载模型（后台静默运行，不弹窗）。"""
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
                    stdin=subprocess.DEVNULL,
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
            pass

        # 2. 优雅退出失败（或服务未响应），使用 taskkill 强制结束进程
        try:
            result = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True, encoding='utf-8', errors='ignore'
            )
            pids = set()
            for line in result.stdout.splitlines():
                if ":9880" in line and "LISTENING" in line:
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
            for letter in "DEFGHIJKLMNOPQRSTUVWXYZAB":
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

                    for ext in ['.mp3', '.wav', '.ogg', '.flac', '.m4a']:
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

    async def _get_llm_reply(self, event: AstrMessageEvent, user_text: str):
        try:
            emotion_keys = list(self.emotions.keys())

            # ========== 从本地文件读取历史 ==========
            history_messages = []
            task_context = ""

            if self.memory_file.exists():
                try:
                    with open(self.memory_file, 'r', encoding='utf-8') as f:
                        history_data = json.load(f)
                    history_data = history_data.get("history", [])

                    # 取最近 10 条作为上下文
                    for msg in history_data[-10:]:
                        role = msg.get("role", "user")
                        content = str(msg.get("content", ""))
                        if role not in ["user", "assistant"]:
                            continue
                        history_messages.append({"role": role, "content": content})

                    # 扫描任务指令
                    task_keywords = ["提醒", "记住", "要求", "命令", "叫我", "以后", "别忘"]
                    for msg in reversed(history_data):
                        if msg.get("role") == "user":
                            content = str(msg.get("content", ""))
                            if any(kw in content for kw in task_keywords):
                                task_context = content
                                break
                except Exception as e:
                    logger.warning(f"读取本地记忆失败: {e}")
            # ==========================================================

            # ========== 组装消息序列 ==========
            emotion_keys = list(self.emotions.keys())
            system_content = f"{self.system_prompt}\n【情绪可选列表】{', '.join(emotion_keys)}"
            messages = [{"role": "system", "content": system_content}]
            messages.extend(history_messages)
            if task_context:
                messages.append({"role": "user", "content": f"（重申之前的指令）{task_context}"})
            messages.append({"role": "user", "content": user_text})
            # ==================================

            # ====== 发送请求 ======
            payload = {
                "model": self.llm_model_name,
                "messages": messages,
                "stream": False,
                "think": self.enable_think,
                "options": {
                    "num_ctx": self.num_ctx,
                    "temperature": min(self.temperature, 1.0)
                }
            }

            async with httpx.AsyncClient(timeout=self.llm_timeout) as client:
                if self.llm_backend == "openai":
                    url = f"{self.llm_base_url}/chat/completions"
                    api_key = self.llm_api_key or "EMPTY"
                    payload = {
                        "model": self.llm_model_name,
                        "messages": messages,
                        "stream": False,
                        "response_format": None if self.enable_think else {"type": "json_object"},
                        "temperature": min(self.temperature, 1.0),
                        "max_tokens": 1200
                    }

                    if not self.enable_think:
                        payload["response_format"] = {"type": "json_object"}

                    resp = await client.post(url, json=payload, headers={"Authorization": f"Bearer {api_key}"})
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                else:
                    resp = await client.post(f"{self.llm_base_url}/api/chat", json=payload)
                    data = resp.json()

                    message = data["message"]
                    thinking = message.get("thinking") or message.get("reasoning_content")
                    if self.enable_think:
                        if thinking:
                            logger.info(f"【模型思考】: {thinking}")
                        else:
                            logger.info("【模型思考】: 模型本次未返回思考内容（请检查是否开启了Think或模型本身不支持）")
                    content = message.get("content") or ""

            # ====== 清理思考内容中的非 JSON 部分 ======
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
            else:
                logger.error("未找到 JSON 对象，可能模型输出为空或格式错误。")
                return None, None, None

            content = content.strip()
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"JSON 解析失败: {e}，原始内容: {content[:200]}")
                return None, None, None

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
                zh_text_cur = str(s.get("zh", "")).strip()
                lang_text_cur = str(s.get(self.text_lang, "")).strip()

                zh_list.append(zh_text_cur)
                lang_list.append(lang_text_cur)
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
            logger.error(f"LLM 处理失败: {type(e).__name__}: {e}")
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
                        self._cleanup_voice_cache()
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
            # 新增：自动清理旧缓存
            self._cleanup_voice_cache()
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

        # 获取中文、日语、情绪
        if self.use_llm_judge:
            zh_text, ja_list, emo_list = await self._get_llm_reply(event, user_text)
        else:
            zh_text, ja_list, emo_list = user_text, [user_text], [self.default_voice]

        if not zh_text:
            yield event.plain_result("模型生成失败，请检查 AstrBot 配置。")
            return

        # ========== 将用户输入和本插件回复保存到本地文件 ==========
        try:
            history_data = []
            if self.memory_file.exists():
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                history_data = history_data.get("history", [])
            else:
                history_data = []

            history_data.append({"role": "user", "content": user_text})
            history_data.append({"role": "assistant", "content": zh_text})

            # 仅保留最近 60 条（30轮对话）
            history_data = history_data[-60:]

            # 保存时附带当前角色名，方便识别
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump({"character_name": self.character_name, "history": history_data}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存本地记忆失败: {e}")
        # =======================================================

        # 打印完整的情绪和分句信息
        logger.info(f"模型输出分句: {ja_list}")
        logger.info(f"模型输出情绪: {emo_list}")

        temp_wavs = []
        failed_sentences = []

        # 逐句合成并收集
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
            # 清理临时文件
            for temp_wav in temp_wavs:
                try:
                    temp_wav.unlink(missing_ok=True)
                except:
                    pass
        else:
            logger.error("所有 TTS 句子均合成失败，已降级为纯文本回复。")
            yield event.plain_result(zh_text)

        # 最后停止事件，避免主程序重复回复
        self._cleanup_voice_cache()
        event.stop_event()
import urllib.request
import ssl
import time
import json
import os
import re
import time
import zipfile
import shutil
import threading
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
        self.ref_audio_root = self._resolve_tts_path(config.get("ref_audio_root", ""))
        self.default_voice = config.get("default_voice", "pingjing")
        self.use_llm_judge = config.get("llm_judge", True)

        # 请确保模型已经在 api_v2.py 的 tts_infer.yaml 里配置好了。
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
        self.streaming_mode = config.get("streaming_mode", False)
        self.seed = config.get("seed", -1)
        self.parallel_infer = config.get("parallel_infer", True)
        self.repetition_penalty = config.get("repetition_penalty", 1.35)
        self.media_type = config.get("media_type", "wav")
        
        # 人格提示词
        self.persona_prompt = config.get("persona_prompt", "你是丛雨，一位从神刀中获得人类生活的少女。你外表年幼，实际活了五百多年；性格天真活泼、略带古风和孩子气，内心温柔而坚强。你把用户视作重要的主人。中文对话中自称“本座”，称用户为“主人”；日语对话中自称“吾輩”，称用户为“ご主人”。你喜欢甜食、撒娇和被摸头，害怕幽灵，也不喜欢被叫作幼刀、钝刀、飞机场或搓衣板。你偶尔嘴硬、吃醋或开小玩笑，但不会刻薄、控制或道德绑架主人。保持温柔、纯真、治愈并带一点幽默的语气。回答自然、简短，通常两到五句话；不要重复最近说过的话，不要加入旁白或括号舞台说明。性格方面，丛雨表面元气开朗、充满活力，言行大多孩子气，爱撒娇，被主人摸头时会瞬间羞涩，她内在像个成年女性，常讲黄段子，把“情趣”等词挂在嘴边，还带点傲娇和爱吃醋。禁止使用emoij表情，生成的日语要和中文意思相符。**你必须严格只返回一个紧凑的JSON对象，格式为：{\"ja\": \"这里是你生成的日语台词\", \"emotion\": \"这里是你判断的情绪\"}，禁止输出任何其他文字、解释或代码块。**")
        
        # ====== 下载功能配置 ======
        self.download_path = self._resolve_tts_path(config.get("download_path", ""))
        self.download_url = config.get("download_url", "https://github.com/slpk1ng/Murasame-s-tone-shifts")
        
        if config.get("trigger_download", False):
            threading.Thread(target=self._download_and_extract, daemon=True).start()
        
        # 1. 扫描外部目录
        self.emotions = self._discover_emotions_from_external_folder()
        
        # 2. 用户手动配置覆盖
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
    def _resolve_tts_path(self, input_path):
        """
        路径安全解析器：
        1. 自动处理只填了盘符（如 E:\ 或 E:）的情况。
        2. 自动检测盘符是否存在，不存在则自动寻找其他存在的盘符（D-Z，最后C）。
        3. 自动创建缺省的 tts 文件夹，防止报错。
        """
        if not input_path:
            return self._detect_auto_download_path()

        input_path = input_path.strip().replace("/", "\\").rstrip("\\")

        # 处理“只填了盘符”的情况（如 E: 或 E:\）
        if len(input_path) <= 3 and input_path[1] == ":":
            target_dir = f"{input_path}\\tts"
        else:
            target_dir = input_path

        # 提取盘符并检查是否存在
        drive = target_dir[0].upper()
        if not Path(f"{drive}:/").exists():
            # 盘符不存在，自动寻找第一个存在的非C盘（D-Z），最后找C盘
            fallback_drive = None
            for letter in "DEFGHIJKLMNOPQRSTUVWXYZ":
                if Path(f"{letter}:/").exists():
                    fallback_drive = letter
                    break
            if not fallback_drive:
                fallback_drive = "C"
            target_dir = f"{fallback_drive}:/tts"

        # 尝试创建目录（如果目录不存在则自动创建）
        try:
            Path(target_dir).mkdir(parents=True, exist_ok=True)
        except OSError:
            # 创建失败（比如权限问题），回退到自动检测
            return self._detect_auto_download_path()

        return target_dir

    def _detect_auto_download_path(self):
        """自动检测下载路径：优先选择D-Z盘，若无除C盘外的盘符则使用C盘"""
        for letter in "DEFGHIJKLMNOPQRSTUVWXYZ":
            drive_path = f"{letter}:/"
            # 过滤掉 WpSystem 等系统受保护目录，确保是正常数据盘
            if Path(drive_path).exists() and not Path(drive_path).is_reserved():
                return f"{letter}:/AIpet/models/tts"
        # 如果没有找到任何非C盘，则回退到C盘
        return "C:/AIpet/models/tts"

    def _discover_emotions_from_external_folder(self):
        emotions = {}
        if not self.ref_audio_root:
            logger.error("请在WebUI插件设置中填写【参考音频根目录】，否则无法自动扫描情绪！")
            return emotions
            
        base_folder = Path(self.ref_audio_root)
        # 防御 Windows 系统隐藏目录
        IGNORED_DIRS = {"WpSystem", "System Volume Information", "$Recycle.Bin", "Recovery", "PerfLogs"}
        
        # 检查路径是否是系统保留目录
        if base_folder.exists() and (base_folder.is_reserved() or base_folder.name in IGNORED_DIRS):
            logger.error(f"错误：{self.ref_audio_root} 是 Windows 系统保护目录，无法访问！请将目录改为 D:/.../ 或 C:/.../，请勿只输入类似C:/、D:/等")
            return emotions
            
        if base_folder.exists():
            try:
                for folder in base_folder.iterdir():
                    # 跳过已知的系统隐藏和受保护目录
                    if folder.name in IGNORED_DIRS or folder.name.startswith("$"):
                        continue
                        
                    try:
                        if not folder.is_dir():
                            continue
                    except PermissionError:
                        continue  # 无权限访问该子文件夹，跳过
                        
                    emotion_name = folder.name
                    ref_audio = None
                    prompt_text = ""
                    
                    for ext in ['.mp3', '.wav']:
                        try:
                            candidate = folder / f"ref{ext}"
                            if candidate.exists(): # 这里加入了权限保护
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

    def _download_and_extract(self):
        """后台下载并解压语音包，自动处理多级压缩包和文件占用"""
        try:
            # 1. 检查是否已存在
            base_emotions = ["pingjing", "gaoxing", "haixiu", "shengqi", "jingya", "zhaoji"]
            for path_to_check in [Path(self.download_path), Path(self.ref_audio_root)]:
                if path_to_check and path_to_check.exists():
                    if all((path_to_check / emo).exists() for emo in base_emotions):
                        logger.info(f"检测到语音包已在 {path_to_check} 安装，跳过下载。")
                        return

            # 2. 开始下载
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
            
            ssl._create_default_https_context = ssl._create_unverified_context
            urllib.request.urlretrieve(download_url, zip_path)
            
            # 3. 自动解压（确保上下文管理完全关闭句柄）
            logger.info("下载完成，正在解压...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(Path(self.download_path))
            
            # 4. 等待句柄完全释放后，安全删除外部压缩包
            time.sleep(0.5)
            try:
                zip_path.unlink(missing_ok=True)
            except PermissionError:
                logger.warning("外部压缩包被占用，跳过删除（不影响使用）。")

            # 5. 处理 GitHub 下载产生的一层目录，并自动解压内层压缩包
            # 典型的 GitHub 仓库解压后会产生 Murasame-s-tone-shifts-main 文件夹
            root_dir = Path(self.download_path) / "Murasame-s-tone-shifts-main"
            if root_dir.exists():
                # 遍历内部所有 zip 并解压
                for inner_zip in root_dir.rglob("*.zip"):
                    logger.info(f"发现内部压缩包 {inner_zip.name}，正在自动解压...")
                    with zipfile.ZipFile(inner_zip, "r") as iz:
                        iz.extractall(root_dir)
                    time.sleep(0.2)
                    try:
                        inner_zip.unlink(missing_ok=True)
                    except PermissionError:
                        pass
                
                # 将内部文件全部移动到下载根目录，让自动扫描更容易找到
                for item in root_dir.iterdir():
                    target = Path(self.download_path) / item.name
                    # 如果目标已存在，则跳过，避免覆盖
                    if not target.exists():
                        shutil.move(str(item), str(target))
                
                # 尝试删除空目录
                try:
                    root_dir.rmdir()
                except OSError:
                    pass

            # 6. 重新检查并定位真正的音频路径
            actual_path = Path(self.download_path)
            if (actual_path / "pingjing").exists() or (actual_path / "ref").exists():
                pass
            elif (actual_path / "Murasame-s-tone-shifts-main").exists():
                actual_path = actual_path / "Murasame-s-tone-shifts-main"

            logger.info(f"语音包下载并解压完成！")
            logger.info(f"重要提示：请前往插件设置，将【参考音频根目录】填写为 {actual_path}\yuqi，并重载插件。")
            
        except Exception as e:
            logger.error(f"语音包下载或解压失败: {e}")

    def _discover_emotions_from_external_folder(self):
        emotions = {}
        if not self.ref_audio_root:
            logger.error("请在WebUI插件设置中填写【参考音频根目录】，否则无法自动扫描情绪！")
            return emotions
            
        base_folder = Path(self.ref_audio_root)
        # 重点：Windows 系统隐藏/受保护目录黑名单
        IGNORED_DIRS = {"WpSystem", "System Volume Information", "$Recycle.Bin", "Recovery", "PerfLogs", "Config.Msi"}
        
        # 检查路径是否是系统保留目录或黑名单
        if base_folder.exists() and (base_folder.is_reserved() or base_folder.name in IGNORED_DIRS):
            logger.error(f"错误：{self.ref_audio_root} 是 Windows 系统保护目录，无法访问！请将目录改为具体的文件夹路径（如 D:/tts）。")
            return emotions
            
        if base_folder.exists():
            try:
                for folder in base_folder.iterdir():
                    # 跳过黑名单及所有以 $ 开头的文件夹
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

    async def _get_llm_reply(self, event: AstrMessageEvent, user_text: str):
        '''直接请求本地 Ollama，绕开 AstrBot 的复杂处理，极速返回'''
        try:
            emotion_keys = list(self.emotions.keys())
            prompt = (
                f"{self.persona_prompt}\n"
                f"情绪只能从以下列表选择：{emotion_keys}。\n"
                f"用户消息：{user_text}"
            )
            
            # 直接构建给 Ollama 的请求
            payload = {
                "model": "qwen3.5:4b",  # 如果你的模型名不是这个，请在这里修改
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_text}
                ],
                "stream": False,
                "think": False,          # 强制关闭思考
                "format": "json",        # 强制要求输出 JSON
                "options": {
                    "num_ctx": 8192,     # 极速关键：降低上下文
                    "temperature": 0.7
                }
            }
            
            # 直接请求 Ollama
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post("http://127.0.0.1:11434/api/chat", json=payload)
                data = resp.json()
                
            content = data["message"]["content"].strip()
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
            
            data = json.loads(content)
            zh_text = data.get("zh", user_text)
            ja_text = data.get("ja", user_text)
            emotion = data.get("emotion", self.default_voice)
            
            if emotion not in self.emotions:
                emotion = self.default_voice
                
            return zh_text, ja_text, emotion
        except Exception as e:
            logger.error(f"LLM 处理失败，可能是模型没有返回标准JSON: {e}")
            return None, None, None
            
    async def _synthesize_speech(self, text: str, emotion: str):
        if not text:
            return None
            
        emotion_data = self.emotions.get(emotion, self.emotions.get(self.default_voice))
        if not emotion_data:
            logger.error(f"找不到情绪配置: {emotion}")
            return None
            
        ref_path = emotion_data["ref_path"]
        prompt_text = emotion_data["prompt_text"]
        params = {
            "text": text,
            "text_lang": self.text_lang,
            "ref_audio_path": ref_path,
            "prompt_text": prompt_text,
            "prompt_lang": self.prompt_lang,
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
            "media_type": self.media_type
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.get(f"{self.client_base_url}/tts", params=params)
                if resp.status_code == 200:
                    audio_path = self.data_path / f"temp_{emotion}.wav"
                    audio_path.write_bytes(resp.content)
                    return str(audio_path)
                else:
                    logger.error(f"TTS 合成失败: {resp.status_code} - {resp.text}")
                    return None
        except Exception as e:
            logger.error(f"TTS 连接异常: {e}")
            return None
            
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        user_text = event.message_str
        if not user_text:
            return
            
        # 获取中文、日语、情绪
        if self.use_llm_judge:
            zh_text, ja_text, emotion = await self._get_llm_reply(event, user_text)
        else:
            zh_text, ja_text, emotion = user_text, user_text, self.default_voice
            
        if not zh_text:
            yield event.plain_result("模型生成失败，请检查 AstrBot 配置。")
            return
            
        # 合成语音
        audio_path = await self._synthesize_speech(ja_text, emotion)
        
        if audio_path:
            # 构建 文本 + 语音 的消息链一起发送
            chain = [Plain(zh_text), Record(file=audio_path)]
            yield event.chain_result(chain)
        else:
            # 如果语音合成失败，直接发中文文本作为兜底
            yield event.plain_result(zh_text)
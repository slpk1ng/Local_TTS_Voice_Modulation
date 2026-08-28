import asyncio
import os
import time
import base64
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

import psutil
from PIL import Image, ImageChops
import numpy as np
import mss
import astrbot.api.message_components as Comp
from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult, MessageChain
from astrbot.api.star import Context, Star
from astrbot.api import logger
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

# 兼容不同版本：尝试导入 ImagePart，失败则降级为纯文本
try:
    from astrbot.core.agent.message import TextPart, UserMessageSegment, ImagePart
except ImportError:
    from astrbot.core.agent.message import TextPart, UserMessageSegment
    ImagePart = None

# ------------------------------------------------------------------
# 跨平台 GPU 监控模块
# ------------------------------------------------------------------
class GPUManager:
    """自动检测并初始化显卡监控库"""
    def __init__(self):
        self.vendor = "NONE"
        self.nvml_handle = None
        self.amd_handle = None
        self.intel_dev = None

        self._try_init_nvidia()
        self._try_init_amd()
        self._try_init_intel()

    def _try_init_nvidia(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.vendor = "NVIDIA"
            logger.info("已检测到 NVIDIA GPU，使用 pynvml 监控")
        except Exception:
            pass

    def _try_init_amd(self):
        try:
            import amdsmi
            amdsmi.amdsmi_init()
            devices = amdsmi.amdsmi_get_processor_handles()
            if len(devices) > 0:
                self.amd_handle = devices[0]
                self.vendor = "AMD"
                logger.info("已检测到 AMD GPU，使用 amdsmi 监控")
        except Exception:
            pass

    def _try_init_intel(self):
        try:
            import pyzes
            os.environ.setdefault("ZES_ENABLE_SYSMAN", "1")
            self.intel_dev = True
            self.vendor = "INTEL"
            logger.info("已检测到 Intel GPU，使用 Level Zero Sysman 监控")
        except Exception:
            pass

    def get_gpu_utilization(self) -> Optional[float]:
        """返回 GPU 使用率百分比，失败返回 None"""
        if self.vendor == "NVIDIA" and self.nvml_handle:
            try:
                import pynvml
                return pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle).gpu
            except Exception:
                return None
        if self.vendor == "AMD" and self.amd_handle:
            try:
                import amdsmi
                return amdsmi.amdsmi_get_gpu_activity(self.amd_handle)["gfx_activity"]
            except Exception:
                return None
        if self.vendor == "INTEL" and self.intel_dev:
            return None
        return None


class ScreenMonitorPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.plugin_name = self.name
        self.monitor_task: Optional[asyncio.Task] = None
        self._last_event_time = 0
        self._is_recording = False
        self._screenshots: List[Path] = []
        self._prev_screen = None
        self._storage_dir = Path(config.get("storage_dir", "")) if config.get("storage_dir") else Path(get_astrbot_plugin_data_path()) / self.plugin_name
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._finish_lock = asyncio.Lock()
        self._low_load_since = None
        self._process_absent_since = None
        self.gpu_manager = GPUManager()
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        self._pending_start_time = None
        self._potential_event_detected = False
        self._manual_recording = False
        self.is_ollama = self.config.get("llm_backend", "openai").lower() == "ollama"

    async def terminate(self):
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self):
        try:
            while True:
                try:
                    # 读取配置：进程列表（兼容字符串或列表）
                    process_names_config = self.config.get("process_name", [])
                    if isinstance(process_names_config, str):
                        process_names_config = [process_names_config]

                    is_process_running = False
                    cpu_percent = 0.0
                    gpu_percent = 0.0
                    high_load = False

                    # ========== 确定当前是否使用“进程模式” ==========
                    use_process_mode = False
                    if process_names_config:
                        # 检查指定的进程是否在运行
                        for proc in psutil.process_iter(['name']):
                            try:
                                proc_name = proc.info['name']
                                if proc_name and any(proc_name.lower() == p.lower() for p in process_names_config):
                                    is_process_running = True
                                    break
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                continue

                        # 如果配置了进程但进程未运行，且启用了回退，则切换到 CPU/GPU 模式
                        fallback_enabled = self.config.get("enable_process_fallback", True)
                        if not is_process_running and fallback_enabled:
                            use_process_mode = False
                        else:
                            use_process_mode = True
                    else:
                        # 未配置进程，直接使用 CPU/GPU 模式
                        use_process_mode = False

                    # ========== 如果是 CPU/GPU 模式，计算负载 ==========
                    if not use_process_mode:
                        cpu_percent = psutil.cpu_percent(interval=None)
                        gpu_percent = self.gpu_manager.get_gpu_utilization() if self.config.get("enable_gpu", False) else 0.0

                        # 解析 CPU 阈值区间
                        cpu_min, cpu_max = self._parse_range(
                            self.config.get("cpu_threshold", "50-70"),
                            default_min=50,
                            default_max=70
                        )

                        # 读取高负载判定模式（默认 or）
                        load_mode = self.config.get("high_load_mode", "or").lower()

                        if load_mode == "and":
                            high_load = (cpu_min <= cpu_percent <= cpu_max)
                            if self.config.get("enable_gpu", False) and gpu_percent is not None:
                                gpu_min, gpu_max = self._parse_range(
                                    self.config.get("gpu_threshold", "50-70"),
                                    default_min=50,
                                    default_max=70
                                )
                                high_load = high_load and (gpu_min <= gpu_percent <= gpu_max)
                        else:
                            # 默认 OR 模式：CPU 或 GPU 任一满足即触发
                            high_load = (cpu_min <= cpu_percent <= cpu_max)
                            if self.config.get("enable_gpu", False) and gpu_percent is not None:
                                gpu_min, gpu_max = self._parse_range(
                                    self.config.get("gpu_threshold", "50-70"),
                                    default_min=50,
                                    default_max=70
                                )
                                high_load = high_load or (gpu_min <= gpu_percent <= gpu_max)

                    # ========== 决定是否应该开始录制 ==========
                    should_start = False
                    if use_process_mode:
                        should_start = is_process_running
                    else:
                        # 高负载且屏幕有变化时，标记潜在事件
                        screen_changed = False
                        if high_load:
                            screen = self._capture_screen()
                            if screen is not None:
                                if self._prev_screen is None:
                                    screen_changed = True
                                else:
                                    diff_ratio = self._image_diff_ratio(self._prev_screen, screen)
                                    if diff_ratio >= self.config.get("screen_change_threshold", 5.0):
                                        screen_changed = True
                                self._prev_screen = screen
                        if high_load and screen_changed:
                            should_start = True

                    # ========== 处理录制中的状态 ==========
                    if self._is_recording:
                        # 手动录制：忽略自动结束条件，一直录制直到用户停止
                        if self._manual_recording:
                            if self._screenshots:
                                last_shot = self._screenshots[-1]
                                if time.time() - last_shot.stat().st_mtime >= self.config.get("screenshot_interval", 5.0):
                                    self._take_screenshot()
                            else:
                                self._take_screenshot()
                        else:
                            end_condition = False

                            if use_process_mode:  # 使用 use_process_mode
                                # 进程模式：检测进程是否退出
                                if not is_process_running:
                                    if self._process_absent_since is None:
                                        self._process_absent_since = time.time()
                                    elif time.time() - self._process_absent_since >= self.config.get("process_end_duration", 5):
                                        end_condition = True
                                else:
                                    self._process_absent_since = None
                            else:
                                # CPU/GPU 模式：检测是否进入低负载区间
                                low_cpu_min, low_cpu_max = self._parse_range(
                                    self.config.get("low_cpu_threshold", "0-30"),
                                    default_min=0,
                                    default_max=30
                                )
                                low_cpu = low_cpu_min <= cpu_percent <= low_cpu_max

                                low_gpu = True  # 默认认为GPU低负载（未启用GPU监控时忽略）
                                if self.config.get("enable_gpu", False) and gpu_percent is not None:
                                    low_gpu_min, low_gpu_max = self._parse_range(
                                        self.config.get("low_gpu_threshold", "0-30"),
                                        default_min=0,
                                        default_max=30
                                    )
                                    low_gpu = low_gpu_min <= gpu_percent <= low_gpu_max

                                # 低负载判定：CPU和GPU必须同时处于低区间（AND逻辑）
                                low_load = low_cpu and low_gpu

                                if low_load:
                                    if self._low_load_since is None:
                                        self._low_load_since = time.time()
                                    elif time.time() - self._low_load_since >= self.config.get("low_load_duration", 15):
                                        end_condition = True
                                else:
                                    self._low_load_since = None

                            if end_condition:
                                await self._finish_recording()
                                self._low_load_since = None
                                self._process_absent_since = None
                            else:
                                # 未到结束条件，继续截图
                                if self._screenshots:
                                    last_shot = self._screenshots[-1]
                                    if time.time() - last_shot.stat().st_mtime >= self.config.get("screenshot_interval", 5.0):
                                        self._take_screenshot()
                                else:
                                    self._take_screenshot()

                    else:
                        # ========== 非录制状态，处理启动条件 ==========
                        now = time.time()
                        if self._pending_start_time is not None:
                            # 正在等待启动
                            if should_start:  # 改为判断 should_start（进程运行或高负载+屏幕变化）
                                if now - self._pending_start_time >= self.config.get("process_start_duration", 5):
                                    self._pending_start_time = None
                                    await self._start_recording()
                                    self._low_load_since = None
                                    self._process_absent_since = None
                                    self._potential_event_detected = False
                            else:
                                # 条件不满足，取消等待
                                self._pending_start_time = None
                                self._potential_event_detected = False
                        else:
                            # 没有在等待
                            if should_start:
                                if now - self._last_event_time >= self.config.get("cooldown", 300):
                                    self._pending_start_time = now
                                    self._potential_event_detected = True
                                    wait_time = self.config.get("process_start_duration", 5)
                                    logger.info(f"检测到潜在事件，等待 {wait_time} 秒后开始记录...")
                            else:
                                self._potential_event_detected = False

                    # 循环间隔
                    await asyncio.sleep(self.config.get("check_interval", 2.0))

                except Exception as e:
                    logger.error(f"监控循环异常: {e}")
                    await asyncio.sleep(5)

        except asyncio.CancelledError:
            logger.info("监控任务已取消")
            raise

    @staticmethod
    def _parse_resolution(value: str) -> Optional[tuple]:
        mapping = {
            "原始": None,
            "4k": (3840, 2160),
            "2k": (2560, 1440),
            "1080p": (1920, 1080),
            "720p": (1280, 720),
            "480p": (854, 480)
        }
        key = str(value).strip().lower()
        if key in mapping:
            return mapping[key]
        if "x" in key:
            try:
                w, h = key.split("x")
                return (int(w), int(h))
            except (ValueError, IndexError):
                return None
        return None

    def _capture_screen(self) -> Optional[Image.Image]:
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                shot = sct.grab(monitor)
                return Image.frombytes("RGB", shot.size, shot.rgb)
        except Exception as e:
            logger.error(f"截图失败: {e}")
            return None

    def _image_diff_ratio(self, img1: Image.Image, img2: Image.Image) -> float:
        try:
            if img1.size != img2.size:
                return 100.0
            diff = ImageChops.difference(img1, img2)
            arr = np.array(diff)
            return np.count_nonzero(arr) / arr.size * 100
        except Exception:
            return 0.0

    def _is_meaningful_image(self, img: Image.Image) -> bool:
        """
        检测图片是否包含有效内容，用于过滤黑屏、纯色等无意义截图。
        返回 True 表示有内容，False 表示无意义。
        """
        try:
            # 缩小图片以加快计算速度（例如缩至100x100）
            small = img.resize((100, 100))
            arr = np.array(small)

            # 1. 黑屏检测：平均亮度极低
            mean_brightness = arr.mean()
            if mean_brightness < 10.0:
                return False

            # 2. 纯色/低信息量检测：像素标准差极低（说明颜色基本一致）
            std_dev = arr.std()
            if std_dev < 5.0:
                return False

            return True
        except Exception:
            return True  # 如果检测出错，不拦截，保留图片

    def _take_screenshot(self):
        try:
            img = self._capture_screen()
            if img is None:
                return

            # 解析目标分辨率
            target_res = self._parse_resolution(self.config.get("screenshot_resolution", "原始"))
            if target_res is not None:
                target_width, target_height = target_res
                # 仅当原始尺寸大于目标时才缩放
                if img.width > target_width or img.height > target_height:
                    ratio = min(target_width / img.width, target_height / img.height)
                    new_width = int(img.width * ratio)
                    new_height = int(img.height * ratio)
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                    logger.info(f"截图已缩放: {img.width}x{img.height} -> {new_width}x{new_height}")

            # 过滤黑屏/纯色截图（新增逻辑）
            if self.config.get("ignore_meaningless_screenshots", True) and not self._is_meaningful_image(img):
                logger.info("跳过无意义截图（黑屏/纯色）")
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            # 强制将 quality 转换为整数，并处理异常
            try:
                quality = int(self.config.get("screenshot_quality", 85))
            except (ValueError, TypeError):
                quality = 85
                logger.warning("screenshot_quality 配置无效，已回退为 85")

            if 0 < quality < 100:
                path = self._storage_dir / f"event_{timestamp}.jpg"
                img.save(path, "JPEG", quality=quality)
            else:
                path = self._storage_dir / f"event_{timestamp}.png"
                img.save(path)
            self._screenshots.append(path)
            logger.info(f"已保存截图: {path}")
        except Exception as e:
            logger.error(f"保存截图失败: {e}")

    async def _start_recording(self):
        self._is_recording = True
        self._screenshots = []
        self._take_screenshot()
        logger.info("开始记录屏幕事件")

    def _clear_screenshots(self):
        for img in self._screenshots:
            try:
                img.unlink(missing_ok=True)
            except Exception:
                pass
        self._screenshots = []

    async def _finish_recording(self):
        if self._finish_lock.locked():
            return
        async with self._finish_lock:
            self._is_recording = False
            self._manual_recording = False
            self._last_event_time = time.time()
            logger.info(f"结束记录，共 {len(self._screenshots)} 张截图")
            if not self._screenshots:
                return
            target_umo = self.config.get("target_umo", "")
            if not target_umo:
                logger.warning("未配置目标 UMO，无法发送消息")
                self._clear_screenshots()
                return
            if target_umo.count(':') < 2:
                logger.warning("目标 UMO 格式不正确，应为 platform:message_type:session_id，忽略发送")
                self._clear_screenshots()
                return
            try:
                max_images = int(self.config.get("max_images", 3))
                if len(self._screenshots) > max_images:
                    indices = np.linspace(0, len(self._screenshots) - 1, max_images).astype(int)
                    selected = [self._screenshots[i] for i in indices]
                else:
                    selected = self._screenshots

                logger.info(f"选中的图片路径: {[str(p) for p in selected]}")

                send_each_image_separately = self.config.get("send_each_image_separately", True)

                if send_each_image_separately:
                    previous_description = ""  # 记录上次生成的描述
                    for img_path in selected:
                        if not img_path.exists():
                            logger.warning(f"截图文件不存在，跳过: {img_path}")
                            continue
                        # 只传当前图片，并传入之前的描述作为上下文
                        description = await self._generate_description(
                            [img_path], 
                            focus_img=img_path, 
                            prev_desc=previous_description
                        )
                        # 构建消息链并发送
                        chain = MessageChain()
                        if self.config.get("send_images", True):
                            chain.file_image(str(img_path))
                        chain.message(description)
                        await self.context.send_message(target_umo, chain)
                        logger.info(f"已发送图片 {img_path.name} 及其描述")
                        # 更新描述，供下张图片使用
                        previous_description = description
                else:
                    # 合并发送所有图片
                    chain = MessageChain()
                    if self.config.get("send_images", True):
                        for img_path in selected:
                            if not img_path.exists():
                                logger.warning(f"截图文件不存在，跳过: {img_path}")
                                continue
                            chain.file_image(str(img_path))
                    description = await self._generate_description(selected)
                    chain.message(description)
                    if len(chain.chain) == 0:
                        chain = MessageChain().message("（没有可发送的内容）")
                    await self.context.send_message(target_umo, chain)
                    logger.info("事件描述已发送")

            except Exception as e:
                logger.error(f"发送消息失败: {e}")
            finally:
                self._clear_screenshots()

    @staticmethod
    def _parse_range(value, default_min: float, default_max: float) -> tuple:
        if value is None:
            return default_min, default_max
        if isinstance(value, (int, float)):
            return float(value), float(value)
        if isinstance(value, (list, tuple)):
            vals = [float(x) for x in value[:2]]
            if len(vals) == 1:
                return vals[0], vals[0]
            return min(vals[0], vals[1]), max(vals[0], vals[1])
        text = str(value).strip()
        parts = None
        for sep in ['~', '-', ',', ' ', '\t', '\n']:
            if sep in text:
                parts = [p.strip() for p in text.split(sep) if p.strip() != '']
                if len(parts) >= 2:
                    break
        if parts and len(parts) >= 2:
            try:
                first = float(parts[0])
                second = float(parts[1])
                return min(first, second), max(first, second)
            except ValueError:
                pass
        try:
            single = float(text)
            return single, single
        except ValueError:
            return default_min, default_max

    async def _generate_description(self, image_paths: List[Path], focus_img: Optional[Path] = None, prev_desc: str = "") -> str:
        """
        根据配置生成事件描述。
        :param image_paths: 本次请求使用的图片列表（通常只传当前图片）
        :param focus_img: 当前要重点描述的图片
        :param prev_desc: 上一次生成的描述（作为上下文）
        """
        persona_id = self.config.get("persona", "")
        persona_prompt = ""
        if persona_id:
            try:
                persona_obj = await self.context.persona_manager.get_persona(persona_id)
                if persona_obj:
                    persona_prompt = persona_obj.system_prompt
            except Exception as e:
                logger.error(f"获取人格 {persona_id} 失败: {e}")
        if not persona_prompt:
            persona_prompt = self.config.get("persona", "你是一个观察者，用幽默的口吻描述发生的事情。")

        return await self._generate_description_direct(image_paths, persona_prompt, focus_img, prev_desc)

    async def _generate_description_direct(self, image_paths: List[Path], persona_prompt: str, focus_img: Optional[Path] = None, prev_desc: str = "") -> str:
        import base64
        import httpx

        llm_backend = self.config.get("llm_backend", "openai").lower()
        is_ollama = (llm_backend == "ollama")
        
        # 读取超时时间配置（默认120秒）
        timeout = self.config.get("request_timeout", 120)

        if is_ollama:
            api_base = self.config.get("ollama_api_url", "http://127.0.0.1:11434").rstrip("/")
            api_key = ""
            model_name = self.config.get("model_name", "").strip()
            if not model_name:
                logger.error("未指定 Ollama 模型名称，请在插件配置中添加 model_name")
                return "（Ollama 模型名称未配置）"
        else:
            api_base = self.config.get("api_base_url", "").strip()
            api_key = self.config.get("api_key", "").strip()
            model_name = self.config.get("model_name", "").strip()
            if not api_base:
                logger.error("未配置 API 地址（api_base_url），请在插件配置中添加")
                return "（API 地址未配置）"
            if not model_name:
                logger.error("未配置模型名称（model_name），请在插件配置中添加")
                return "（模型名称未配置）"

        # 构建基础提示词
        prompt_template = self.config.get("prompt_template", "")
        if not prompt_template:
            prompt_template = "请你以观察者的视角（用户正在操作电脑，你需要观察用户，而不是想象自己在操控电脑！），仔细观察这些连续截图，用符合以下人设的语气简要地说出来，不得长篇大论。\n【人设】{persona}"
        prompt_text = prompt_template.replace("{persona}", persona_prompt)

        # 防止复读的核心修改：不直接回传旧句子，而是强调“只看新变化，严禁重复原话”
        if prev_desc:
            prompt_text += "\n\n【前情摘要】上一张截图大概发生了什么，绝对禁止生硬地复述图片！可以用你的人设来想象你如果是ta的话，ta现在会对用户说些什么。"
            prompt_text += "\n【严格要求】必须一句话直击核心，不得重复、生硬地描述图片的所有内容，找关键点来和用户复盘和之前的截图对比，这张新的截图有什么进展！"
        
        if focus_img is not None:
            prompt_text += f"\n\n这可能是同一事件的新截图（{focus_img.name}），请重点描述此张图片展现的进展和变化，语言必须精炼，杜绝啰嗦复读。"

        # 构建图片内容（只包含传入的图片，通常只有1张）
        content_parts = []
        images = []
        for img_path in image_paths:
            try:
                img_bytes = img_path.read_bytes()
                b64 = base64.b64encode(img_bytes).decode()
                if is_ollama:
                    images.append(b64)
                else:
                    mime = "image/jpeg" if img_path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
                    data_uri = f"data:{mime};base64,{b64}"
                    content_parts.append({"type": "image_url", "image_url": {"url": data_uri}})
            except Exception as e:
                logger.error(f"图片编码失败 {img_path}: {e}")

        if is_ollama:
            messages = [{"role": "user", "content": prompt_text, "images": images}]
        else:
            content_parts.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content_parts}]

        # 构造 endpoint
        if is_ollama:
            endpoint = api_base.rstrip("/") + "/api/chat"
            headers = {}
        else:
            api_base_clean = api_base.rstrip("/")
            if not api_base_clean.endswith("/v1"):
                endpoint = api_base_clean + "/v1/chat/completions"
            else:
                endpoint = api_base_clean + "/chat/completions"
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

        max_retries = 5
        last_error = ""

        for attempt in range(max_retries):
            try:
                logger.info(f"请求地址: {endpoint}")
                logger.info(f"请求模型: {model_name}")

                async with httpx.AsyncClient(timeout=timeout) as client:
                    if is_ollama:
                        # 获取深度思考配置
                        enable_deep_think = self.config.get("enable_deep_think", True)
                        # 添加 repeat_penalty 参数（Ollama 专属），1.3 倍能有效防止复读
                        body = {
                            "model": model_name,
                            "messages": messages,
                            "stream": False,
                            "think": enable_deep_think,  # 启用深度思考
                            "options": {"repeat_penalty": 1.3}
                        }
                    else:
                        # OpenAI 兼容 API 使用 frequency_penalty 防止复读
                        body = {"model": model_name, "messages": messages, "frequency_penalty": 0.9}

                    resp = await client.post(endpoint, json=body, headers=headers)

                    if resp.status_code != 200:
                        error_msg = resp.text if resp.text else "响应体为空"
                        logger.warning(f"第 {attempt+1} 次请求失败: 状态码 {resp.status_code}, 错误信息: {error_msg[:200]}")
                        last_error = f"{resp.status_code} - {error_msg[:200]}"
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2)
                            continue
                        else:
                            return f"（API错误：{last_error}）"

                    data = resp.json()
                    if is_ollama:
                        content = data.get("message", {}).get("content", "")
                        # 获取思考过程（如果有），仅输出到日志，不包含在最终回复中
                        thinking = data.get("message", {}).get("thinking", "")
                        if thinking:
                            logger.info(f"模型思考过程: {thinking}")
                    else:
                        content = data["choices"][0]["message"]["content"]

                    logger.info(f"直连 API 返回：{content[:100]}")
                    return content.strip() if content else "（未获取到描述）"

            except httpx.TimeoutException as e:
                logger.warning(f"第 {attempt+1} 次请求超时: {e}。超时时间设置为 {timeout} 秒，若仍超时请在配置中调大 request_timeout")
                last_error = f"timeout: {e}"
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                else:
                    return f"（直连 API 超时，请检查服务是否响应或调大 request_timeout）"
            except httpx.HTTPStatusError as e:
                response_text = e.response.text if e.response else "无响应"
                logger.warning(f"第 {attempt+1} 次请求 HTTP 错误: {e}, 响应内容: {response_text[:200]}")
                last_error = f"{e} - {response_text[:200]}"
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                else:
                    return f"（直连 API HTTP 错误：{last_error}）"
            except Exception as e:
                logger.warning(f"第 {attempt+1} 次请求异常: {type(e).__name__}: {e}")
                last_error = f"{type(e).__name__}: {e}"
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                else:
                    return f"（直连 API 失败：{last_error}）"

        return "（直连 API 多次尝试后仍然失败）"


    # ---------- 指令 ----------
    @filter.command("am_help")
    async def am_help(self, event: AstrMessageEvent):
        """查看所有可用命令"""
        help_text = (
            "📋 **Aftermath 插件命令列表**\n\n"
            "---\n"
            "`/umo` — 获取当前会话的 UMO，用于配置 target_umo。\n"
            "`/am_status` — 查看当前监控状态、启用的 GPU 类型、已记录截图数。\n"
            "`/am_test` — 手动触发一次录制（每 5 秒截一张，共 4 张，然后发送）。\n"
            "`/am_clear` — 清除当前会话的记忆，防止旧话题干扰。\n"
            "`/am_start` — 手动开始持续录制，直到输入 `/am_stop` 停止。\n"
            "`/am_stop` — 停止录制，将截图发送给 LLM 并返回消息。\n"
        )
        yield event.plain_result(help_text)

    @filter.command("am_status")
    async def am_status(self, event: AstrMessageEvent):
        """查看监控状态"""
        recording = "正在记录" if self._is_recording else "空闲"
        gpu_vendor = self.gpu_manager.vendor if self.config.get("enable_gpu", False) else "未启用"
        yield event.plain_result(f"当前状态: {recording}\n已记录截图数: {len(self._screenshots)}\nGPU监控: {gpu_vendor}")

    @filter.command("umo")
    async def umo(self, event: AstrMessageEvent):
        '''获取当前会话的 UMO'''
        yield event.plain_result(f"当前会话 UMO 为: {event.unified_msg_origin}")

    @filter.command("am_start")
    async def am_start(self, event: AstrMessageEvent):
        """手动开始持续录制，直到 /am_stop 停止"""
        if self._is_recording:
            yield event.plain_result("已经在录制中，无需重复启动。")
            return
        self._manual_recording = True
        self._is_recording = True
        self._screenshots = []
        self._take_screenshot()
        yield event.plain_result("手动录制已开始，将按截图间隔持续截图，直到发送 /am_stop 停止。")

    @filter.command("am_stop")
    async def am_stop(self, event: AstrMessageEvent):
        """手动停止录制并发送消息"""
        if not self._is_recording:
            yield event.plain_result("当前没有录制任务。")
            return
        self._manual_recording = False
        await self._finish_recording()  # 该方法会发送消息并清空截图列表
        yield event.plain_result("手动录制已停止，事件描述已发送。")

    @filter.command("am_test")
    async def am_test_cmd(self, event: AstrMessageEvent):
        """手动触发一次录制：每5秒截一张图，共4次，然后发送给LLM"""
        asyncio.create_task(self._run_am_test())
        yield event.plain_result("已开始手动录制：每5秒截一张图，共4次，完成后发送。")

    async def _run_am_test(self):
        self._is_recording = True
        self._screenshots = []
        for _ in range(4):
            self._take_screenshot()
            await asyncio.sleep(5)
        self._is_recording = False
        await self._finish_recording()

    @filter.command("am_clear")
    async def am_clear(self, event: AstrMessageEvent):
        """清除当前会话的记忆，防止旧话题干扰"""
        try:
            umo = event.unified_msg_origin
            if not umo:
                yield event.plain_result("无法获取当前会话标识。")
                return
            conv_mgr = self.context.conversation_manager
            new_cid = await conv_mgr.new_conversation(umo)
            yield event.plain_result(
                f"✅ 已为新模型清空对话历史 (新对话ID: {new_cid})。\n"
                "⚠️ 为防止其他模块误读旧消息，建议您手动清除平台历史消息缓存：\n"
                "在 AstrBot 数据目录中执行：\n"
                "sqlite3 data/data_v4.db \"DELETE FROM platform_message_history WHERE unified_msg_origin='{umo}';\"\n"
                "（如果字段名不同，请查看实际表结构）"
            )
            logger.info(f"已清除会话记忆: {umo}")
        except Exception as e:
            logger.error(f"清除记忆失败: {e}")
            yield event.plain_result(f"清除记忆失败: {e}")
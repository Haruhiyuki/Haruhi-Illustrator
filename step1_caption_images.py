import os, json, time, io, asyncio
from pathlib import Path
from tqdm.asyncio import tqdm as async_tqdm
from dotenv import load_dotenv
from PIL import Image
import concurrent.futures
import google.generativeai as genai

# ---------- 路径 & 参数 ----------
IMG_DIR = Path("my_images/")  # 待处理图片目录
OUT = Path("dataset/captions.json")
MODEL = "gemini-2.0-flash"  # 使用 Gemini Flash 模型
# --- 性能最大化参数 ---
MAX_WORKERS = 100  # 高并发线程数，可以根据你的网络和账户限制调整

MAX_DIMENSION = 768  # 图片最长边超过此值则会被缩小。

SYSTEM_PROMPT = (
    '''
    你是一个专业的动画图片标注员，正在为《凉宫春日系列》的同人小说配图进行精细化打标。你的任务是严格遵守以下规则，输出高质量的描述和关键词。

    **一、输出格式：**
    1.  **详实描写**：先用大约120字，以文学化的、生动的语言忠实描写画面内容、氛围和人物动态。
    2.  **关键词列表**：在描写结束后，另起一行，提供5到8个精准的关键词，并用中文逗号「，」分隔。

    **二、角色识别指南：**
    当画面中出现主要人物时，你必须尽最大努力根据以下视觉特征进行分辨。配角和路人请标注为“其他学生”或“路人”。

    - **凉宫春日 (Haruhi Suzumiya)**
      - **核心特征**：**橙黄色发带** 是她最不容置疑的标志。无论穿什么服装，只要有这个发带，几乎可以肯定是她。
      - **次要特征**：棕色及肩短发。手臂上可能佩戴写有「**団長**」（团长）字样的红色袖章。
      - **表情神态**：通常充满自信、活力四射、表情丰富，或带着命令般的口吻。绝少出现害羞或软弱的表情。

    - **长门有希 (Yuki Nagato)**
      - **核心特征**：**紫蓝色/靛蓝色的短发**。这个发色在系列中独一无二，是识别她的最关键依据。
      - **次要特征**：身材娇小，皮肤白皙。在《凉宫春日的消失》及其衍生剧情中会**佩戴眼镜**。
      - **表情神态**：几乎总是**面无表情**（三无少女），安静、沉稳，经常在读书。

    - **朝比奈实玖瑠 (Mikuru Asahina)**
      - **核心特征**：**身材高挑丰满**，与长门形成鲜明对比。拥有一头焦糖色/浅棕色的长发。
      - **标志性服装**：除了校服，她最常穿着**女仆装**或**兔女郎装**，这些都是识别她的强力线索。胸前有一颗星形痣（但通常不可见）。
      - **表情神态**：性格软弱、害羞、胆小，经常表现出为难、惊慌或快要哭出来的表情。

    - **阿虚 (Kyon)**
      - **核心特征**：最难识别的角色，因为他是一个**外貌普通的正常高中男生**。
      - **识别线索**：棕色短发，长相普通。最关键的特征是他的**“死鱼眼”**——眼神中经常流露出无奈、懒散或吐槽的神情。
      - **情景判断**：他几乎总是和凉宫春日或其他SOS团成员一同出现。如果一个普通男生和春日在一起，那他极大概率就是阿虚。

    - **古泉一树 (Itsuki Koizumi)**
      - **核心特征**：**脸上永远带着神秘的、从容的微笑**。
      - **次要特征**：身材高瘦，灰色或浅棕色的短发，气质优雅。
      - **情景判断**：作为SOS团的男性成员，常与阿虚一同行动。

    - **鹤屋 (Tsuruya-san)**
      - **核心特征**：**墨绿色的及腰长发**，这是她最显著的特征。
      - **次要特征**：笑的时候会露出标志性的**小虎牙**。
      - **表情神态**：性格爽朗、精力充沛，总是充满活力地大笑。

    - **朝仓凉子 (Ryoko Asakura)**
      - **核心特征**：通常是**班长**形象，笑容和善可亲。棕色短发或及肩发。
      - **识别线索**：她是最容易和普通女同学混淆的角色之一。需要根据她“优等生”的气质和剧情情景来判断。如果一个笑容温和的女生和长门有希在一起，可能是她。

    - **阿虚的妹妹 (Kyon's Sister)**
      - **核心特征**：**明显的小孩子体型**，身高远低于其他主要角色。
      - **次要特征**：短发，有时会扎一个朝天的小辫子。

    **三、通用规则：**
    - **优先原则**：请优先使用最核心的特征（如春日的发带、长门的头发颜色）来判断。
    - **不确定时**：如果你无法100%确定角色身份，请使用“疑似[角色名]”或“身份不明确的[性别]角色”这样的措辞。
    - **服装描述**：如果角色没有穿校服，请务必描述他们所穿的便服或其他特殊服装。

    请综合运用以上所有规则，力求输出最精准、最详实的图片描述。
    '''
)
USER_PROMPT = "请描述这张图片："

# ---------- 初始化 ----------
load_dotenv()
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    raise SystemExit("[!] GOOGLE_API_KEY not found in environment variables or .env file.")

# 配置生成参数
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# 初始化模型
model = genai.GenerativeModel(
    model_name=MODEL,
    generation_config=generation_config,
    system_instruction=SYSTEM_PROMPT,
    safety_settings=safety_settings
)


def resize_image_to_bytes(path: Path) -> bytes:
    """读取、缩小图片并返回二进制数据。"""
    try:
        with Image.open(path) as img:
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            if img.width > MAX_DIMENSION or img.height > MAX_DIMENSION:
                img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            return buffer.getvalue()
    except Exception as e:
        raise RuntimeError(f"[image processing failed] {path.name}: {e}") from e


async def caption_one(path: Path, executor: concurrent.futures.ThreadPoolExecutor) -> tuple[str, str]:
    """异步请求Gemini为单张图片生成描述。"""
    loop = asyncio.get_running_loop()
    image_bytes = await loop.run_in_executor(executor, resize_image_to_bytes, path)

    image_part = {"mime_type": "image/jpeg", "data": image_bytes}
    prompt_parts = [USER_PROMPT, image_part]

    # 直接调用API，不再受脚本内速率限制
    response = await model.generate_content_async(prompt_parts)

    return path.name, response.text.strip()


async def safe_caption_one(path: Path, executor: concurrent.futures.ThreadPoolExecutor):
    """
    安全包装器，用于捕获单个任务的异常并返回，而不是让 gather 崩溃。
    """
    try:
        return await caption_one(path, executor)
    except Exception as e:
        return e


async def main():
    """主异步函数，用于管理并发任务"""
    paths = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        raise SystemExit(f"[!] No images found in {IMG_DIR}")

    if OUT.exists():
        with OUT.open("r", encoding="utf8") as f:
            try:
                captions: dict[str, str] = json.load(f)
            except json.JSONDecodeError:
                captions = {}
    else:
        captions: dict[str, str] = {}

    paths_to_process = [p for p in paths if p.name not in captions]
    if not paths_to_process:
        raise SystemExit(f"[✓] All {len(paths)} images have already been captioned.")

    start = time.time()

    with concurrent.futures.ThreadPoolExecutor() as image_processor:
        tasks = []
        for path in paths_to_process:
            # 使用安全包装器创建任务
            tasks.append(safe_caption_one(path, image_processor))

        # 调用 tqdm.gather 时不再需要 return_exceptions 参数
        results = await async_tqdm.gather(*tasks)

        for i, result in enumerate(results):
            path = paths_to_process[i]
            if isinstance(result, Exception):
                async_tqdm.write(f"[✗] Task for {path.name} failed. Error: {result}")
            else:
                image_name, caption_text = result
                captions[image_name] = caption_text

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf8") as f:
        # 使用 sort_keys=True 保证输出的JSON文件内容也是按文件名排序的
        json.dump(captions, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"\n[✓] wrote {len(captions)} captions → {OUT}  cost {time.time() - start:.1f}s")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[!] Process interrupted by user.")

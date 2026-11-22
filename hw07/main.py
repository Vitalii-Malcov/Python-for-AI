from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
OUT_DIR = Path(__file__).parent
SEED = 42  # фиксированный seed — результат воспроизводим

# Локальная генерация на CPU, float32 (на CPU half-точность не поддерживается)
pipe = StableDiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.float32)
pipe = pipe.to("cpu")

# Общий negative prompt — убирает типичные артефакты генерации
NEGATIVE = "blurry, low quality, deformed, bad anatomy, extra fingers, watermark, text, signature"

# Три промпта разной сложности и тематики.
# Простой → средний → сложный (структура: объект, детали, окружение, стиль, свет, качество)
prompts = [
    {
        "name": "01_simple_cat",
        "level": "простой",
        "prompt": "a cat sitting on a roof",
        "negative": None,  # специально без улучшений — для сравнения
    },
    {
        "name": "02_medium_frankfurt",
        "level": "средний",
        "prompt": ("Frankfurt am Main skyline at golden hour, view from the Main river, "
                   "reflections on the water, cinematic, highly detailed"),
        "negative": NEGATIVE,
    },
    {
        "name": "03_complex_robot_dev",
        "level": "сложный",
        "prompt": ("a small friendly robot programmer sitting at a wooden desk, "
                   "typing Python code on a glowing laptop, cozy room with plants and bookshelves, "
                   "rainy window in the background, warm volumetric light, "
                   "digital painting, Pixar-like 3D style, soft shadows, "
                   "highly detailed, sharp focus, 8k"),
        "negative": NEGATIVE,
    },
]

md_lines = [
    "# ДЗ 7 — Stable Diffusion",
    f"Модель: `{MODEL}` (локально, CPU), 512x512, steps=25, guidance_scale=7.5, seed={SEED}",
    "",
]

for p in prompts:
    print(f"Генерирую {p['name']} ({p['level']})...")
    image = pipe(
        p["prompt"],
        negative_prompt=p["negative"],
        height=512,
        width=512,
        num_inference_steps=25,
        guidance_scale=7.5,
        generator=torch.Generator("cpu").manual_seed(SEED),
    ).images[0]
    path = OUT_DIR / f"{p['name']}.png"
    image.save(path)
    print("  сохранено:", path.name)

    md_lines += [
        f"## {p['name']} — {p['level']} промпт",
        f"**Prompt:** {p['prompt']}",
        f"**Negative prompt:** {p['negative'] or '—'}",
        f"![{p['name']}]({path.name})",
        "",
    ]

(OUT_DIR / "prompts.md").write_text("\n".join(md_lines), encoding="utf-8")
print("Готово: prompts.md и 3 изображения в папке hw07")

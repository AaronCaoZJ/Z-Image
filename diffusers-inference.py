import torch
from diffusers import ZImagePipeline

# 配置 torch.compile 的动态编译行为
torch._dynamo.config.cache_size_limit = 64  # 增加缓存大小以支持更多动态尺寸
torch._dynamo.config.suppress_errors = True  # 抑制一些不重要的错误

# 1. Load the pipeline
# Use bfloat16 for optimal performance on supported GPUs
# pipe = ZImagePipeline.from_pretrained(
#     "Tongyi-MAI/Z-Image-Turbo",
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=False,
# )
# pipe.to("cuda")
pipe = ZImagePipeline.from_pretrained(
    "/home/zhijun/Code/Z-Image/ckpts/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")

# [Optional] Attention Backend
# Diffusers uses SDPA by default. Switch to Flash Attention for better efficiency if supported:
# pipe.transformer.set_attention_backend("flash")    # Enable Flash-Attention-2
# pipe.transformer.set_attention_backend("_flash_3") # Enable Flash-Attention-3

# [Optional] Model Compilation
# Compiling the DiT model accelerates inference, but the first run will take longer to compile.
# Use dynamic=True to support different image sizes without recompilation
pipe.transformer = torch.compile(pipe.transformer, mode='default', dynamic=True)

# [Optional] CPU Offloading
# Enable CPU offloading for memory-constrained devices.
# pipe.enable_model_cpu_offload()

# prompt = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
prompt = "A mid-range phone selfie captured a young East Asian woman with long black hair taking a selfie in front of a mirror in a brightly lit elevator. She was wearing a black off shoulder short top with a white floral pattern and dark jeans. Her head tilted slightly, and her lips curled up in a kiss, very cute and playful. She held a dark gray smartphone in her right hand, covering part of her face, with the rear camera facing the mirror. The elevator walls are made of polished stainless steel, reflecting the fluorescent lights and main body above the head. There is a vertical panel on the left wall with many circular buttons and a small digital display screen. Below the button, you can see a metal armrest. There is a rectangular sign with text on the back wall. The ground is covered with dark marble tiles with white texture. The overall lighting is artificial light, bright, and has the characteristics of an elevator interior."

import time
output_path = "outputs"
# 2. Generate images and measure time
# 第一次运行（编译 + 推理）
start = time.time()
image = pipe(
    prompt=prompt,
    height=1536,
    width=1536,
    num_inference_steps=9,
    guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
first_run_time = time.time() - start
print(f"第一次运行（含编译）: {first_run_time:.2f}s")
image.save(f"{output_path}/first_run.png")

# 第二次运行（只推理，使用编译缓存）
start = time.time()
image = pipe(
    prompt=prompt,
    height=1024,
    width=768,
    num_inference_steps=9,
    guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(43),
).images[0]
second_run_time = time.time() - start
print(f"第二次运行（编译后）: {second_run_time:.2f}s")
image.save(f"{output_path}/second_run.png")

print(f"加速比: {first_run_time / second_run_time:.2f}x")

start = time.time()
image = pipe(
    prompt=prompt,
    height=1024,
    width=1536,
    num_inference_steps=9,
    guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(43),
).images[0]
third_run_time = time.time() - start
print(f"第三次运行（编译后）: {third_run_time:.2f}s")
image.save(f"{output_path}/third_run.png")

start = time.time()
image = pipe(
    prompt=prompt,
    height=1024,
    width=1536,
    num_inference_steps=9,
    guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(43),
).images[0]
fourth_run_time = time.time() - start
print(f"第四次运行（编译后）: {fourth_run_time:.2f}s")
image.save(f"{output_path}/fourth_run.png")


start = time.time()
image = pipe(
    prompt=prompt,
    height=1536,
    width=768,
    num_inference_steps=9,
    guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(43),
).images[0]
fifth_run_time = time.time() - start
print(f"第五次运行（编译后）: {fifth_run_time:.2f}s")
image.save(f"{output_path}/fifth_run.png")
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = DiffusionPipeline.from_pretrained("TenStrip/LTX2.3-10Eros", dtype=torch.bfloat16, device_map="cuda")
pipe.to("cuda")

prompt = "A man with short gray hair plays a red electric guitar."
image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/guitar-man.png"
)

output = pipe(image=image, prompt=prompt).frames[0]
export_to_video(output, "output.mp4")

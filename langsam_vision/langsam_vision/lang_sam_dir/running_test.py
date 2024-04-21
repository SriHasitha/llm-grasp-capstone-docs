'''
    This is just adapted from the example in the readme,
    The main usage is for the built image to have the weights cached.
'''
from pynvml import *
nvmlInit()
h = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(h)
print(f'total    : {info.total}')
print(f'free     : {info.free}')
print(f'used     : {info.used}')
from PIL import Image
from lang_sam import LangSAM

model = LangSAM()
image_pil = Image.open("car.jpeg").convert("RGB")
text_prompt = "wheel"
masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)

print('all ok')
import argparse
from pathlib import Path
import torch
import asyncio
import websockets
from PIL import Image
from io import BytesIO
import numpy as np

from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device, smart_inference_mode
from utils.dataloaders import LoadImages

async def detect_plastic_bottle(weights='last.pt', img_size=640, conf_thres=0.50, iou_thres=0.45, classes=None, agnostic_nms=False,max_det=1000, source=None):
    device = select_device('')
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt

    model.warmup(imgsz=(1 if pt or model.triton else 1, 3, img_size, img_size))

    img = Image.open(BytesIO(source))
    img = img.resize((640, 640)).convert('RGB')  # Resize and convert to RGB format

    img_tensor = torch.from_numpy(np.array(img)).unsqueeze(0).permute(0, 3, 1, 2).to(model.device).float() / 255.0

    pred = model(img_tensor, augment=False)
    # pred = non_max_suppression(pred, conf_thres, 0.45, max_det=1000)

    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)


    if pred[0] is not None and len(pred[0]) > 0:
        return "1"
    else:
        return "0"
   

async def p_detect():
    url = "ws://192.168.4.1:81"

    async with websockets.connect(url) as ws:
        # await ws.send("s")
        while True:
            image_bytes = await ws.recv()

            if type(image_bytes) == str:
                print(image_bytes)
                continue

            prediction = await detect_plastic_bottle(source=image_bytes)

            print(prediction)

            if prediction == "1":
                await ws.send(prediction)
                # await wait_for(5)
            await ws.send("s")

        
                

            

if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(p_detect())
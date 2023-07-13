import clip
import logging
import torch
import numpy as np
from fastapi import FastAPI, UploadFile
from PIL import Image

from run import load_safety_model, normalized


# Patch from https://github.com/openai/CLIP/issues/79#issuecomment-1624202950
def _node_get(node: torch._C.Node, key: str):
    """Gets attributes of a node which is polymorphic over return type."""
    sel = node.kindOf(key)
    return getattr(node, sel)(key)


torch._C.Node.__getitem__ = _node_get

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    logging.warning("GPU is unavailable, running on CPU")

safety_model = load_safety_model()
model, preprocess = clip.load("ViT-L/14", device=device, jit=True)

app = FastAPI()


@app.post('/classify')
async def classify(image: UploadFile):
    image = preprocess(Image.open(image.file).resize((256, 256))).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    emb = np.asarray(normalized(image_features.detach().cpu()))
    nsfw_value = safety_model.predict(emb)
    res = float(nsfw_value[0][0])

    return {'nsfw': res}

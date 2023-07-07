import os
import numpy as np
import torch
import clip
from PIL import Image
from argparse import ArgumentParser


def load_safety_model(clip_model="ViT-L/14"):
    """load the safety model"""
    import autokeras as ak  # pylint: disable=import-outside-toplevel
    from tensorflow.keras.models import load_model  # pylint: disable=import-outside-toplevel

    cache_folder = "./"

    if clip_model == "ViT-L/14":
        model_dir = cache_folder + "/clip_autokeras_binary_nsfw"
        dim = 768
    else:
        raise ValueError("Unknown clip model")
    if not os.path.exists(model_dir):
        os.makedirs(cache_folder, exist_ok=True)

        from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel

        path_to_zip_file = cache_folder + "/clip_autokeras_binary_nsfw.zip"
        if clip_model == "ViT-L/14":
            url_model = "https://raw.githubusercontent.com/LAION-AI/CLIP-based-NSFW-Detector/main/clip_autokeras_binary_nsfw.zip"
        elif clip_model == "ViT-B/32":
            url_model = (
                "https://raw.githubusercontent.com/LAION-AI/CLIP-based-NSFW-Detector/main/clip_autokeras_nsfw_b32.zip"
            )
        else:
            raise ValueError("Unknown model {}".format(clip_model))  # pylint: disable=consider-using-f-string
        urlretrieve(url_model, path_to_zip_file)
        import zipfile  # pylint: disable=import-outside-toplevel

        with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
            zip_ref.extractall(cache_folder)

    loaded_model = load_model(model_dir, custom_objects=ak.CUSTOM_OBJECTS)
    
    return loaded_model

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image_path", default="_vyr_6097Sexy-Push-Up-Bikini-Brasilianisch-Bunt-2.jpg", help="path to image to be checked")
    opt = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    safety_model = load_safety_model()
    model, preprocess = clip.load("ViT-L/14", device=device)

    # test_embeddings = np.random.rand(10**2, 768).astype("float32")
    # nsfw_values = safety_model.predict(test_embeddings, batch_size=test_embeddings.shape[0])

    image = preprocess(Image.open(opt.image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
    emb = np.asarray(normalized(image_features.detach().cpu()))
    nsfw_value = safety_model.predict(emb)

    print(nsfw_value)
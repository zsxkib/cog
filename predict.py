# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import subprocess
import time
from cog import BasePredictor, Input, Path
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

OFFLINE = True
MODEL_CACHE = "models"
BASE_URL = f"https://weights.replicate.delivery/default/BRIA-RMBG-2.0/{MODEL_CACHE}/"
os.environ["HF_DATASETS_OFFLINE"] = "1" if OFFLINE else "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1" if OFFLINE else "0"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE


def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")
    
class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        
        model_files = [
            "models--briaai--RMBG-2.0.tar",
        ]

        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)

        for model_file in model_files:
            url = BASE_URL + model_file

            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)
                
        torch.set_float32_matmul_precision("high")
        
        # Load model
        self.model = AutoModelForImageSegmentation.from_pretrained(
            "briaai/RMBG-2.0", 
            cache_dir=MODEL_CACHE,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model.to("cuda")
        
        # Setup image transformation pipeline
        self.transform_image = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict(
        self,
        image: Path = Input(description="Input image for background removal"),
        output_format: str = Input(
            description="Format of the output image",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="Quality when saving the output image, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
            default=80,
            ge=0,
            le=100,
        ),
    ) -> Path:
        """Run background removal on the input image"""
        # Load and preprocess image
        im = Image.open(str(image))
        im = im.convert("RGB")
        image_size = im.size
        
        # Transform and predict
        input_images = self.transform_image(im).unsqueeze(0).to("cuda")
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        
        # Post-process
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        
        # Apply alpha mask
        im.putalpha(mask)
        
        # Save and return result with specified format and quality
        output_path = Path(f"/tmp/output.{output_format}")
        
        if output_format != "png":
            # For WEBP and JPEG, use quality parameter and optimization
            im.save(str(output_path), quality=output_quality, optimize=True)
        else:
            # For PNG, ignore quality parameter as it's lossless
            im.save(str(output_path), optimize=True)
        
        return output_path
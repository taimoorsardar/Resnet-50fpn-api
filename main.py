from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
import os
import requests
from supabase import create_client, Client
import matplotlib.pyplot as plt

from utils.image_processor import ImageProcessor

from constants import SUPABASE_KEY, SUPABASE_URL, CATS, STATE_DICT_PATH, MODEL_MODULE

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize the image processor instance
image_processor = ImageProcessor(CATS, STATE_DICT_PATH, model=MODEL_MODULE)

app = FastAPI(
    title="Model Inference API",
    description="An API for image classification and CAM generation using a custom model",
    version="1.0",
)

def fetch_image_from_url(url: str) -> Image.Image:
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to fetch image from URL")
    return Image.open(BytesIO(response.content))

def upload_image_to_supabase(image: Image.Image, image_name: str) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    file_bytes = buf.getvalue()

    # Define file options as a dictionary
    file_options = {"content-type": "image/png"}
    
    # Upload the file to the 'images' bucket
    response = supabase.storage.from_("images").upload(image_name, file_bytes, file_options)
    
    # Get the public URL for the uploaded image
    public_url = supabase.storage.from_("images").get_public_url(image_name)
    return public_url

def generate_cam_overlay(original_image: np.ndarray, cam: np.ndarray, alpha: float = 0.6, cmap_name: str = 'jet') -> Image.Image:
    cmap = plt.get_cmap(cmap_name)
    cam_color = cmap(cam)
    cam_color = (cam_color[:, :, :3] * 255).astype('uint8')
    orig_img = Image.fromarray(original_image.astype('uint8'))
    cam_img = Image.fromarray(cam_color).resize(orig_img.size, resample=Image.BILINEAR)
    overlay = Image.blend(orig_img, cam_img, alpha)
    return overlay

@app.post("/process_image/")
async def process_image(image_url: str = Form(None)):
    try:
        if image_url:
            image = fetch_image_from_url(image_url)
        else:
            raise HTTPException(status_code=400, detail="image_url must be provided")

        if image.mode not in ('L', 'RGB'):
            image = image.convert('RGB')
        image_np = np.array(image)
        
        result_wrapper = image_processor.execute_cams_pred(image_np)
        predicted_categories = {k: float(v) for k, v in result_wrapper.predicted_categories.items()}
        scores = result_wrapper.classification_scores
        best_idx = int(np.argmax(scores))
        cam_for_best = result_wrapper.global_cams[best_idx]
        cam_norm = (cam_for_best - np.min(cam_for_best)) / (np.ptp(cam_for_best) + 1e-8)
        overlay_image = generate_cam_overlay(result_wrapper.image, cam_norm)
        processed_image_url = upload_image_to_supabase(overlay_image, "processed_image.png")
        
        return {"classification_scores": scores.tolist(), "predicted_categories": predicted_categories, "processed_image_url": processed_image_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Accept an image file, run model inference (classification and CAM generation)
    and return the prediction results.
    """
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        if image.mode not in ('L', 'RGB'):
            image = image.convert('RGB')
        image_np = np.array(image)
        
        # Run model inference
        result_wrapper = image_processor.execute_cams_pred(image_np)
        
        # Convert numpy.float32 in predicted_categories to native float
        predicted_categories = {
            k: float(v) for k, v in result_wrapper.predicted_categories.items()
        }
        # For the CAM overlay: if you have CAMs per category, you can choose one.
        # Here we select the CAM corresponding to the category with the highest score.
        scores = result_wrapper.classification_scores
        best_idx = int(np.argmax(scores))
        # Assuming result_wrapper.global_cams is shaped (num_categories, H, W)
        cam_for_best = result_wrapper.global_cams[best_idx]
        # Ensure the CAM is normalized between 0 and 1.
        cam_norm = (cam_for_best - np.min(cam_for_best)) / (np.ptp(cam_for_best) + 1e-8)
        
        # Generate the overlay image (Base64 encoded PNG)
        overlay_image = generate_cam_overlay(result_wrapper.image, cam_norm)
         
        response = {
            "classification_scores": result_wrapper.classification_scores.tolist(),
            "predicted_categories": predicted_categories,
            "overlay_image": overlay_image
        }
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

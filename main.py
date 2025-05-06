# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
import io
import os
import base64
import matplotlib.pyplot as plt

# Import your ImageProcessor from your utils package
from utils.image_processor import ImageProcessor

# Define the target categories, path to the checkpoint, and model module.
CATS = ["suspicious_site"]
STATE_DICT_PATH = "weights\\checkpoint.pth"
# Set the model to the module that contains your model definitions.
MODEL_MODULE = "architecture.resnet50_fpn"

# Initialize the image processor instance
image_processor = ImageProcessor(CATS, STATE_DICT_PATH, model=MODEL_MODULE)

app = FastAPI(
    title="Model Inference API",
    description="An API for image classification and CAM generation using a custom model",
    version="1.0",
)

def generate_cam_overlay(original_image: np.ndarray, cam: np.ndarray, alpha: float = 0.6, cmap_name: str = 'jet') -> str:
    """
    Generates an overlay image of the CAM on the original image and returns it as a base64 encoded PNG.
    
    Parameters:
        original_image (np.ndarray): Original image as a NumPy array (H, W, 3).
        cam (np.ndarray): CAM as a 2D NumPy array (values normalized between 0 and 1).
        alpha (float): Blending factor.
        cmap_name (str): Name of the matplotlib colormap to use.
        
    Returns:
        str: Base64 encoded PNG image.
    """
    # Get the colormap and apply it to the CAM to create a colored heatmap.
    cmap = plt.get_cmap(cmap_name)
    cam_color = cmap(cam)  # This returns an (H, W, 4) RGBA array (values in [0,1])
    cam_color = (cam_color[:, :, :3] * 255).astype('uint8')  # Discard alpha and convert to 8-bit
    
    # Convert original image and CAM heatmap to PIL images.
    orig_img = Image.fromarray(original_image.astype('uint8'))
    cam_img = Image.fromarray(cam_color)
    
    # Resize the CAM heatmap to match the original image (if needed).
    cam_img = cam_img.resize(orig_img.size, resample=Image.BILINEAR)
    
    # Blend the original image with the CAM heatmap.
    overlay = Image.blend(orig_img, cam_img, alpha)
    
    # Save overlay to a BytesIO buffer as PNG.
    buf = io.BytesIO()
    overlay.save(buf, format='PNG')
    buf.seek(0)
    
    # Encode the image in Base64.
    overlay_base64 = base64.b64encode(buf.read()).decode('utf-8')
    # Optionally, prepend the data URL header:
    return f"data:image/png;base64,{overlay_base64}"


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
    # Run the API with uvicorn: python main.py
    port = int(os.environ.get["PORT",8080])
    uvicorn.run(app, host="0.0.0.0", port=port)

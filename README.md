Below are two files you can add to your repository: a **README.md** for documentation and a **requirements.txt** listing the dependencies.

---

**README.md**

```markdown
# Model Inference API with CAM Generation

This repository contains a REST API built with FastAPI for image classification and Class Activation Map (CAM) generation using a custom neural network model. The API loads the model, processes an image (with scaling, CAM overlay, and classification), and returns the results as a JSON response including an overlay image (encoded in Base64).

## Project Structure

```
.
├── architecture
│   ├── resnet50.py
│   ├── resnet50_fpn.py
│   └── torchutils.py
├── utils
│   ├── image_wrapper.py
│   ├── image_processor.py
│   └── imutils.py
├── weights
│   └── checkpoint.pth
├── main.py
├── README.md
└── requirements.txt
```

- **architecture/**  
  Contains the model definition files including a ResNet50 backbone and its variants for generating CAMs.

- **utils/**  
  Contains helper functions and classes:
  - `imutils.py` for image pre-processing.
  - `image_wrapper.py` for wrapping image data and computed CAMs.
  - `image_processor.py` for loading images, executing the model, and computing CAMs.

- **weights/**  
  Contains the model checkpoint file (saved weights).

- **main.py**  
  Contains the FastAPI application with endpoints to handle image uploads, run inference, and return results (classification scores and Base64-encoded overlay image).

## Features

- **Image Classification:**  
  Compute classification scores for target categories.

- **CAM Generation:**  
  Compute Class Activation Maps (CAMs) and overlay them on the original image.

- **REST API:**  
  Endpoints created with FastAPI that accept image uploads, process the image using your custom model, and return a JSON response.

- **Base64 Image Encoding:**  
  The API returns the CAM overlay as a Base64-encoded PNG image. A helper function is provided to decode and display the image when needed.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/taimoorsardar/Resnet-50fpn-api.git
   cd Resnet_50fpn_api
   ```
2. **Switch to a branch**
      ```bash
   git branch
   git checkout <branch-name>
   ```
3. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Running the API

Start the API using Uvicorn:

```bash
uvicorn main:app --reload
```

The API will be available at [http://localhost:8000](http://localhost:8000) and you can access the interactive documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

## API Endpoints

- **POST `/predict/`**  
  - **Description:** Accepts an image file, runs classification and CAM generation, and returns:
    - `classification_scores`: List of classification scores.
    - `predicted_categories`: Dictionary of predicted categories (with scores as native floats).
    - `overlay_image`: A Base64-encoded PNG image showing the CAM overlay on the original image.
  
  - **Example Request (using curl):**

    ```bash
    curl -X 'POST' \
      'http://localhost:8000/predict/' \
      -F 'file=@path_to_your_image.png'
    ```

## Decoding the Overlay Image

A helper function is provided to decode the Base64 overlay image. For example:

```python
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

def decode_image(base64_string: str) -> Image.Image:
    if base64_string.startswith("data:image"):
        _, base64_data = base64_string.split(",", 1)
    else:
        base64_data = base64_string
    image_data = base64.b64decode(base64_data)
    image = Image.open(BytesIO(image_data))
    return image

# Example usage:
overlay_image_b64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
decoded_img = decode_image(overlay_image_b64)
decoded_img.show()
```

## Notes

- **Extensibility:**  
  You can extend the API by adding additional endpoints or functionalities (e.g., handling intermediate CAM layers, saving results to disk, etc.).


---

### How to Use

1. Place the above **README.md** and **requirements.txt** files in the root of your project.
2. Follow the instructions in the README to install dependencies and run the API.
3. Use the provided endpoints and helper functions in your client code or interactive Python sessions.

These files should help you get started with documentation and setting up your environment. Let me know if you need any further modifications or additional details!
# Model Inference API with CAM Generation & Supabase Storage

This repository contains a REST API built with FastAPI for image classification and Class Activation Map (CAM) generation using a custom neural network model. The API loads the model, processes an image (with scaling, CAM overlay, and classification), and returns the results as a JSON response including an overlay image (stored in Supabase and returned as a public URL).

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
│   ├── imutils.py
│   ├── supabase_utils.py  # Handles image upload to Supabase
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
  - `supabase_utils.py` for uploading images to Supabase and retrieving public URLs.

- **weights/**  
  Contains the model checkpoint file (saved weights).

- **main.py**  
  Contains the FastAPI application with endpoints to handle image uploads, run inference, store the processed image in Supabase, and return results.

## Features

- **Image Classification:**  
  Compute classification scores for target categories.

- **CAM Generation:**  
  Compute Class Activation Maps (CAMs) and overlay them on the original image.

- **REST API:**  
  Endpoints created with FastAPI that accept image uploads, process the image using your custom model, store the processed image in Supabase, and return a JSON response.

- **Supabase Storage:**  
  The processed image is uploaded to Supabase and a public URL is returned in the response.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/taimoorsardar/Resnet-50fpn-api.git
   cd Resnet-50fpn-api
   ```
   
2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the dependencies:**

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
  - **Description:** Accepts an image file, runs classification and CAM generation, uploads the processed image to Supabase, and returns:
    - `classification_scores`: List of classification scores.
    - `predicted_categories`: Dictionary of predicted categories (with scores as native floats).
    - `image_url`: A public URL of the processed image stored in Supabase.
  
  - **Example Request (using curl):**

    ```bash
    curl -X 'POST' \
      'http://localhost:8000/predict/' \
      -F 'file=@path_to_your_image.png'
    ```

## Using Supabase for Image Storage

The API now integrates with Supabase for storing processed images. The `main.py` file handles:

- Uploading the processed image to the `images` bucket.
- Retrieving the public URL of the uploaded image.

To set up Supabase, ensure you have:

1. A Supabase account ([https://supabase.io](https://supabase.io)).
2. A project with a `storage` bucket named `images`.
3. A `constants.py` file containing your Supabase credentials:

   ```ini
   SUPABASE_URL="your_supabase_url"
   SUPABASE_KEY="your_supabase_key"
   ```

## Notes

- **Extensibility:**  
  You can extend the API by adding additional endpoints or functionalities (e.g., handling intermediate CAM layers, saving metadata to a database, etc.).

---

### How to Use

1. Place the above **README.md** and **requirements.txt** files in the root of your project.
2. Follow the instructions in the README to install dependencies and run the API.
3. Use the provided endpoints and helper functions in your client code or interactive Python sessions.

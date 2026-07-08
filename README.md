# RetinaScan AI - Backend API

The backend for RetinaScan AI is a modular Python Flask API that loads the fine-tuned ResNet50 model, processes retinal fundus images, and generates explainability outputs (Grad-CAM heatmaps and clinical clinical text descriptions).

---

## Codebase Architecture

The backend is structured to separate routing logic, configuration parameters, and core services:

*   **`main.py`**: The API entrypoint. Handles Flask routing, CORS headers, health endpoints, and delegates requests to service instances.
*   **`config.py`**: Centralized configuration management. Resolves absolute directory paths and configures classification categories and threshold metrics.
*   **`services/model_service.py`**: Configures TensorFlow dynamic GPU allocations, loads the `.keras` model, and runs inference.
*   **`services/xai_service.py`**: Computes Grad-CAM heatmaps using TensorFlow `GradientTape` over the final convolution layer (`conv5_block3_out`).
*   **`services/nlp_service.py`**: Interprets spatial heatmap hotspot configurations and translates them into clinical warning narratives using random medical sentence structures.
*   **`utils/image_utils.py`**: Pure utility functions handling byte streams, color-space normalizations, Matplotlib color mappings, and base64 formatting.
*   **`tests/test_pipeline.py`**: Integration testing script confirming that all modules (preprocessing, predictions, Grad-CAM, NLP descriptions, and blending overlays) execute correctly.

---

## Installation & Setup

1.  Navigate to the backend directory:
    ```bash
    cd backend
    ```
2.  Activate the virtual environment:
    *   **Windows (PowerShell):** `.\.venv\Scripts\Activate.ps1`
    *   **Windows (CMD):** `.\.venv\Scripts\activate.bat`
    *   **macOS / Linux:** `source .venv/bin/activate`
3.  Ensure the fine-tuned model file `best_resnet50_model.keras` is in the `assets/` folder.
4.  Run the Flask server:
    ```bash
    python main.py
    ```

---

## API Documentation

### 1. Diagnostic Inference Endpoint
*   **Route:** `/predict`
*   **Method:** `POST`
*   **Payload Format:** `multipart/form-data`
*   **Parameters:**
    *   `image`: Retinal fundus image file (PNG, JPG, or TIFF).
*   **Response Format:** `application/json`
*   **Success Response (200 OK):**
    ```json
    {
      "predicted_class": 2,
      "probabilities": [0.0001, 0.0034, 0.9852, 0.0111, 0.0002],
      "gradcam_image": "base64_encoded_jpeg_string...",
      "explanation": "Significant activation in these zones corresponds with hard exudates...",
      "is_mock": false
    }
    ```

### 2. Operational Health Check
*   **Route:** `/status`
*   **Method:** `GET`
*   **Success Response (200 OK):**
    ```json
    {
      "status": "healthy",
      "mock_mode": false,
      "model_loaded": true
    }
    ```

---

## Verification Testing

You can run the integration test pipeline to verify that all modules are imported correctly, model paths resolve, and predictions execute. This generates dummy canvases and pushes them through all services:
```bash
python tests/test_pipeline.py
```
**Expected Output:**
```
Starting pipeline verification test...
Creating dummy retinal image...
Testing image preprocessing...
Image preprocessing passed!
Testing model service loading from path: .../backend/assets/best_resnet50_model.keras...
Model loaded successfully!
Testing model prediction...
Probabilities: [0.00000062 ... 0.99999797 ... ]
Predicted Class: 2 (Moderate)
Model prediction passed!
Testing XAI Grad-CAM generation...
Heatmap shape: (7, 7)
XAI Grad-CAM passed!
Testing NLP Explainer...
Explanation: ...
NLP Explainer passed!
Testing heatmap overlay blending...
Overlay and encoding passed!

[SUCCESS] All pipeline checks passed successfully!
```

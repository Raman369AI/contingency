from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import onnxruntime as ort
from fastapi.background import BackgroundTasks
import itertools
from fastapi.middleware.cors import CORSMiddleware
import torchvision.transforms as transforms
# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
LABELS = ['cat', 'dog','Other']  # Replace with your actual class labels

class ONNXModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None
        self.input_name = None

    def __enter__(self):
        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            del self.session  # Proper cleanup of ONNX session
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize model on startup
    with ONNXModel("model.onnx") as model:
        app.state.model = model
        yield
    # Cleanup happens automatically when context exits

app = FastAPI(lifespan=lifespan)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomRotation(degrees=15),
     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    ])

def allowed_file(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
from fastapi import status
from fastapi.responses import Response


origins = [
    "https://localhost:9002",  #  The origin of your React app (port 3000 is common for Create React App)
    "https://127.0.0.1:9002", #  Include this as well
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  #  Allow all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],  #  Allow all headers
)
@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(400, detail="Invalid file type")

    try:
        # Load and preprocess image
        image = Image.open(file.file).convert('RGB')
        processed_img = transform(image).unsqueeze(0).numpy()
        
        # Get model from application state
        model = app.state.model
        
        # Run inference
        outputs = model.session.run(
            None, 
            {model.input_name: processed_img}
        )
        logits = outputs[0]
        
        # Post-process results
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        print(probabilities[0])
        max_prob = np.max(probabilities[0])
        print(max_prob)

        pred_class = [2 if max_prob < 0.95 else np.argmax(probabilities)][0]

        #pred_class = np.argmax(probabilities)
        
        return JSONResponse({
            'class': LABELS[pred_class],
            'confidence': float(max_prob),
            'all_predictions': dict(zip(LABELS, probabilities[0].tolist()))
        })
        
    except Exception as e:
        raise HTTPException(500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

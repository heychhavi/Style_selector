from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
from PIL import Image
import base64
from io import BytesIO
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from wardrobe_recommender import WardrobeRecommender
from outfit_visualizer import OutfitVisualizer
from fashion_agent import FashionAgent

# Initialize FastAPI
api = FastAPI()

# Add CORS middleware
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
recommender = WardrobeRecommender()
visualizer = OutfitVisualizer()
fashion_agent = FashionAgent(api_key="pin_MTI0MDAwODA6MzYxNjQ_KeyxdDEM06aOH5gk")

class StylePreferences(BaseModel):
    occasion: str
    budget: float
    colorPreferences: List[str]
    stylePreferences: List[str]

@api.post("/upload")
async def upload_photos(files: List[UploadFile] = File(...)):
    try:
        image_urls = []
        for file in files:
            contents = await file.read()
            image = Image.open(BytesIO(contents))
            
            # Save image and get URL
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            image_urls.append(f"data:image/jpeg;base64,{img_str}")
        
        return {"imageUrls": image_urls}
    except Exception as e:
        return {"error": str(e)}, 400

@api.post("/analyze")
async def analyze_style(imageUrls: List[str]):
    try:
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        
        style_features = []
        for img_url in imageUrls:
            # Convert base64 to image
            img_data = base64.b64decode(img_url.split(',')[1])
            image = Image.open(BytesIO(img_data))
            
            # Get features
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            style_features.append(outputs.logits)
        
        # Aggregate features
        style_profile = torch.mean(torch.stack(style_features), dim=0)
        
        return {"styleProfile": style_profile.tolist()}
    except Exception as e:
        return {"error": str(e)}, 400

@api.post("/recommendations")
async def get_recommendations(preferences: StylePreferences):
    try:
        recommendations = recommender.get_outfit_recommendations(
            style_profile=None,
            occasion=preferences.occasion,
            budget=preferences.budget,
            preferences={
                "color_preference": preferences.colorPreferences,
                "style_preference": preferences.stylePreferences
            }
        )
        
        return {"recommendations": recommendations}
    except Exception as e:
        return {"error": str(e)}, 400

@api.post("/preview")
async def generate_preview(imageUrl: str, outfitItems: List[dict]):
    try:
        # Convert base64 to image
        img_data = base64.b64decode(imageUrl.split(',')[1])
        image = Image.open(BytesIO(img_data))
        
        # Generate preview
        preview = visualizer.generate_outfit_preview(
            user_image=image,
            outfit_items=outfitItems
        )
        
        return {"previewUrl": f"data:image/jpeg;base64,{preview}"}
    except Exception as e:
        return {"error": str(e)}, 400

if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=8504) 
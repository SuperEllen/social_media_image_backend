"""
FastAPI Backend for Social Media Engagement Prediction

This API accepts image uploads and user information, extracts image features,
and predicts engagement metrics (likes, comments, engagement rate).
"""

import json
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict

from feature_extractor import extract_image_features
from predictor import predict_engagement


app = FastAPI(
    title="Social Media Engagement Predictor API",
    description="Analyze images and predict social media engagement",
    version="1.0.0"
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UserInfo(BaseModel):
    follower_base: int
    followees: int
    number_of_posts: int
    number_of_tags: int
    number_of_mentions: int


class AnalyzeResponse(BaseModel):
    features: Dict[str, Any]
    prediction: Dict[str, Any]


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "Engagement Predictor API is running"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(
    image: UploadFile = File(..., description="Image file to analyze"),
    userInfo: str = Form(..., description="JSON string containing user information")
):
    """
    Analyze an image and predict engagement.
    
    - **image**: The image file to analyze (JPEG, PNG, etc.)
    - **userInfo**: JSON string with user information:
        - followers: Number of followers
        - avg_likes: Average likes per post
        - post_frequency: Posts per week
        - account_age: Account age in months
    
    Returns features and engagement prediction.
    """
    
    # Validate image
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Parse user info
    try:
        user_info_dict = json.loads(userInfo)
        user_info = UserInfo(**user_info_dict)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in userInfo")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid userInfo format: {str(e)}")
    
    # Read image bytes
    try:
        image_bytes = await image.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {str(e)}")
    
    # Extract features
    try:
        features = extract_image_features(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")
    
    # Predict engagement
    try:
        prediction = predict_engagement(features, user_info.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    return AnalyzeResponse(features=features, prediction=prediction)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi.responses import JSONResponse
import json
from fastapi import APIRouter, File, UploadFile
from controllers.productController import analyze_image_controller

router = APIRouter()

@router.post("/upload")
async def analyze_image(file: UploadFile = File(...)):
    """
    Endpoint to receive an image, upload it to Gemini, and analyze it.
    """
    try:
        analysis = await analyze_image_controller(file)
        return JSONResponse(content=json.loads(analysis))  # Deserialize formatted JSON string to dict
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Error during image analysis: {str(e)}"})


from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from automl.train import train_model
import pandas as pd
import io

router = APIRouter()

@router.post("/train")
async def train(
    file: UploadFile = File(...),
    target: str = Query(..., description="Target column name")
):
    try:
        contents = await file.read()
        content_type = file.content_type or file.filename.lower()

        # Detect format
        if content_type.endswith(".csv") or "csv" in content_type:
            df = pd.read_csv(io.BytesIO(contents))
        elif content_type.endswith(".json") or "json" in content_type:
            df = pd.read_json(io.BytesIO(contents))
        elif content_type.endswith(".parquet") or "parquet" in content_type:
            df = pd.read_parquet(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV, JSON, or Parquet.")

        result = train_model(df, target_column=target)
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Training failed: {str(e)}")
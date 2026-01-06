# app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from .models import ExtractResponse
from .ocr import extract_from_pdf

app = FastAPI(
    title="Contract OCR Service",
    version="1.0.0"
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/extract", response_model=ExtractResponse)
async def extract_contract(file: UploadFile = File(...)):
  
    # Basic validation
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="Empty file.")

        extraction = extract_from_pdf(pdf_bytes)

        return ExtractResponse(
            success=True,
            message="Extraction completed.",
            extraction=extraction
        )
    except Exception as ex:
        # In production, log ex with traceback
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Error during extraction: {str(ex)}",
                "extraction": None
            }
        )

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os

from translator import Translator
from summarizer import Summarizer
from diarizer import Diarizer
from transcriber import Transcriber

diarizer = Diarizer()

app = FastAPI(title="Lisan Backend API", version="1.0.0")

origins = [
    "https://enchanting-pothos-2cd2b8.netlify.app",  
    "http://localhost:5500",                        
    "http://127.0.0.1:5500"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

translator_service = Translator()
summarizer_service = Summarizer()
diarizer_service = Diarizer()
transcriber_service = Transcriber()

class TranslationRequest(BaseModel):
    text: str
    target_lang: str = "ar"

class SummaryRequest(BaseModel):
    text: str
    target_lang: str = "ar"

@app.get("/")
def root():
    return {"status": "Online", "service": "Lisan AI Backend Engine"}

@app.post("/api/translate")
def translate_api(req: TranslationRequest):
    translated_text = translator_service.translate(req.text, req.target_lang)
    return {"original": req.text, "translated": translated_text}

@app.post("/api/summarize")
def summarize_api(req: SummaryRequest):
    summary_text = summarizer_service.summarize(req.text, req.target_lang)
    return {"summary": summary_text}

@app.post("/api/diarize")
async def handle_diarization(file: UploadFile = File(...)):
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="حجم الملف كبير جداً! الحد الأقصى هو 25 ميجابايت.")
    
    await file.seek(0)

    temp_filename = f"temp_{file.filename}"
    
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        result_text = diarizer.process_diarization(temp_filename)
        return {"diarization": result_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

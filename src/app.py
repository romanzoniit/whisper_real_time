from fastapi import FastAPI
from src.recognition.recoginition import Recognition
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

rec = Recognition()


class Item(BaseModel):
    model: str
    language: str
    transcribe_timeout: float


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", encoding="utf-8") as file:
        return file.read()


@app.post("/transcribe", tags=["Transcribe"])
async def transcribe(item: Item,
                     ):
    if item.language == 'en':
        rec.recognize(language=item.language)
    else:
        return {"transcription": rec.recognize()}

import google.generativeai as genai
import os

class Diarizer:
    def __init__(self):
        genai.configure(api_key=os.getenv("GENAI_API_KEY"))
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    def process_diarization(self, file_path: str):
        audio_file = genai.upload_file(path=file_path)
        prompt = """
        response = self.model.generate_content([audio_file, prompt])
        return response.text

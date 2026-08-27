import google.generativeai as genai
import os

class Transcriber:
    def __init__(self):
        GENAI_API_KEY = os.getenv("GENAI_API_KEY")
        genai.configure(api_key=GENAI_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def transcribe_audio(self, file_path: str) -> str:
        try:
            audio_file = genai.upload_file(path=file_path)
            response = self.model.generate_content([
                "Write a full, accurate transcript of this audio in its original language without summarizing.",
                audio_file
            ])
            genai.delete_file(audio_file.name)
            return response.text
        except Exception as e:
            return f"Error in transcription: {str(e)}"

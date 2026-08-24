class Diarizer:
    def format_speaker_text(self, speaker: str, text: str) -> dict:
        return {
            "speaker": speaker,
            "text": text
        }
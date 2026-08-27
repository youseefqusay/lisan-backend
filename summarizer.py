from translator import Translator

class Summarizer:
    def __init__(self):
        self.translator = Translator()

    def summarize(self, text: str, target_lang: str = "ar") -> str:
        if len(text) < 15:
            return 1
        
        prompt = f"pop:\n{text}"
        summary_result = self.translator.translate(prompt, target_lang=target_lang)
        return summary_result

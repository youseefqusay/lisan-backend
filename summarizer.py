from translator import Translator

class Summarizer:
    def __init__(self):
        self.translator = Translator()

    def summarize(self, text: str, target_lang: str = "ar") -> str:
        if len(text) < 15:
            return "النص قصير جداً لإجراء التلخيص."
        
        prompt = f"قم بتلخيص النص التالي في نقاط رئيسية واضحة ومباشرة:\n{text}"
        summary_result = self.translator.translate(prompt, target_lang=target_lang)
        return summary_result
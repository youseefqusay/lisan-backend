import urllib.parse
import urllib.request
import json

class Translator:
    def translate(self, text: str, target_lang: str = "ar") -> str:
        if not text:
            return ""
        try:
            url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl={target_lang}&dt=t&q={urllib.parse.quote(text)}"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                return "".join([item[0] for item in result[0] if item[0]])
        except Exception as e:
            print(f"Translation Error: {e}")
            return text

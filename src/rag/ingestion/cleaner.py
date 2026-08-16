import re

class TextCleaner:
    def clean(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

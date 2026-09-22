import re
from html import unescape

def normalize_text(text: str) -> str:
 
    if not isinstance(text, str):
        return ""

    text = unescape(text)

    text = re.sub(r"<[^>]+>", " ", text)

    text = text.lower()

    text = re.sub(r"\s+", " ", text)

    text = text.strip()

    return text

if __name__ == "__main__":
    texto_teste = "I REALLY loved this movie!!! <br /><br /> It was amazing."

    resultado = normalize_text(texto_teste)

    print("Texto original:")
    print(texto_teste)

    print("\nTexto normalizado:")
    print(resultado)
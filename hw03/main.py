import os

from dotenv import load_dotenv
from google import genai

# Загружаем переменные из .env в окружение (ключ не хранится в коде)
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY не найден. Создайте файл .env по образцу .env.example")

client = genai.Client(api_key=api_key)

prompt = "Объясни в трёх предложениях, что такое токенизация в больших языковых моделях."

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt,
)

print("Вопрос:", prompt)
print("Ответ Gemini:\n", response.text)
# Сколько токенов ушло на запрос и ответ — связь с темой урока «Токенизация»
print("Токены (запрос / ответ):",
      response.usage_metadata.prompt_token_count,
      "/",
      response.usage_metadata.candidates_token_count)

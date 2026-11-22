import os

from dotenv import load_dotenv

load_dotenv()
# WebBaseLoader просит указать User-Agent, иначе выводит предупреждение
os.environ.setdefault("USER_AGENT", "python-for-ai-homework/1.0")

from langchain_community.document_loaders import WebBaseLoader  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("GEMINI_API_KEY не найден в .env")

URL = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
MAX_CHARS = 15_000  # ограничиваем объём текста, чтобы не тратить лишние токены

# 1. Загружаем страницу и превращаем HTML в текст
docs = WebBaseLoader(URL).load()
text = " ".join(docs[0].page_content.split())[:MAX_CHARS]

# 2. Шаблон промпта
prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты делаешь краткие и точные конспекты статей на русском языке."),
    ("human",
     "Сделай суммаризацию текста веб-страницы:\n"
     "1) главная мысль — одно предложение;\n"
     "2) 5 ключевых тезисов списком;\n"
     "3) кому эта статья будет полезна.\n\n"
     "Текст:\n{text}"),
])

# 3. Модель
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3,
)

# 4. Цепочка LCEL: промпт → модель → строка
chain = prompt | llm | StrOutputParser()

summary = chain.invoke({"text": text})
print(f"URL: {URL}\n")
print(summary)

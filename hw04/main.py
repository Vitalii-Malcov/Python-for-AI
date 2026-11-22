import logging
import os
import time
from collections import deque

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY не найден в .env")

# Таймаут на HTTP-запрос: 30 секунд (в google-genai указывается в миллисекундах)
client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=30_000),
)


class RateLimiter:
    """Не больше max_calls запросов за period секунд (скользящее окно).

    Защищает от блокировки API: бесплатный тариф Gemini ограничивает
    число запросов в минуту, при превышении приходит ошибка 429.
    """

    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self.calls: deque[float] = deque()

    def wait(self) -> None:
        while True:
            now = time.monotonic()
            # Удаляем из окна запросы старше period секунд
            while self.calls and now - self.calls[0] >= self.period:
                self.calls.popleft()
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return
            sleep_for = self.period - (now - self.calls[0])
            logger.info("Достигнут лимит %s запросов/%s c, ждём %.1f c",
                        self.max_calls, self.period, sleep_for)
            time.sleep(sleep_for)


# 5 запросов в минуту — с запасом под лимиты бесплатного тарифа
limiter = RateLimiter(max_calls=5, period=60)


def is_retryable(exc: BaseException) -> bool:
    """Повторяем только временные ошибки: таймауты, сеть, 429 и 5xx."""
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
        return True
    if isinstance(exc, errors.APIError):
        return exc.code in (429, 500, 502, 503, 504)
    return False


@retry(
    retry=retry_if_exception(is_retryable),
    stop=stop_after_attempt(5),                           # максимум 5 попыток
    wait=wait_exponential(multiplier=2, min=2, max=60),   # 2, 4, 8, 16... секунд
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,                                         # после 5 попыток — исходная ошибка
)
def ask_gemini(prompt: str) -> str:
    limiter.wait()  # сначала соблюдаем лимит, потом отправляем запрос
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )
    return response.text


if __name__ == "__main__":
    questions = [
        "Что такое rate limit в API? Одно предложение.",
        "Зачем нужен таймаут у HTTP-запроса? Одно предложение.",
        "Что такое exponential backoff? Одно предложение.",
        "Чем ошибка 429 отличается от 503? Одно предложение.",
        "Что такое идемпотентный запрос? Одно предложение.",
        "Зачем библиотека tenacity? Одно предложение.",
        "Что такое эмбеддинг? Одно предложение.",
    ]
    # 7 вопросов при лимите 5/мин — увидим в логах, как срабатывает RateLimiter
    for i, q in enumerate(questions, start=1):
        try:
            answer = ask_gemini(q)
            print(f"{i}. {q}\n   → {answer.strip()}\n")
        except httpx.TimeoutException:
            print(f"{i}. Таймаут: сервер не ответил за 30 c даже после повторов")
        except errors.APIError as e:
            print(f"{i}. Ошибка API {e.code}: {e.message}")

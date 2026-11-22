import re
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

BASE_DIR = Path(__file__).parent
MODEL_NAME = "distilbert-base-uncased"
SAMPLE_SIZE = 2000   # небольшой датасет, чтобы обучение шло на CPU
MAX_LEN = 128
SEED = 42

# 1. Датасет: отзывы о фильмах IMDB (0 — негатив, 1 — позитив)
raw = load_dataset("stanfordnlp/imdb", split="train").shuffle(seed=SEED).select(range(SAMPLE_SIZE))
df = raw.to_pandas()
print("Исходный размер:", len(df))


# 2. Очистка данных от шума
def clean_text(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", text)      # HTML-переносы из IMDB
    text = re.sub(r"<[^>]+>", " ", text)        # прочие HTML-теги
    text = re.sub(r"http\S+", " ", text)        # ссылки
    text = re.sub(r"\s+", " ", text)            # лишние пробелы
    return text.strip()


df["text"] = df["text"].apply(clean_text)
df = df[df["text"].str.len() > 20]               # пустые и слишком короткие
df = df.drop_duplicates(subset="text")           # дубликаты
print("После очистки:", len(df))

# 3. Разбиение на обучающую и тестовую выборки (80/20, с сохранением баланса классов)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label"])
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)


train_ds = Dataset.from_pandas(train_df, preserve_index=False).map(tokenize, batched=True)
test_ds = Dataset.from_pandas(test_df, preserve_index=False).map(tokenize, batched=True)

# 4. Fine-tuning: предобученный DistilBERT + новый слой классификации на 2 класса
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "negative", 1: "positive"},
    label2id={"negative": 0, "positive": 1},
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}


args = TrainingArguments(
    output_dir=str(BASE_DIR / "results"),
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=20,
    seed=SEED,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

trainer.train()

# 5. Оценка качества на тестовых данных
pred = trainer.predict(test_ds)
y_pred = np.argmax(pred.predictions, axis=-1)
print("\n=== Качество на тестовой выборке ===")
print(classification_report(test_df["label"], y_pred, target_names=["negative", "positive"]))

# Проверка на своих примерах
examples = [
    "This movie was absolutely wonderful, great acting and story!",
    "Boring, too long and the ending made no sense.",
]
inputs = tokenizer(examples, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
logits = model(**{k: v.to(model.device) for k, v in inputs.items()}).logits
for text, label_id in zip(examples, logits.argmax(dim=-1).tolist()):
    print(f"{model.config.id2label[label_id]:>8} ← {text}")

trainer.save_model(str(BASE_DIR / "model"))
tokenizer.save_pretrained(str(BASE_DIR / "model"))
print("Модель сохранена в hw10/model")

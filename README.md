# AI Text Detection

Проект - система детекции сгенерированных текстов на основе автокодировщика.

Как работает:
1. Тексты разбиваются на предложения и кодируются в эмбеддинги.
2. Из эмбеддингов строятся эмбеддинги через комбинацию пуллингов (mean pooling + mean diff pooling).
3. Автоэнкодер обучается только на человеческих текстах - AI-тексты он восстанавливает хуже, поэтому высокая ошибка реконструкции = признак AI.
4. Порог классификации подбирается по ROC-кривой (или может задаваться), гиперпараметры - через Optuna.

Что умеет:

- Генерировать и скачивать датасеты (IvyPanda, ASAP2, PERSUADE + AI-эссе).
- Обучать модель с поиском гиперпараметров, сохранять результаты и загружать на HuggingFace.
- Отдавать предсказания через REST API (FastAPI) — на вход список текстов, на выходе score + метка Human/AI.
- Деплоиться в Kubernetes (Yandex Cloud) через автоматизированный bash-скрипт.

Взаимодействие с пользователем происходит через плагин для браузера (Chrome) - AI Text Checker.

Стек: PyTorch, FastAPI, Optuna, Hydra, sentence-transformers, Kubernetes/Yandex Cloud.

---

## AI Text Checker. Инструкция по установке

1. Открой страницу расширений

В адресной строке Chrome введи:
```
chrome://extensions/
```

2. Включи режим разработчика

В правом верхнем углу страницы переключи тумблер **Режим разработчика**.

3. Загрузи расширение

Нажми кнопку **Загрузить распакованное расширение** и выбери распакованную папку `./plugins/ai-checker-extension`.

### Использование

1. Выдели весь текст
2. Кликни **правой кнопкой мыши**
3. Выбери пункт **Проверить на ИИ**
4. В центре экрана появится результат:
    - **AI** — текст написан нейросетью
    - **HUMAN** — текст написан человеком

---

## Локальный запуск

```bash
uv sync

docker compose up --build

curl http://localhost:8080/health
# {"status":"ok"}

curl -X POST http://localhost:8080/score \
-H "Content-Type: application/json" \
-d '{"texts": ["Just text"], "threshold": 0.0008}'
# {"results":[{"text":"Just text","score":0.0007377,"label":"Human"}],"model":"all-mpnet-base-v2","threshold":0.0008}
```

---

## Справка по командам

```bash
(m2gpu) anonim-macbook-14 :: ~/IdeaProjects/ai-text-detection ‹main*› » uv run commands.py --help

Available commands:

  generate_ai          →  Generate AI essays using prompt strategies
  download_datasets    →  Parse combined_essays.jsonl and split into IvyPanda / ASAP2 / PERSUADE CSVs
  generate_embeddings  →  Generate embeddings of the datasets
  train                →  Train model
```

## Генерация ии-текстов

### Базовый запуск (Mistral, все 5 стратегий)
```bash
uv run commands.py generate_ai
```

### Сменить модель на LLaMA
```bash
uv run commands.py generate_ai model=llama
```

### Только стратегии 1, 3 и 5
```bash
uv run commands.py generate_ai generation.strategies=[1,3,5]
```

### Обработать 500 тем начиная с 1000-й
```bash
uv run commands.py generate_ai run.start=1000 run.count=500
```

### Комбинировать
```bash
uv run commands.py generate_ai model=llama run.count=100 generation.temperature=0.9
```

## Деплой в Яндекс Облако.
```bash
./deploy.sh
```

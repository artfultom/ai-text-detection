# Скачивание набора текстов, написанных человеком

TODO

# Генерация ии-текстов

## Базовый запуск (Mistral, все 5 стратегий)
```bash
uv run commands.py generate_ai
```

## Сменить модель на LLaMA
```bash
uv run commands.py generate_ai model=llama
```

## Только стратегии 1, 3 и 5
```bash
uv run commands.py generate_ai generation.strategies=[1,3,5]
```

## Обработать 500 тем начиная с 1000-й
```bash
uv run commands.py generate_ai run.start=1000 run.count=500
```

## Комбинировать
```bash
uv run commands.py generate_ai model=llama run.count=100 generation.temperature=0.9
```

## Помощь
```bash
uv run commands.py --help
```

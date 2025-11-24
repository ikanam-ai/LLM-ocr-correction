# LLM OCR Correction

## Инструкция по использованию скриптов

### Описание:
- Набор вспомогательных скриптов для корректировки и оценки OCR-результатов с использованием LLM и вспомогательных утилит.

### Файлы и назначение:
- `ocr_llm_correction.py`: основной скрипт для коррекции результатов OCR с помощью LLM. Читает папку с результатами OCR и записывает скорректированные варианты.
- `compute_ocr_metrics.py`: скрипт для подсчёта метрик OCR (до/после коррекции). Принимает директорию (или одиночный файл) с корректированными результатами и сохраняет CSV с метриками.
- `ocr_prompt.py`: промпты, используемый для вызовов LLM.
- `serve_model.sh`: скрипт для локального запуска модели.
- `.env`: файл с настройками окружения (для OpenAI API).

### Запуск моделей Qwen-3-VL:

1. Установить менеджер пакетов [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Создать и активировать виртуальное окружение:
```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
```
3. Установить vLLM:
```bash
uv pip install -U vllm
uv pip install qwen-vl-utils==0.0.14
```
4. Запустить модель с помощью скрипта:
```bash
source serve_model.sh
```

### Запуск скриптов:
- Перед запуском нужно создать виртуальное окружение и установить зависимости (можно также использовать uv):

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

### Примеры использования:
- Запуск коррекции OCR-результатов (пример):

```bash
python ocr_llm_correction.py --input train_data/ocr_results/ --output-dir corrected_results
```

- Подсчёт метрик после коррекции (пример):

```bash
python compute_ocr_metrics.py --corrected-dir corrected_results/ --output ocr_summary.csv
```

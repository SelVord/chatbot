# 🤖 RAG Chatbot

Чатбот с поддержкой собственных PDF-документов. Отвечает только на основе загруженных данных — не придумывает.

## 📦 Технологии

| Компонент | Библиотека |
|-----------|-----------|
| UI | Streamlit |
| RAG-цепочка | LangChain |
| Векторный поиск | FAISS |
| LLM (платно) | OpenAI GPT-4o-mini |
| LLM (бесплатно) | Ollama (llama3.2) |
| Эмбеддинги (платно) | OpenAI text-embedding-3-small |
| Эмбеддинги (бесплатно) | sentence-transformers / all-MiniLM-L6-v2 |
| База данных | SQLite |
| PDF-парсинг | PyPDF |

---

## 🚀 Установка

### 1. Клонируйте / распакуйте проект

```bash
cd rag_chatbot
```

### 2. Создайте виртуальное окружение

```bash
python -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows
```

### 3. Установите зависимости

```bash
pip install -r requirements.txt
```

> ⚠️ Для бесплатных локальных эмбеддингов (`local`) может потребоваться:
> ```bash
> pip install torch sentence-transformers
> ```

### 4. Настройте `.env`

```bash
cp .env.example .env
```

Откройте `.env` и заполните нужные поля:

```env
# Если используете OpenAI:
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_PROVIDER=openai

# Если используете Ollama (бесплатно, локально):
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
EMBEDDING_PROVIDER=local
```

### 5. (Опционально) Установите Ollama для локальной работы

```bash
# Установка: https://ollama.com
ollama pull llama3.2
ollama serve
```

### 6. Запустите приложение

```bash
streamlit run app.py
```

Откроется браузер по адресу **http://localhost:8501**

---

## 🗂️ Структура проекта

```
rag_chatbot/
├── app.py                         # Streamlit-приложение
├── config.py                      # Конфигурация
├── requirements.txt
├── .env.example
│
├── core/
│   ├── database.py                # SQLite: сессии, сообщения, документы
│   ├── document_processor.py      # Загрузка и разбивка PDF
│   ├── embeddings.py              # Фабрика эмбеддингов
│   ├── llm.py                     # Фабрика LLM
│   ├── rag_chain.py               # RAG-пайплайн
│   └── vector_store.py            # FAISS: создание / загрузка / обновление
│
└── data/
    ├── chat_history.db            # SQLite база (создаётся автоматически)
    ├── uploads/                   # Загруженные PDF
    └── vector_stores/             # FAISS-индексы по сессиям
```

---

## 💡 Использование

1. **Создайте сессию** — нажмите `➕` в боковом меню и введите название.
2. **Настройте бота** — задайте имя и системный промпт (инструкцию).
3. **Загрузите PDF** — нажмите `Индексировать файлы`.
4. **Общайтесь** — задавайте вопросы в поле ввода.
5. **Просматривайте источники** — под каждым ответом есть раскрывающийся блок со ссылками на страницы.

---

## ⚙️ Параметры конфигурации (`config.py`)

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `CHUNK_SIZE` | 1000 | Размер чанка (символов) |
| `CHUNK_OVERLAP` | 200 | Перекрытие между чанками |
| `TOP_K_DOCS` | 5 | Сколько чанков извлекать |
| `RELEVANCE_THRESHOLD` | 0.30 | Порог схожести (0–1) |
| `MAX_HISTORY_PAIRS` | 4 | Пар сообщений в контексте |

---

## 🔒 Особенности

- **Только по документам** — бот отказывается отвечать, если релевантность ниже порога.
- **Несколько сессий** — разные базы знаний для разных задач.
- **Персистентность** — FAISS-индексы и история сохраняются на диск.
- **Экспорт** — история чата скачивается в JSON.
- **Гибкость** — OpenAI или Ollama, платные или бесплатные эмбеддинги — всё переключается.

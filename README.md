# RAG Knowledge Base — MCP‑сервер

MCP‑сервер, который превращает папку с документами в поисковую базу знаний для AI‑агента в IDE. Внутри — Corrective RAG на LangGraph, гибридный поиск (BM25 + вектор → Reciprocal Rank Fusion) и внешняя LLM: Ollama или любой OpenAI‑совместимый API (LM Studio, vLLM, llama.cpp server).

- Архитектура и принятые решения — [ARCHITECTURE.md](ARCHITECTURE.md)
- Дневник разработки — [REPORT.md](REPORT.md)

## Быстрый старт

Нужны Docker с Docker Compose и запущенная LLM (см. ниже).

```bash
git clone <repo> && cd <repo>
docker compose up --build
```

Первая сборка образа занимает несколько минут: в образ скачивается модель эмбеддингов `intfloat/multilingual-e5-large` (~2,2 ГБ). После этого сервер работает без доступа в интернет.

MCP‑эндпоинт: **`http://localhost:8000/mcp`**, проверка живости: `http://localhost:8000/health`.

> [!IMPORTANT]
> **Индексация выполняется в фоне.** После запуска база знаний пустая. `index_folder("./sample_docs")` сразу возвращает `state: running`, а сами эмбеддинги считаются в фоне: на CPU для демо‑базы (~700 КБ, ~1100 чанков) это **около 10–12 минут**. Прогресс и оставшееся время показывает `index_status` (`chunks_done / chunks_total`, `eta_s`). Пока индексация не дошла до `state: completed`, `ask_question` и `find_relevant_docs` отвечают по уже проиндексированной части и добавляют в ответ поле `warning` с процентом готовности: ответ может быть неполным.
>
> Поэтому запустите индексацию сразу после старта сервера — до того, как задавать агенту вопросы по базе. Индекс хранится в Docker‑томе `rag-data`: после перезапуска контейнера повторная индексация не нужна, а повторный `index_folder` пропускает неизменённые файлы и занимает секунды.

### LLM

LLM нужна только инструменту `ask_question`; индексация и поиск работают без неё. Сервер стартует и при недоступной LLM, а `index_status` показывает её состояние.

| Вариант | Что сделать |
|---|---|
| **Ollama на хосте** (по умолчанию) | `ollama pull qwen2.5:3b`, затем `docker compose up`. Ollama должна слушать не только localhost: `OLLAMA_HOST=0.0.0.0` |
| **Ollama в Docker** (профиль `local-llm`) | `LLM_BASE_URL=http://ollama:11434 docker compose --profile local-llm up --build` — поднимет Ollama и скачает модель |
| **LM Studio / другой OpenAI‑совместимый сервер** | создать `.env` (см. ниже) и `docker compose up` |

Пример `.env` для LM Studio на хосте с qwen3:

```dotenv
LLM_PROVIDER=openai
LLM_BASE_URL=http://host.docker.internal:1234/v1
LLM_MODEL=qwen/qwen3-4b
LLM_PROMPT_SUFFIX=/no_think
```

Все параметры с описанием — в [.env.example](.env.example).

## Подключение к IDE

**VS Code (Copilot, режим агента).** Конфиг уже лежит в репозитории — [.vscode/mcp.json](.vscode/mcp.json):

```json
{
  "servers": {
    "rag-knowledge-base": { "type": "http", "url": "http://localhost:8000/mcp" }
  }
}
```

Откройте папку проекта в VS Code, в панели Chat выберите режим **Agent** и убедитесь, что в списке инструментов включён сервер `rag-knowledge-base`.

**Claude Code:** `claude mcp add --transport http rag-knowledge-base http://localhost:8000/mcp`

**MCP Inspector:** `npx @modelcontextprotocol/inspector`, транспорт *Streamable HTTP*, URL `http://localhost:8000/mcp`. Для `ask_question` увеличьте *Request Timeout* (по умолчанию 10 с) или включите *Reset Timeout on Progress* — сервер шлёт уведомления о прогрессе после каждого шага графа.

## Инструменты

| Инструмент | Вход | Что делает |
|---|---|---|
| `index_status` | — | Количество файлов и чанков, типы файлов, время последней индексации, прогресс фоновой индексации, модель эмбеддингов, доступность LLM |
| `index_folder` | `path`, `glob="**/*"` | Индексирует папку (`.md .txt .py .js .ts .json .yaml`). Большие папки индексируются в фоне: результат `state: running`, прогресс — в `index_status` |
| `find_relevant_docs` | `query`, `top_k=5` | Гибридный поиск без LLM: чанки с файлом, строками, разделом, RRF‑score и рангами BM25/вектора |
| `ask_question` | `question` | Corrective RAG: переформулировка → поиск → оценка чанков → (расширение запроса и повтор, до 2 раз) → ответ со ссылками на источники |

## Сценарий проверки

1. `index_status()` → `"empty": true`
2. `index_folder("./sample_docs")` → инструмент сразу возвращает `state: running`; индексация ~700 КБ демо‑документов идёт **в фоне около 10–12 минут на CPU** (см. сноску в «Быстром старте»)
3. `index_status()` → прогресс (`chunks_done / chunks_total`, `percent`, `eta_s`); дождитесь `"state": "completed"` — после этого `files`, `chunks`, `files_by_type`, `last_indexed_at`
4. `ask_question("Какая минимальная стоимость заказа в мастерской «Ржавый гусь»?")` → ответ с источниками
5. `find_relevant_docs("PURGE_LINE_LENGTH_MM")` → чанк из `sample_docs/lab/tools/purge.py`

Свои документы: положите путь к папке в `DOCS_DIR` (в `.env` или в окружении) и вызовите `index_folder("./docs")`.

## Демо‑база знаний (`sample_docs/`, ~700 КБ)

| Папка | Содержимое | Язык | Источник |
|---|---|---|---|
| `wiki/` | 3DPrint.Wiki: принтеры Anet, Bambu Lab, Creality, Snapmaker, Wanhao…, электроника, прошивки | RU | перевод на русский, CC BY‑SA 4.0 |
| `klipper/` | документация Klipper (выравнивание стола, датчики, pressure advance, резонансы, TMC, FAQ) и модули `klippy/extras/*.py` | EN | Klipper3d/klipper, GPLv3 |
| `lab/` | внутренняя база вымышленной мастерской 3D‑печати «Ржавый гусь»: регламенты, инциденты, парк принтеров, профили филаментов, утилиты, логи | RU + EN | сгенерировано ИИ для проекта |

`lab/` покрывает все форматы: `.md`, `.txt`, `.py`, `.js`, `.ts`, `.json`, `.yaml`.

## Проверочные факты

Эти факты придуманы для проекта и есть только в `sample_docs/lab/` — их нет в публичных источниках, поэтому правильный ответ возможен только при чтении базы знаний. Часть фактов записана по‑английски, часть — по‑русски: спрашивать можно на любом языке.

> [!NOTE]
> Спрашивайте агента о фактах после того, как фоновая индексация `./sample_docs` завершилась (`index_status` → `"state": "completed"`, около 10–12 минут на CPU). Если спросить раньше, ответ будет построен по частичному индексу, а сервер добавит к нему предупреждение с процентом готовности и оставшимся временем.

| # | Вопрос | Ожидаемый ответ | Где лежит |
|---|---|---|---|
| 1 | Когда и кем основана мастерская «Ржавый гусь»? | 3 февраля 2019 года, Глеб Сорокин | `lab/history.md` |
| 2 | Как зовут талисман мастерской? | Гусь Шуруп | `lab/README.md`, `lab/history.md` |
| 3 | Какое значение e-steps у принтера «Сойка»? | 97.3 шага/мм | `lab/fleet.yaml` |
| 4 | При какой температуре печатать и сушить «Дымчатый кварц» PETG (Smoky Quartz PETG)? | Сопло 236 °C, стол 83 °C; сушка 6 ч при 61 °C | `lab/filaments.json`, `lab/procedures/filament_storage.md` |
| 5 | Какая длина линии очистки сопла (purge line)? | 143 мм (`PURGE_LINE_LENGTH_MM`) | `lab/tools/purge.py` |
| 6 | Минимальная стоимость заказа и цена за грамм? | 740 ₽, 4,35 ₽/г, срочный заказ ×1,85 | `lab/tools/pricing.js`, `lab/faq_clients.md` |
| 7 | Как часто меняют сопла на принтерах мастерской? | Каждые 312 часов печати | `lab/tools/maintenance.ts`, `lab/fleet.yaml` |
| 8 | Сколько максимум может идти ночная печать без присмотра? | 9 ч 40 мин, чек‑лист NS‑7 | `lab/procedures/night_shift.md` |
| 9 | Что случилось 17 марта 2025 года на принтере «Кедр»? | «Синее облако»: выпал термистор, блок нагрелся до ~287 °C, ущерб 18 600 ₽ | `lab/incidents/2025-03-17-blue-cloud.md`, `lab/logs/klippy_kedr_2025-03-17.txt` |
| 10 | Какой размер и допуск у калибровочного «Гусиного кубика» (Goose Cube)? | 23,7 мм, допуск ±0,08 мм | `lab/calibration/goose_cube.md` |
| 11 | Когда можно забрать заказ и какая гарантия? | По средам и субботам 11:00–19:30; гарантия 21 день | `lab/faq_clients.md`, `lab/tools/order_status.ts` |
| 12 | На каком слое упало задание #4367 и почему? | На 212‑м слое, «спагетти» — деталь оторвалась от стола | `lab/logs/print_log_2025-09.txt` |
| 13 | Почему 2 августа 2025 года «Мамонт» врезался соплом в стол? | Перебит провод нормально разомкнутого концевика Z; заменён на оптический нормально замкнутый | `lab/incidents/2025-08-02-mamont-z-crash.md` |

## Локальная разработка

```bash
uv sync                    # Python 3.12, зависимости из uv.lock
uv run pytest              # тесты: без сети и без LLM
uv run ruff check . && uv run ruff format --check .
uv run rag-kb              # сервер на http://localhost:8000/mcp
```

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) на каждый push в `main` и pull request запускает те же `ruff` и `pytest`, затем собирает Docker‑образ и проверяет `/health` запущенного контейнера.

При локальном запуске с включённым системным прокси или VPN задайте `NO_PROXY=localhost,127.0.0.1`: иначе запросы Python‑клиентов к localhost могут уйти в прокси.

## Лицензии демо‑документов

- `sample_docs/wiki/` — перевод 3DPrint.Wiki, CC BY‑SA 4.0 ([ATTRIBUTION.md](sample_docs/wiki/ATTRIBUTION.md))
- `sample_docs/klipper/` — Klipper, GPLv3 ([ATTRIBUTION.md](sample_docs/klipper/ATTRIBUTION.md), [COPYING](sample_docs/klipper/COPYING))
- `sample_docs/lab/` — сгенерировано ИИ для этого проекта, все люди, места и события вымышлены

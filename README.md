# TFG-LLM-RE

Optimizacion de la respuesta de los LLMs en el proceso de Ingenieria de Requisitos.

TFG de Ingenieria de Software que evalua modelos LLM locales (Qwen 2.5 7B, Llama 3.1 8B, Llama 3.2 3B) frente a modelos via API (NVIDIA NIM: Llama 3.1 70B, Llama 3.1 8B, Mistral 7B) en 5 tareas de ingenieria de requisitos, comparando 5 estrategias de prompting.

## Ejecucion local

### Requisitos

- Python 3.10+
- [Ollama](https://ollama.ai) (para modelos locales)
- Cuenta en [NVIDIA NIM](https://build.nvidia.com/) (opcional, para modelos API)

### Instalacion

```bash
git clone https://github.com/antsuebae/TFG-LLM-RE.git
cd TFG-LLM-RE
./run.sh setup
```

Descarga los modelos locales con Ollama:

```bash
ollama pull qwen2.5:7b-instruct-q5_K_M
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull llama3.2:3b-instruct-q4_K_M
```

Para los modelos API, crea un archivo `.env` en la raiz del proyecto:

```
NVIDIA_API_KEY=tu_clave_aqui
```

### Uso

#### Interfaz web (Streamlit)

```bash
./run.sh streamlit
```

Abre `http://localhost:8501`. Desde ahi puedes:

- **Pipeline Documento**: Sube un fichero de requisitos (.txt, .csv, .md, .pdf) y ejecuta el analisis completo con uno o varios modelos y estrategias. Genera informe con clasificacion F/NF, ambiguedad, completitud, testabilidad e inconsistencias, y permite reescribir los requisitos defectuosos.
- **Clasificar Requisito**: Clasifica un requisito individual como Funcional o No Funcional.
- **Analizar Calidad**: Analiza ambiguedad, completitud o testabilidad de un requisito.
- **Validar Consistencia**: Compara dos requisitos para detectar inconsistencias.
- **Resultados Experimentos**: Dashboard con tres pestanas: historial del pipeline, resultados de benchmark (heatmaps, boxplots, radar, F1 vs tiempo) y comparativa Local vs API.

#### Pipeline por linea de comandos

```bash
# Analizar un documento de requisitos
./run.sh pipeline requisitos.txt

# Con modelo y estrategia especificos
./run.sh pipeline requisitos.pdf --model llama8b --strategy chain_of_thought

# Saltar analisis de inconsistencias (mas rapido)
./run.sh pipeline requisitos.csv --skip-inconsistency
```

#### Experimentos (benchmark)

```bash
# Dry-run para verificar configuracion
./run.sh experiment --dry-run --task classification --models qwen7b

# Experimento completo
./run.sh experiment --task classification --models qwen7b llama8b --strategies few_shot chain_of_thought

# Reanudar desde checkpoint
./run.sh experiment --task classification --resume results/checkpoints/checkpoint_classification_XXXXX.json

# Generar graficos y analisis estadistico
./run.sh analysis results/experiments/results_classification_XXXXX.csv

# Analisis de todas las tareas a la vez
./run.sh analysis --all --results-dir results/experiments/v2
```

#### API REST

```bash
./run.sh api
```

Documentacion interactiva en `http://localhost:8000/docs`.

#### Verificar dependencias

```bash
./run.sh check
```

---

## Ejecucion con Docker

Alternativa que no requiere instalar Python, Ollama ni dependencias en el sistema host. Solo requiere Docker.

### Requisitos

- [Docker](https://docs.docker.com/get-docker/) con Docker Compose

### Inicio rapido

```bash
# 1. Construir la imagen
./run.sh docker-build

# 2. Arrancar todos los servicios (modo CPU, sin GPU)
./run.sh docker-up

# 3. Descargar modelos en el contenedor (solo la primera vez, ~15 min)
./run.sh docker-models
```

Los servicios quedan accesibles en:

| Servicio | URL |
|----------|-----|
| Streamlit | http://localhost:8501 |
| API REST | http://localhost:8000/docs |

Ollama corre internamente en la red Docker (no expuesto al host). Para acceder a Ollama desde el host mientras Docker esta activo, usa el Ollama local del sistema.

### Con GPU

```bash
./run.sh docker-up-gpu
```

Si `nvidia-container-toolkit` no esta instalado, el script lo detecta e instala automaticamente segun la distribucion (Fedora/RHEL, Ubuntu/Debian, openSUSE, Arch). Requiere `sudo`.

### Gestion de servicios

```bash
./run.sh docker-down          # detener todos los servicios
./run.sh docker-logs          # ver logs en tiempo real
./run.sh docker-logs app      # logs solo del frontend
./run.sh docker-logs api      # logs solo de la API
```

Los modelos de Ollama se almacenan en un volumen Docker nombrado (`ollama_data`) y persisten entre reinicios. Los resultados de experimentos se montan desde `./results/` del host.

---

## Estructura del proyecto

```
config/                  Configuracion de experimentos (YAML)
data/                    Datasets de evaluacion (CSV)
src/
  app.py                 Frontend Streamlit (6 paginas)
  pipeline.py            Pipeline de analisis de documentos
  experiment.py          Pipeline de experimentos con checkpoint/resume
  prompts.py             25 prompts (5 tareas x 5 estrategias) + parsers
  models.py              Wrappers para Ollama y NVIDIA NIM con reintentos
  metrics.py             Metricas de evaluacion (F1, accuracy, precision, recall)
  analysis.py            Graficos y analisis estadistico (ANOVA, t-test, Cohen's d)
  api.py                 API REST (FastAPI)
  clean_datasets.py      Limpieza y normalizacion de datasets
Dockerfile               Imagen Docker para app y API
docker-compose.yml       Orquestacion de servicios (Ollama + Streamlit + API)
results/                 Resultados generados (excluido de git)
  experiments/           CSVs de resultados de experimentos
  checkpoints/           Checkpoints temporales
  analysis/              Graficos y tablas generadas
  logs/                  Logs de ejecucion
  pipeline/              Resultados del pipeline de documentos
```

## Modelos evaluados

| Modelo | Tipo | Backend | Parametros |
|--------|------|---------|-----------|
| Qwen 2.5 7B Instruct (Q5) | Local | Ollama | 7B |
| Llama 3.1 8B Instruct (Q4) | Local | Ollama | 8B |
| Llama 3.2 3B Instruct (Q4) | Local | Ollama | 3B |
| Llama 3.1 70B Instruct | API | NVIDIA NIM | 70B |
| Llama 3.1 8B Instruct | API | NVIDIA NIM | 8B |
| Mistral 7B v0.3 Instruct | API | NVIDIA NIM | 7B |

## Estrategias de prompting

| Estrategia | Descripcion |
|------------|-------------|
| Question Refinement (QR) | Reformula la pregunta antes de responder |
| Cognitive Verifier (CV) | Descompone el problema en preguntas auxiliares |
| Persona + Context (PC) | Asigna rol de experto en RE al modelo |
| Few-Shot (FS) | Incluye ejemplos resueltos en el prompt |
| Chain of Thought (CoT) | Razonamiento explicito paso a paso |

## Tareas de RE evaluadas

| Codigo | Tarea | Dataset | Muestras |
|--------|-------|---------|---------|
| A1 | Clasificacion F/NF | promise_nfr_v2.csv | 625 |
| A2 | Deteccion de ambiguedad | ambiguity_dataset_v2.csv | 262 |
| A3 | Evaluacion de completitud | completeness_dataset_v2.csv | 150 |
| V1 | Deteccion de inconsistencias | inconsistency_dataset.csv | 30 pares |
| V2 | Evaluacion de testabilidad | testability_dataset_v2.csv | 97 |

## Diseno experimental

- **Factorial:** 5 tareas × 6 modelos × 5 estrategias × 5 iteraciones = 750 configuraciones
- **Seeds:** [42, 123, 456, 789, 1024]
- **Temperatura:** 0.4
- **Hardware:** Intel i7-13650HX, NVIDIA RTX 4060 Max-Q 8 GB, 32 GB RAM, Fedora Linux 43

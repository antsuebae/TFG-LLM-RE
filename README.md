# TFG-LLM-RE

Optimizacion de la respuesta de los LLMs en el proceso de Ingenieria de Requisitos.

TFG de Ingenieria de Software que evalua modelos LLM locales (Qwen 7B, Llama 8B) frente a modelos via API (NVIDIA NIM: Llama 70B, Llama 8B, Mistral 7B) en 5 tareas de ingenieria de requisitos, comparando 5 estrategias de prompting.

## Requisitos previos

- Python 3.10+
- [Ollama](https://ollama.ai) (para modelos locales)
- Cuenta en [NVIDIA NIM](https://build.nvidia.com/) (opcional, para modelos API)

## Instalacion

```bash
git clone https://github.com/antsuebae/TFG-LLM-RE.git
cd TFG-LLM-RE
./run.sh setup
```

Para los modelos locales, descarga los modelos con Ollama:

```bash
ollama pull qwen2.5-coder:7b-instruct-q5_K_M
ollama pull llama3.1:8b-instruct-q4_K_M
```

Para los modelos API, crea un archivo `.env` en la raiz del proyecto:

```
NVIDIA_API_KEY=tu_clave_aqui
```

## Uso

### Interfaz web (Streamlit)

```bash
./run.sh streamlit
```

Abre `http://localhost:8501`. Desde ahi puedes:

- **Pipeline Documento**: Sube un fichero de requisitos (.txt, .csv, .md, .pdf) y ejecuta el analisis completo con uno o varios modelos y estrategias. Genera informe con clasificacion F/NF, ambiguedad, completitud, testabilidad e inconsistencias. Permite generar un documento con los requisitos corregidos.
- **Clasificar Requisito**: Clasifica un requisito individual como Funcional o No Funcional.
- **Analizar Calidad**: Analiza ambiguedad, completitud o testabilidad de un requisito.
- **Validar Consistencia**: Compara dos requisitos para detectar inconsistencias.
- **Resultados**: Dashboard con ejecuciones guardadas del pipeline y resultados de experimentos.
- **Comparar Modelos**: Tabla comparativa de rendimiento entre modelos locales y API.

### Pipeline por linea de comandos

```bash
# Analizar un documento de requisitos
./run.sh pipeline data/revolution_fest_requisitos.txt

# Con modelo y estrategia especificos
./run.sh pipeline data/dataset_ejemplo.pdf --model llama8b --strategy chain_of_thought

# Saltar analisis de inconsistencias (mas rapido)
./run.sh pipeline requisitos.csv --skip-inconsistency
```

### Experimentos (benchmark)

```bash
# Dry-run para verificar
./run.sh experiment --dry-run --task classification --models qwen7b

# Ejecutar experimento completo
./run.sh experiment --task classification --models qwen7b llama8b --strategies few_shot chain_of_thought

# Generar graficos y analisis estadistico
./run.sh analysis results/results_classification_XXXXX.csv
```

### API REST

```bash
./run.sh api
```

Documentacion en `http://localhost:8000/docs`.

## Estructura

```
config/                  Configuracion de experimentos
data/                    Datasets (PROMISE NFR + 4 datasets anotados)
src/
  app.py                 Frontend Streamlit
  pipeline.py            Pipeline de analisis de documentos
  experiment.py          Pipeline de experimentos (benchmark)
  prompts.py             25 prompts (5 tareas x 5 estrategias) + parsers
  models.py              Wrappers para Ollama y NVIDIA NIM
  metrics.py             Metricas de evaluacion
  analysis.py            Graficos y analisis estadistico
  api.py                 API REST (FastAPI)
results/                 Resultados generados (excluido de git)
```

## Modelos

| Modelo | Tipo | Backend |
|--------|------|---------|
| Qwen 2.5 Coder 7B | Local | Ollama |
| Llama 3.1 8B | Local | Ollama |
| Llama 3.1 70B | API | NVIDIA NIM |
| Llama 3.1 8B | API | NVIDIA NIM |
| Mistral 7B | API | NVIDIA NIM |

## Estrategias de prompting

| Estrategia | Descripcion |
|------------|-------------|
| Question Refinement (QR) | Reformula la pregunta antes de responder |
| Cognitive Verifier (CV) | Analisis paso a paso con verificacion |
| Persona + Context (PC) | Asigna rol de experto al modelo |
| Few-Shot (FS) | Incluye ejemplos resueltos en el prompt |
| Chain of Thought (CoT) | Razonamiento explicito paso a paso |

## Tareas de RE

1. **Clasificacion F/NF** - Funcional vs No Funcional
2. **Deteccion de ambiguedad** - Identifica terminos vagos, pronombres y cuantificadores
3. **Evaluacion de completitud** - Detecta elementos faltantes
4. **Deteccion de inconsistencias** - Busca contradicciones entre pares de requisitos
5. **Evaluacion de testabilidad** - Determina si un requisito es verificable

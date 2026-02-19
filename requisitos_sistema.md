# Especificacion de Requisitos del Sistema

## Pipeline de Evaluacion Comparativa de LLMs para Analisis de Requisitos

**Proyecto:** TFG - Evaluacion de LLMs Locales vs APIs en Ingenieria de Requisitos
**Autor:** [Tu Nombre]
**Fecha:** Febrero 2026
**Version:** 2.0

---

## 1. Introduccion

### 1.1 Proposito
Este documento especifica los requisitos del sistema de experimentacion y analisis que permite evaluar y comparar el rendimiento de modelos LLM locales (Qwen 7B, Llama 8B) frente a modelos via API (NVIDIA NIM: Llama 70B, Llama 8B, Mistral 7B) en 5 tareas de ingenieria de requisitos de software: clasificacion F/NF, deteccion de ambiguedad, evaluacion de completitud, deteccion de inconsistencias y evaluacion de testabilidad.

### 1.2 Alcance
El sistema abarca:
- Carga y procesamiento de datasets de requisitos (PROMISE + 4 datasets anotados)
- Pipeline de analisis de documentos de requisitos (clasificacion, ambiguedad, completitud, testabilidad, inconsistencias)
- Ejecucion automatizada de experimentos con multiples modelos, tareas y estrategias de prompting
- Recoleccion y almacenamiento de resultados con metadata estructurada
- Calculo de metricas de evaluacion (binarias y multi-clase)
- Generacion de reportes, visualizaciones comparativas y analisis estadistico
- Generacion de documento de requisitos corregido (reescritura automatica)
- Frontend web interactivo (Streamlit) con 6 paginas
- API REST (FastAPI) para analisis de requisitos

### 1.3 Definiciones y Acronimos
| Termino | Definicion |
|---------|------------|
| LLM | Large Language Model |
| RE | Requirements Engineering (Ingenieria de Requisitos) |
| F/NF | Funcional / No Funcional |
| PROMISE | Dataset academico de requisitos (Cleland-Huang et al., 2007) |
| QR | Question Refinement (estrategia de prompt) |
| CV | Cognitive Verifier (estrategia de prompt) |
| PC | Persona + Context (estrategia de prompt) |
| FS | Few-Shot (estrategia de prompt con ejemplos) |
| CoT | Chain of Thought (estrategia de prompt con razonamiento paso a paso) |
| NIM | NVIDIA Inference Microservices (API de inferencia en la nube) |

---

## 2. Descripcion General del Sistema

### 2.1 Perspectiva del Producto
Sistema desarrollado en Python que combina un pipeline experimental de linea de comandos con un frontend web interactivo (Streamlit) y una API REST (FastAPI). Se ejecuta en hardware local (RTX 4060 8GB VRAM) utilizando Ollama para modelos locales y la API de NVIDIA NIM para modelos en la nube.

### 2.2 Funcionalidades Principales
1. **Gestion de Datasets:** Carga, validacion y muestreo de requisitos desde CSV, TXT, MD y PDF
2. **Analisis Multi-tarea:** 5 tareas de RE (clasificacion, ambiguedad, completitud, inconsistencias, testabilidad)
3. **Multi-estrategia:** 5 estrategias de prompting (QR, CV, PC, FS, CoT)
4. **Pipeline de Documentos:** Analisis completo de un documento de requisitos con informe de calidad
5. **Ejecucion de Experimentos:** Diseno factorial automatizado con checkpoint/resume
6. **Procesamiento de Respuestas:** Parsing y validacion de salidas por tarea
7. **Calculo de Metricas:** Precision, Recall, F1-Score, Accuracy (binarias y multi-clase)
8. **Puntuacion de Calidad:** Quality score 0-100 por requisito
9. **Generacion de Documento Corregido:** Reescritura automatica de requisitos con problemas
10. **Generacion de Reportes:** Tablas comparativas, graficos, analisis estadistico (ANOVA, t-tests, Cohen's d)
11. **Frontend Web Interactivo:** Streamlit con 6 paginas (pipeline, clasificacion, calidad, consistencia, resultados, comparacion)
12. **API REST:** Endpoints para clasificacion, analisis, elicitacion y validacion de requisitos

### 2.3 Usuarios
| Usuario | Descripcion |
|---------|-------------|
| Investigador | Ejecuta experimentos, analiza resultados, configura parametros |
| Ingeniero de requisitos | Usa el pipeline/frontend para analizar documentos de requisitos y obtener informes de calidad |
| Revisor/Tutora | Consulta resultados y reportes generados |

### 2.4 Restricciones
- Hardware: RTX 4060 con 8GB VRAM
- Modelos locales cuantizados (Q4/Q5) para ajustarse a memoria disponible
- NVIDIA NIM: API gratuita con limites de rate (reintentos exponenciales implementados)

### 2.5 Dependencias
- Ollama (gestion modelos locales)
- NVIDIA NIM API (modelos en la nube)
- Python 3.14
- Streamlit (frontend web)
- FastAPI + Uvicorn (API REST)
- pypdf (extraccion de requisitos desde PDF)
- Librerias: pandas, numpy, scipy, matplotlib, seaborn, scikit-learn, tqdm, openai, python-dotenv

---

## 3. Requisitos Funcionales

### RF-01: Carga de Datasets
**Descripcion:** El sistema debe cargar datasets de requisitos desde archivos CSV/JSON.
**Entrada:** Ruta al archivo del dataset
**Salida:** Dataset cargado en memoria con requisitos y etiquetas
**Criterio de aceptacion:** Dataset PROMISE (621 requisitos) carga en menos de 5 segundos

### RF-02: Validacion de Datasets
**Descripcion:** El sistema debe validar que los datasets contengan los campos requeridos (texto, etiqueta).
**Criterio de aceptacion:** Detecta y reporta requisitos sin etiqueta o con campos vacios

### RF-03: Muestreo Aleatorio Reproducible
**Descripcion:** El sistema debe generar muestras aleatorias de requisitos usando seeds fijos.
**Entrada:** Tamano de muestra (n=50), seed (42, 123, 456, 789, 1024)
**Salida:** Subconjunto de n requisitos
**Criterio de aceptacion:** Misma seed produce identica muestra en ejecuciones diferentes

### RF-04: Construccion de Prompts Multi-estrategia
**Descripcion:** El sistema debe construir prompts segun la tarea y la estrategia seleccionada (QR, CV, PC, FS, CoT).
**Entrada:** Tarea (classification, ambiguity, completeness, inconsistency, testability), estrategia, texto del requisito
**Salida:** Prompt formateado listo para enviar al modelo
**Criterio de aceptacion:** 25 combinaciones validas (5 tareas x 5 estrategias) generan prompts correctos

### RF-05: Invocacion de Modelos Locales
**Descripcion:** El sistema debe invocar modelos locales via Ollama API.
**Entrada:** Modelo (qwen2.5-coder:7b, llama3.1:8b), prompt, temperatura
**Salida:** Respuesta del modelo en texto con tiempo y tokens/s
**Criterio de aceptacion:** Timeout configurable, reintentos ante fallos de conexion

### RF-06: Invocacion de Modelos NVIDIA NIM
**Descripcion:** El sistema debe invocar modelos via la API de NVIDIA NIM (Llama 70B, Llama 8B, Mistral 7B).
**Entrada:** Prompt, temperatura, API key (desde .env)
**Salida:** Respuesta del modelo con tiempo y tokens/s
**Criterio de aceptacion:** Manejo de rate limits con reintentos exponenciales y errores de API

### RF-07: Parsing de Respuestas Multi-tarea
**Descripcion:** El sistema debe extraer la respuesta estructurada de cada tarea desde las respuestas de los modelos.
**Entrada:** Respuesta en texto del modelo, nombre de la tarea
**Salida:**
- Classification: F, NF o INVALID
- Ambiguity: is_ambiguous (bool), ambiguity_type (vague_term/pronoun/quantifier/none), ambiguous_words (list)
- Completeness: is_complete (bool), missing_elements (list)
- Inconsistency: is_inconsistent (bool), description (str)
- Testability: is_testable (bool), reason (measurable/vague/subjective)
**Criterio de aceptacion:** Parsing exitoso en >95% de respuestas validas para cada tarea

### RF-08: Almacenamiento de Resultados
**Descripcion:** El sistema debe guardar resultados de cada evaluacion en formato estructurado.
**Salida:** Archivo CSV/JSON con: modelo, estrategia, tarea, iteracion, requisito_id, prediccion, ground_truth, tiempo_respuesta
**Criterio de aceptacion:** Resultados persistidos antes de iniciar siguiente evaluacion

### RF-09: Calculo de Metricas
**Descripcion:** El sistema debe calcular metricas de clasificacion por configuracion.
**Entrada:** Predicciones y etiquetas reales
**Salida:** Precision, Recall, F1-Score, Accuracy (binarias y multi-clase segun tarea)
**Criterio de aceptacion:** Calculos verificados contra implementacion de scikit-learn

### RF-10: Ejecucion de Experimento Factorial
**Descripcion:** El sistema debe ejecutar el diseno factorial completo: 5 tareas x 5 modelos x 5 estrategias x 5 iteraciones.
**Entrada:** Configuracion del experimento (modelos, estrategias, tareas, seeds)
**Salida:** 125 conjuntos de resultados por tarea (5 modelos x 5 estrategias x 5 iteraciones)
**Criterio de aceptacion:** Ejecucion completa sin intervencion manual, checkpoint/resume ante fallos

### RF-11: Generacion de Tablas Comparativas
**Descripcion:** El sistema debe generar tablas resumen de metricas por modelo y estrategia.
**Salida:** Tabla en formato Markdown/CSV
**Criterio de aceptacion:** Incluye media y desviacion estandar de 5 iteraciones

### RF-12: Generacion de Graficos
**Descripcion:** El sistema debe generar visualizaciones de resultados.
**Salida:** Graficos de barras, boxplots, heatmaps, radar, velocidad (PNG/PDF)
**Criterio de aceptacion:** Graficos legibles, con leyendas y etiquetas claras

### RF-13: Analisis Estadistico
**Descripcion:** El sistema debe realizar tests estadisticos de significancia.
**Salida:** Resultados de ANOVA, t-tests pareados, Cohen's d
**Criterio de aceptacion:** p-valores y tamanos de efecto calculados correctamente

### RF-14: Logging de Ejecucion
**Descripcion:** El sistema debe registrar eventos durante la ejecucion.
**Salida:** Logs con timestamps, errores y progreso (modulo logging + tqdm)
**Criterio de aceptacion:** Suficiente informacion para diagnosticar fallos

### RF-15: Configuracion por Archivo
**Descripcion:** El sistema debe permitir configuracion via archivo YAML.
**Entrada:** Archivo experiment_config.yaml (modelos, estrategias, tareas, seeds, rutas)
**Criterio de aceptacion:** Cambios de configuracion sin modificar codigo fuente

### RF-16: Pipeline de Analisis de Documentos
**Descripcion:** El sistema debe permitir cargar un documento de requisitos y ejecutar las 5 tareas de analisis (clasificacion, ambiguedad, completitud, testabilidad, inconsistencias), generando un informe completo de calidad.
**Entrada:** Documento de requisitos (TXT, MD, CSV o PDF), modelo, estrategia
**Salida:** Informe en HTML, Markdown y CSV con metricas de calidad por requisito
**Criterio de aceptacion:** Pipeline ejecuta las 5 tareas secuencialmente y genera informes en los 3 formatos

### RF-17: Carga de Documentos PDF
**Descripcion:** El sistema debe extraer requisitos desde documentos PDF, soportando formato REM-CMS (campos Descripcion) y texto libre.
**Entrada:** Archivo PDF con requisitos
**Salida:** Lista de requisitos extraidos como texto
**Criterio de aceptacion:** Extrae correctamente campos Descripcion de PDFs REM-CMS; para otros PDFs, extrae lineas de texto

### RF-18: Deteccion de Ambiguedad
**Descripcion:** El sistema debe detectar ambiguedad en requisitos, clasificando el tipo (vague_term, pronoun, quantifier) e identificando las palabras ambiguas.
**Entrada:** Texto del requisito
**Salida:** is_ambiguous (bool), ambiguity_type, ambiguous_words
**Criterio de aceptacion:** Detecta terminos vagos, pronombres sin referencia y cuantificadores imprecisos

### RF-19: Evaluacion de Completitud
**Descripcion:** El sistema debe evaluar si un requisito esta completo, identificando elementos faltantes (error_handling, boundary_conditions, acceptance_criteria, preconditions).
**Entrada:** Texto del requisito
**Salida:** is_complete (bool), missing_elements (list)
**Criterio de aceptacion:** Identifica correctamente al menos 4 tipos de elementos faltantes

### RF-20: Deteccion de Inconsistencias
**Descripcion:** El sistema debe detectar inconsistencias entre pares de requisitos, describiendo el conflicto encontrado.
**Entrada:** Dos textos de requisitos
**Salida:** is_inconsistent (bool), description del conflicto
**Criterio de aceptacion:** Analiza pares de requisitos y detecta contradicciones directas e indirectas

### RF-21: Evaluacion de Testabilidad
**Descripcion:** El sistema debe evaluar si un requisito es testable (verificable objetivamente), clasificando la razon (measurable, vague, subjective).
**Entrada:** Texto del requisito
**Salida:** is_testable (bool), reason
**Criterio de aceptacion:** Distingue entre requisitos medibles, vagos y subjetivos

### RF-22: Calculo de Puntuacion de Calidad
**Descripcion:** El sistema debe calcular una puntuacion de calidad (quality_score, 0-100) para cada requisito basada en los resultados de ambiguedad, completitud y testabilidad.
**Entrada:** Resultados de analisis del requisito
**Salida:** Puntuacion numerica 0-100
**Criterio de aceptacion:** Penalizaciones: -30 por ambiguo, -30 por incompleto (+ hasta -15 por elementos faltantes), -25 por no testable

### RF-23: Frontend Web Interactivo
**Descripcion:** El sistema debe proporcionar un frontend web con Streamlit con 6 paginas: Pipeline Documento, Clasificar Requisito, Analizar Calidad, Validar Consistencia, Resultados Experimentos y Comparar Modelos.
**Entrada:** Interaccion del usuario via navegador
**Salida:** Visualizaciones interactivas, tablas, metricas y botones de descarga
**Criterio de aceptacion:** Todas las paginas son funcionales y permiten seleccionar modelos y estrategias

### RF-24: API REST para Analisis de Requisitos
**Descripcion:** El sistema debe exponer una API REST (FastAPI) con endpoints para clasificacion (/classify), analisis de defectos (/analyze), elicitacion (/elicit), validacion de consistencia (/validate) y analisis de documento (/analyze-document).
**Entrada:** Peticiones HTTP con texto de requisitos y modelo seleccionado
**Salida:** Respuestas JSON estructuradas
**Criterio de aceptacion:** Todos los endpoints responden correctamente con modelos locales y NVIDIA NIM

### RF-25: Seleccion de Multiples Estrategias por Ejecucion
**Descripcion:** El sistema debe permitir seleccionar y ejecutar multiples estrategias de prompting en una misma ejecucion del pipeline, con ejecucion secuencial o paralela.
**Entrada:** Lista de estrategias seleccionadas, modo de ejecucion (secuencial/paralelo)
**Salida:** Resultados por cada combinacion modelo x estrategia
**Criterio de aceptacion:** El frontend permite seleccionar multiples modelos y estrategias simultaneamente

### RF-26: Generacion de Documento de Requisitos Corregido
**Descripcion:** El sistema debe reescribir automaticamente los requisitos que presenten problemas (ambiguedad, incompletitud, no testabilidad), generando un documento corregido descargable.
**Entrada:** Resultados del analisis, modelo para reescritura
**Salida:** Documento Markdown/TXT con requisitos corregidos, indicando original, corregido y cambios realizados
**Criterio de aceptacion:** Solo reescribe requisitos con problemas detectados; mantiene la intencion original

### RF-27: Almacenamiento Estructurado con Metadata
**Descripcion:** El sistema debe guardar cada ejecucion del pipeline en un subdirectorio con results.csv, informe.html, informe.md, inconsistencias.json y metadata.json.
**Entrada:** Resultados del pipeline
**Salida:** Directorio organizado con todos los artefactos
**Criterio de aceptacion:** metadata.json contiene timestamp, documento, modelo, estrategia, n_requirements y resumen de metricas

### RF-28: Dashboard de Resultados con Comparacion
**Descripcion:** El sistema debe mostrar un dashboard de resultados que permita visualizar y comparar ejecuciones previas del pipeline y experimentos, con tablas resumen, heatmaps de calidad y detalle por ejecucion.
**Entrada:** Ejecuciones guardadas en results/pipeline/
**Salida:** Tabla comparativa de ejecuciones, detalle expandible y descarga de informes
**Criterio de aceptacion:** Permite seleccionar multiples ejecuciones para comparacion lado a lado

---

## 4. Requisitos No Funcionales

### RNF-01: Rendimiento - Tiempo de Respuesta Local
**Descripcion:** Los modelos locales deben generar respuestas en tiempo razonable.
**Metrica:** < 30 segundos por requisito para modelos de 7-8B parametros
**Prioridad:** Alta

### RNF-02: Rendimiento - Uso de Memoria
**Descripcion:** El sistema debe operar dentro de los limites de VRAM disponible.
**Metrica:** < 7GB VRAM para modelos cuantizados
**Prioridad:** Alta

### RNF-03: Fiabilidad - Tolerancia a Fallos
**Descripcion:** El sistema debe recuperarse de fallos sin perder progreso.
**Metrica:** Checkpoint por cada configuracion (modelo x estrategia x iteracion), resume automatico desde checkpoint JSON
**Prioridad:** Alta

### RNF-04: Fiabilidad - Consistencia
**Descripcion:** Ejecuciones con mismos parametros deben producir resultados identicos.
**Metrica:** Varianza 0 en clasificaciones con temperatura 0
**Prioridad:** Alta

### RNF-05: Usabilidad - Facilidad de Ejecucion
**Descripcion:** El sistema debe ejecutarse con comandos simples via CLI y, ademas, ofrecer un frontend Streamlit accesible desde el navegador.
**Metrica:** Maximo 3 comandos para ejecutar experimento completo; frontend accesible con `streamlit run app.py`
**Prioridad:** Media

### RNF-06: Usabilidad - Documentacion
**Descripcion:** El sistema debe incluir documentacion de uso.
**Metrica:** README con instrucciones de instalacion, ejecucion y descripcion de la estructura del proyecto
**Prioridad:** Media

### RNF-07: Mantenibilidad - Modularidad
**Descripcion:** El codigo debe estar organizado en modulos independientes.
**Metrica:** Separacion clara: modelos, prompts, metricas, pipeline, analisis, API, frontend
**Prioridad:** Media

### RNF-08: Portabilidad - Dependencias
**Descripcion:** Las dependencias deben ser facilmente instalables.
**Metrica:** Archivo requirements.txt funcional
**Prioridad:** Media

### RNF-09: Seguridad - API Keys
**Descripcion:** Las credenciales no deben estar en el codigo fuente.
**Metrica:** API key via variable de entorno o archivo .env (excluido de git)
**Prioridad:** Alta

### RNF-10: Escalabilidad - Nuevos Modelos
**Descripcion:** Debe ser facil anadir nuevos modelos al experimento.
**Metrica:** Anadir modelo requiere solo cambio en configuracion (experiment_config.yaml)
**Prioridad:** Baja

### RNF-11: Rendimiento - API NIM
**Descripcion:** El sistema debe gestionar los rate limits de la API de NVIDIA NIM con reintentos exponenciales.
**Metrica:** Reintentos automaticos con backoff exponencial (max 3 reintentos); tiempo maximo de espera configurable
**Prioridad:** Alta

### RNF-12: Rendimiento - Frontend
**Descripcion:** El frontend Streamlit debe responder con baja latencia para operaciones que no involucren LLMs.
**Metrica:** < 2 segundos para carga de paginas, navegacion y operaciones de lectura de resultados
**Prioridad:** Media

---

## 5. Casos de Uso

### CU-01: Ejecutar Experimento Multi-tarea

**Actor:** Investigador
**Precondiciones:**
- Datasets disponibles (PROMISE + 4 anotados)
- Modelos locales instalados en Ollama
- API key de NVIDIA NIM configurada en .env

**Flujo Principal:**
1. El investigador configura los parametros del experimento (tareas, modelos, estrategias, iteraciones) via CLI o archivo YAML
2. El investigador ejecuta el comando de inicio (e.g., `python experiment.py --task classification --models qwen7b nim_llama70b`)
3. El sistema carga y valida el dataset correspondiente a la tarea
4. El sistema itera por cada configuracion (modelo x estrategia x iteracion):
   - Genera muestra aleatoria con seed correspondiente
   - Construye prompts segun tarea y estrategia (5 tareas x 5 estrategias = 25 combinaciones)
   - Invoca modelo y obtiene respuestas
   - Parsea la respuesta segun la tarea
   - Almacena resultados y guarda checkpoint
5. El sistema calcula metricas agregadas
6. El sistema genera tablas y graficos comparativos
7. El sistema presenta resumen de resultados

**Flujo Alternativo - Error en Modelo:**
4a. Si el modelo falla, el sistema reintenta con backoff exponencial (max 3 reintentos)
4b. Si persiste el fallo, registra error y continua con siguiente requisito

**Postcondiciones:**
- Resultados guardados en archivos CSV con checkpoint JSON
- Metricas calculadas y almacenadas
- Reportes visuales generados

---

### CU-02: Consultar Resultados de Experimento

**Actor:** Investigador / Tutora
**Precondiciones:** Experimento o pipeline ejecutado previamente

**Flujo Principal:**
1. El usuario accede al dashboard de resultados via Streamlit o carga ficheros directamente
2. El sistema lista ejecuciones disponibles (pipeline y experimentos)
3. El sistema muestra tabla comparativa de metricas
4. El sistema presenta graficos de comparacion (barras, boxplots, heatmaps, radar, velocidad)
5. El usuario puede seleccionar multiples ejecuciones para comparacion lado a lado
6. El usuario puede descargar informes HTML, CSV y Markdown

**Postcondiciones:** Resultados visualizados/exportados

---

### CU-03: Anadir Nueva Estrategia de Prompt

**Actor:** Investigador
**Precondiciones:** Sistema instalado

**Flujo Principal:**
1. El investigador crea plantilla de la nueva estrategia en prompts.py
2. El investigador registra la estrategia en experiment_config.yaml
3. El sistema valida que la estrategia esta definida para las 5 tareas
4. La estrategia queda disponible para futuros experimentos

**Postcondiciones:** Nueva estrategia disponible en el sistema

---

### CU-04: Analizar Requisito Individual

**Actor:** Investigador / Ingeniero de requisitos
**Precondiciones:** Al menos un modelo disponible

**Flujo Principal:**
1. El usuario accede a la pagina correspondiente del frontend (Clasificar, Calidad o Consistencia)
2. El usuario introduce texto de uno o dos requisitos
3. El usuario selecciona modelo y estrategia
4. El sistema construye prompt y envia al modelo
5. El sistema muestra resultado parseado (clasificacion, ambiguedad, completitud, testabilidad o inconsistencia)
6. El usuario puede comparar cambiando modelo o estrategia

**Postcondiciones:** Analisis del requisito obtenido y mostrado

---

### CU-05: Generar Reporte Final

**Actor:** Investigador
**Precondiciones:** Todos los experimentos completados

**Flujo Principal:**
1. El investigador solicita generacion de reporte final
2. El sistema agrega resultados de todos los experimentos
3. El sistema ejecuta analisis estadistico (ANOVA, t-tests, Cohen's d)
4. El sistema genera tablas de resultados
5. El sistema genera graficos comparativos
6. El sistema compila reporte en formato Markdown/PDF

**Postcondiciones:** Reporte listo para inclusion en memoria del TFG

---

### CU-06: Analizar Documento de Requisitos

**Actor:** Ingeniero de requisitos
**Precondiciones:** Al menos un modelo disponible, documento de requisitos preparado

**Flujo Principal:**
1. El usuario accede a la pagina "Pipeline Documento" del frontend Streamlit
2. El usuario sube un documento (TXT, MD, CSV o PDF) o selecciona uno existente
3. El sistema extrae los requisitos del documento (soporte REM-CMS para PDF)
4. El usuario selecciona uno o varios modelos y estrategias
5. El usuario selecciona modo de ejecucion (secuencial o paralelo)
6. El sistema ejecuta las 5 tareas de analisis sobre cada requisito
7. El sistema calcula quality_score por requisito
8. El sistema muestra tabla resumen, detalle por requisito, requisitos con problemas e inconsistencias
9. El usuario puede guardar informes (HTML, Markdown, CSV, metadata.json)

**Flujo Alternativo - Multiples Modelos:**
4a. Si se seleccionan multiples modelos/estrategias, el sistema muestra tabla comparativa modelo x estrategia
4b. El sistema muestra heatmap de calidad media por combinacion

**Postcondiciones:** Informe de calidad completo generado y opcionalmente guardado

---

### CU-07: Generar Documento de Requisitos Corregido

**Actor:** Ingeniero de requisitos
**Precondiciones:** Pipeline de analisis ejecutado (CU-06)

**Flujo Principal:**
1. El usuario selecciona el modelo/estrategia a usar para la correccion
2. El usuario pulsa "Generar Documento Corregido"
3. El sistema identifica requisitos con problemas (ambiguos, incompletos, no testables)
4. El sistema reescribe cada requisito con problemas usando el prompt REWRITE_PROMPT
5. El sistema muestra original vs corregido con lista de cambios realizados
6. El usuario descarga el documento corregido en formato Markdown o TXT

**Postcondiciones:** Documento de requisitos corregido descargable

---

### CU-08: Comparar Resultados entre Modelos y Estrategias

**Actor:** Investigador
**Precondiciones:** Al menos un archivo de resultados de experimentos disponible

**Flujo Principal:**
1. El usuario accede a la pagina "Comparar Modelos" del frontend
2. El sistema carga resultados y calcula metricas por configuracion
3. El sistema muestra tabla de rendimiento por modelo (F1, Accuracy, Precision, Recall, tiempo, tokens/s)
4. El sistema muestra mejor estrategia por modelo
5. El sistema muestra comparacion Local vs API (metricas promedio, tiempos)
6. El sistema presenta tabla de trade-offs (privacidad, coste, latencia, escalabilidad, disponibilidad)

**Postcondiciones:** Comparacion visualizada

---

## 6. Historias de Usuario

### HU-01: Ejecucion Automatizada
**Como** investigador
**Quiero** ejecutar todos los experimentos de forma automatizada
**Para** no tener que supervisar manualmente cada evaluacion

**Criterios de aceptacion:**
- Puedo lanzar el experimento completo con un solo comando
- El sistema continua aunque haya errores puntuales
- Puedo ver el progreso en tiempo real (tqdm)
- Los resultados se guardan automaticamente

---

### HU-02: Reproducibilidad
**Como** investigador
**Quiero** que los experimentos sean reproducibles
**Para** validar resultados y permitir replicacion por otros investigadores

**Criterios de aceptacion:**
- Los seeds aleatorios estan fijos y documentados
- Misma configuracion produce mismos resultados
- Versiones de modelos y librerias quedan registradas

---

### HU-03: Comparacion Visual
**Como** investigador
**Quiero** visualizar graficamente las diferencias entre modelos
**Para** identificar rapidamente patrones y tendencias

**Criterios de aceptacion:**
- Graficos de barras comparando F1-Score por modelo
- Boxplots mostrando variabilidad entre iteraciones
- Heatmaps de metricas por modelo x estrategia
- Graficos exportables en alta resolucion

---

### HU-04: Analisis de Costos y Trade-offs
**Como** investigador
**Quiero** conocer los trade-offs entre modelos locales y API
**Para** evaluar la viabilidad de cada enfoque

**Criterios de aceptacion:**
- Registro de tokens/s y tiempo de respuesta por modelo
- Comparacion local vs API en rendimiento, privacidad y coste
- Tabla de trade-offs en el dashboard

---

### HU-05: Tolerancia a Interrupciones
**Como** investigador
**Quiero** poder pausar y reanudar experimentos
**Para** no perder progreso si hay cortes de energia o necesito usar el equipo

**Criterios de aceptacion:**
- Sistema guarda checkpoint por cada configuracion completada
- Puedo reanudar desde ultimo checkpoint con --resume
- No se repiten evaluaciones ya completadas

---

### HU-06: Prueba Rapida de Configuracion
**Como** investigador
**Quiero** probar una configuracion con pocos requisitos
**Para** validar que todo funciona antes de ejecutar el experimento completo

**Criterios de aceptacion:**
- Modo "dry-run" con 5 requisitos
- Verificacion de conectividad con modelos
- Validacion de formato de salida

---

### HU-07: Deteccion de Anomalias
**Como** investigador
**Quiero** que el sistema detecte respuestas anomalas
**Para** identificar problemas en los modelos o prompts

**Criterios de aceptacion:**
- Alerta si modelo no devuelve respuesta valida (INVALID)
- Registro de respuestas que no pudieron parsearse
- Estadisticas de tasa de error por modelo

---

### HU-08: Exportacion de Datos
**Como** investigador
**Quiero** exportar resultados en multiples formatos
**Para** usar los datos en otras herramientas de analisis

**Criterios de aceptacion:**
- Exportacion a CSV para analisis en Excel/R
- Exportacion a JSON (metadata, inconsistencias)
- Informes en HTML y Markdown
- Descarga directa desde el frontend

---

### HU-09: Analisis de Documento Completo
**Como** ingeniero de requisitos
**Quiero** subir un documento de requisitos y obtener un informe completo de calidad
**Para** identificar problemas de ambiguedad, completitud y testabilidad en mis requisitos

**Criterios de aceptacion:**
- Puedo subir documentos TXT, MD, CSV o PDF
- El sistema ejecuta las 5 tareas de analisis sobre cada requisito
- Obtengo un quality_score (0-100) por requisito y una media global
- Puedo descargar el informe en HTML, Markdown y CSV

---

### HU-10: Correccion Automatica de Requisitos
**Como** ingeniero de requisitos
**Quiero** que el sistema reescriba automaticamente los requisitos con problemas
**Para** obtener una version mejorada de mi documento sin tener que corregir manualmente cada requisito

**Criterios de aceptacion:**
- Solo reescribe requisitos con problemas detectados
- Muestra original vs corregido con los cambios realizados
- Mantiene la intencion original del requisito
- Puedo descargar el documento corregido en Markdown o TXT

---

### HU-11: Comparacion Multi-estrategia
**Como** investigador
**Quiero** ejecutar varias estrategias de prompting a la vez sobre el mismo documento
**Para** comparar cual produce mejores resultados de analisis

**Criterios de aceptacion:**
- Puedo seleccionar multiples modelos y estrategias en una sola ejecucion
- El sistema muestra tabla comparativa de resultados por combinacion
- El sistema muestra heatmap de calidad media modelo x estrategia
- Puedo ejecutar en modo secuencial o paralelo

---

## 7. Diagrama de Arquitectura

```
+-----------------------------------------------------------+
|                    CAPA DE PRESENTACION                    |
|                                                           |
|  +---------------------------+  +----------------------+  |
|  |  Frontend Streamlit       |  |  API REST FastAPI    |  |
|  |  (6 paginas)              |  |  /classify           |  |
|  |  - Pipeline Documento     |  |  /analyze            |  |
|  |  - Clasificar Requisito   |  |  /elicit             |  |
|  |  - Analizar Calidad       |  |  /validate           |  |
|  |  - Validar Consistencia   |  |  /analyze-document   |  |
|  |  - Resultados Experimentos|  |  /health             |  |
|  |  - Comparar Modelos       |  |                      |  |
|  +---------------------------+  +----------------------+  |
+-----------------------------------------------------------+
                          |
                          v
+-----------------------------------------------------------+
|                    CAPA DE LOGICA                          |
|                                                           |
|  +---------------------------+  +----------------------+  |
|  |  Pipeline de Documentos   |  |  Pipeline de         |  |
|  |  (pipeline.py)            |  |  Experimentos        |  |
|  |  - Carga docs (CSV/PDF)   |  |  (experiment.py)     |  |
|  |  - Analisis multi-tarea   |  |  - Factorial 5x5x5   |  |
|  |  - Quality score          |  |  - Checkpoint/Resume |  |
|  |  - Reescritura            |  |  - Multi-tarea       |  |
|  |  - Informes HTML/MD       |  |                      |  |
|  +---------------------------+  +----------------------+  |
|                                                           |
|  +---------------------------+  +----------------------+  |
|  |  Constructor de Prompts   |  |  Calculador de       |  |
|  |  (prompts.py)             |  |  Metricas            |  |
|  |  5 tareas x 5 estrategias |  |  (metrics.py)        |  |
|  |  Parsers por tarea        |  |  Precision, Recall,  |  |
|  |  Prompt de reescritura    |  |  F1, Accuracy        |  |
|  +---------------------------+  +----------------------+  |
|                                                           |
|  +---------------------------+                            |
|  |  Analisis y Reportes      |                            |
|  |  (analysis.py)            |                            |
|  |  Graficos, ANOVA,         |                            |
|  |  t-tests, Cohen's d       |                            |
|  +---------------------------+                            |
+-----------------------------------------------------------+
                          |
                          v
+-----------------------------------------------------------+
|                 CAPA DE MODELOS                           |
|                                                           |
|  +---------------------------+  +----------------------+  |
|  |  Modelos Locales          |  |  Modelos API         |  |
|  |  (Ollama)                 |  |  (NVIDIA NIM)        |  |
|  |  - Qwen 7B (Q5)          |  |  - Llama 70B         |  |
|  |  - Llama 8B (Q4)         |  |  - Llama 8B          |  |
|  |                           |  |  - Mistral 7B        |  |
|  |  models.py: reintentos,   |  |  models.py: backoff  |  |
|  |  tokens/s, timeouts       |  |  exponencial, API key|  |
|  +---------------------------+  +----------------------+  |
+-----------------------------------------------------------+
                          |
                          v
+-----------------------------------------------------------+
|                 CAPA DE DATOS                             |
|                                                           |
|  +---------------------------+  +----------------------+  |
|  |  Datasets                 |  |  Resultados          |  |
|  |  data/                    |  |  results/            |  |
|  |  - promise_nfr.csv (250)  |  |  - results_*.csv     |  |
|  |  - ambiguity_dataset (50) |  |  - checkpoint_*.json |  |
|  |  - completeness_data (50) |  |  - pipeline/         |  |
|  |  - inconsistency_data(30) |  |    - metadata.json   |  |
|  |  - testability_data  (50) |  |    - informe.html    |  |
|  +---------------------------+  |    - informe.md      |  |
|                                 |    - results.csv     |  |
|  +---------------------------+  +----------------------+  |
|  |  Configuracion            |                            |
|  |  config/                  |                            |
|  |  - experiment_config.yaml |                            |
|  |  - .env (API keys)        |                            |
|  +---------------------------+                            |
+-----------------------------------------------------------+
```

---

## 8. Matriz de Trazabilidad

| Requisito | Caso de Uso | Historia de Usuario |
|-----------|-------------|---------------------|
| RF-01, RF-02 | CU-01, CU-06 | HU-06 |
| RF-03 | CU-01 | HU-02 |
| RF-04 | CU-01, CU-03, CU-04, CU-06 | HU-06, HU-11 |
| RF-05 | CU-01, CU-04, CU-06 | HU-01 |
| RF-06 | CU-01, CU-04, CU-06 | HU-01, HU-04 |
| RF-07 | CU-01, CU-04, CU-06 | HU-07 |
| RF-08 | CU-01, CU-02 | HU-05, HU-08 |
| RF-09 | CU-01, CU-05, CU-08 | HU-03 |
| RF-10 | CU-01 | HU-01, HU-05 |
| RF-11, RF-12, RF-13 | CU-02, CU-05, CU-08 | HU-03, HU-08 |
| RF-14 | CU-01, CU-06 | HU-07 |
| RF-15 | CU-01, CU-03 | HU-06 |
| RF-16 | CU-06 | HU-09 |
| RF-17 | CU-06 | HU-09 |
| RF-18 | CU-04, CU-06 | HU-09 |
| RF-19 | CU-04, CU-06 | HU-09 |
| RF-20 | CU-04, CU-06 | HU-09 |
| RF-21 | CU-04, CU-06 | HU-09 |
| RF-22 | CU-06 | HU-09 |
| RF-23 | CU-02, CU-04, CU-06, CU-07, CU-08 | HU-03, HU-09, HU-10, HU-11 |
| RF-24 | CU-04 | HU-08 |
| RF-25 | CU-06 | HU-11 |
| RF-26 | CU-07 | HU-10 |
| RF-27 | CU-06 | HU-08, HU-09 |
| RF-28 | CU-02, CU-08 | HU-03, HU-11 |
| RNF-01, RNF-02 | CU-01, CU-06 | HU-01 |
| RNF-03 | CU-01 | HU-05 |
| RNF-04 | CU-01 | HU-02 |
| RNF-05 | CU-01, CU-06 | HU-06, HU-09 |
| RNF-09 | CU-01, CU-06 | - |
| RNF-11 | CU-01, CU-06 | HU-01 |
| RNF-12 | CU-02, CU-06 | HU-09 |

---

## 9. Priorizacion de Requisitos (MoSCoW)

### Must Have (Imprescindibles)
- RF-01, RF-03, RF-04, RF-05, RF-06, RF-07, RF-08, RF-09, RF-10
- RF-12, RF-13, RF-16, RF-18, RF-19, RF-20, RF-21, RF-22
- RNF-01, RNF-02, RNF-03, RNF-09, RNF-11

### Should Have (Importantes)
- RF-02, RF-11, RF-14, RF-15, RF-23, RF-25, RF-26, RF-27
- RNF-04, RNF-05, RNF-12

### Could Have (Deseables)
- RF-17, RF-24, RF-28
- RNF-06, RNF-07, RNF-08

### Won't Have (Fuera de alcance)
- RNF-10 (escalabilidad avanzada a decenas de modelos)
- Integracion con herramientas externas de gestion de requisitos (JIRA, DOORS)

---

## 10. Apendice: Plantillas de Prompts

### Estrategia Question Refinement (QR) - Ejemplo: Clasificacion
```
Clasifica el siguiente requisito de software como F (Funcional) o NF (No Funcional).

Si la clasificacion no es clara, reformula mentalmente la pregunta para
entender mejor el requisito antes de clasificar.

Requisito: "[TEXT]"

Responde SOLO con una de estas opciones exactas:
- F (si es un requisito funcional - describe QUE debe hacer el sistema)
- NF (si es un requisito no funcional - describe COMO debe comportarse el sistema)

Clasificacion:
```

### Estrategia Cognitive Verifier (CV) - Ejemplo: Clasificacion
```
Analiza el siguiente requisito de software paso a paso:

Requisito: "[TEXT]"

Pasos de analisis:
1. Identifica el sujeto principal del requisito
2. Determina si describe una FUNCION (que hace) o una CUALIDAD (como lo hace)
3. Considera: Las funciones son acciones concretas, las cualidades son
   caracteristicas del sistema

Basandote en tu analisis, clasifica como:
- F (Funcional): describe una funcion o capacidad especifica
- NF (No Funcional): describe rendimiento, seguridad, usabilidad, u otra cualidad

Clasificacion (responde solo F o NF):
```

### Estrategia Persona + Context (PC) - Ejemplo: Clasificacion
```
Eres un ingeniero de requisitos senior con 10 anos de experiencia en
IEEE 830 e ISO 29148.

Tu tarea es clasificar requisitos de software en dos categorias:
- F (Funcional): Requisitos que describen QUE debe hacer el sistema
- NF (No Funcional): Requisitos que describen COMO debe ser el sistema

Requisito a clasificar: "[TEXT]"

Como experto, proporciona tu clasificacion (responde unicamente F o NF):
```

### Estrategia Few-Shot (FS) - Ejemplo: Clasificacion
```
Clasifica el requisito como F (Funcional) o NF (No Funcional).

Ejemplos:
- "The system shall allow users to login" -> F
- "The system shall respond within 2 seconds" -> NF
- "The system shall generate monthly reports" -> F
- "The system shall be available 99.9% of the time" -> NF
- "The system shall send email notifications" -> F

Requisito: "[TEXT]"
Clasificacion:
```

### Estrategia Chain of Thought (CoT) - Ejemplo: Clasificacion
```
Clasifica el requisito como F (Funcional) o NF (No Funcional).
Piensa paso a paso:
1. Identifica el sujeto del requisito
2. Determina si describe una ACCION/FUNCION o una CUALIDAD/RESTRICCION
3. Las funciones son F, las cualidades son NF
Muestra tu razonamiento y luego da la clasificacion final como una sola
letra: F o NF.

Requisito: "[TEXT]"

Razonamiento:
```

### Prompt de Reescritura de Requisitos
```
Eres un ingeniero de requisitos senior. Tu tarea es reescribir un
requisito de software para corregir los problemas de calidad detectados.

Requisito original: "[TEXT]"

Problemas detectados:
[LISTA DE PROBLEMAS]

Instrucciones:
- Reescribe el requisito corrigiendo TODOS los problemas listados
- Manten la intencion original del requisito
- Hazlo especifico, medible, completo y testable
- Si es ambiguo, reemplaza terminos vagos por valores concretos
- Si es incompleto, anade los elementos faltantes
- Si no es testable, anade criterios medibles
- Responde SOLO con el requisito reescrito, sin explicaciones adicionales

Requisito corregido:
```

---

*Documento actualizado a la Version 2.0 para reflejar el estado real del sistema implementado*

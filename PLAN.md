# Plan: Swap gemma9b → llama3.2:3b + ampliar datasets

## 1. Configuración del modelo

### 1.1 Descargar Llama 3.2 3B en Ollama
```bash
ollama pull llama3.2:3b-instruct-q4_K_M
```
Verificar que funciona: `ollama run llama3.2:3b-instruct-q4_K_M "Hello"`

### 1.2 Actualizar configuración en 3 sitios
Reemplazar gemma9b por llama3b en:

- **`config/experiment_config.yaml`** → sección `models.local`
  ```yaml
  - name: "llama3.2:3b-instruct-q4_K_M"
    short_name: "llama3b"
    type: "ollama"
  ```
  Eliminar la entrada de gemma9b.

- **`src/app.py`** → `MODEL_CONFIGS` y `MODEL_LABELS`
  ```python
  "llama3b": {"name": "llama3.2:3b-instruct-q4_K_M", "type": "ollama", "short_name": "llama3b"},
  ```
  ```python
  "llama3b": "Llama 3.2 3B (local)",
  ```
  Eliminar las entradas de gemma9b.

- **`src/dag.py`** → si tiene su propia copia de MODEL_CONFIGS, actualizarlo igual.
  (Idealmente refactorizar para importar de un solo sitio, pero no es bloqueante.)

### 1.3 Verificar prompts
Ejecutar un dry-run rápido para confirmar que Llama 3.2 3B sigue el formato de respuesta:
```bash
cd src && python experiment.py --dry-run --task classification --models llama3b
```
Si el invalid_rate es alto (>30%), puede necesitar ajuste de temperatura o max_tokens.

---

## 2. Ampliar datasets

### 2.1 PROMISE_exp (clasificación F/NF)
- **Fuente:** https://zenodo.org/records/12805484
- **Tamaño:** 969 muestras anotadas (F/NF)
- **Acción:** Descargar, mapear columnas a `id,text,label,label_binary`, reemplazar `data/promise_nfr.csv`
- **Verificar:** que `label_binary` contenga solo "F" y "NF"
- **Actualizar** `experiment_config.yaml` → `tasks.classification.sample_size` (subir a 100 o null si se quiere usar todo)

### 2.2 ReqEval (ambigüedad)
- **Fuente:** https://huggingface.co/datasets/metaeval/reqeval-ambiguity-detection
- **Tamaño:** ~200 muestras con anotación de ambigüedad
- **Acción:** Descargar, mapear a `id,text,is_ambiguous,ambiguity_type,ambiguous_words`, reemplazar `data/ambiguity_dataset.csv`
- **Verificar:** que `is_ambiguous` sea True/False (no 0/1 ni yes/no)

### 2.3 Completeness, testability, inconsistency
- No hay datasets públicos conocidos para estas tareas → se mantienen los actuales (50, 50, 30 muestras).
- Esto es una **contribución del TFG** (se documenta en la memoria).

---

## 3. Re-ejecutar experimentos

### 3.1 Diseño factorial actualizado
- **Modelos locales:** llama3b, qwen7b, llama8b (3)
- **Modelos API:** nim_llama70b, nim_llama8b, nim_mistral (3)
- **Estrategias:** 5
- **Iteraciones:** 5
- **Seeds:** [42, 123, 456, 789, 1024]

Total: 6 modelos × 5 estrategias × 5 iteraciones × 5 tareas = **750 configs**

### 3.2 Orden de ejecución sugerido
1. **Dry-run llama3b** (todas las tareas) — validar que funciona
2. **Local fase completa** (llama3b + qwen7b + llama8b, todas las tareas)
3. **API fase completa** (nim_llama70b + nim_llama8b + nim_mistral, todas las tareas)

Nota: los resultados de Qwen7b y Llama8b de la fase anterior NO se reutilizan porque los datasets cambiaron. Hay que re-ejecutar todo.

### 3.3 Estructura de resultados
```
results/
  experimento_v2/
    local/
      results_classification_*.csv
      results_ambiguity_*.csv
      ...
    api/
      results_classification_*.csv
      ...
```

---

## 4. Tokens/s (ya corregido)
El fix en `models.py` (completion_tokens en vez de total_tokens para NIM) ya está aplicado.
Los nuevos experimentos usarán la métrica corregida automáticamente.

---

## 5. Análisis
Tras completar los experimentos:
1. Copiar todos los CSVs (local + api) a una carpeta unificada
2. Ejecutar `analysis.py` con `--all` para generar gráficos y tablas
3. Los resultados piloto (gemma9b, datasets pequeños) se guardan aparte como referencia

---

## Resumen de cambios en código
| Archivo | Cambio |
|---------|--------|
| `config/experiment_config.yaml` | gemma9b → llama3b |
| `src/app.py` | MODEL_CONFIGS + MODEL_LABELS: gemma9b → llama3b |
| `src/dag.py` | MODEL_CONFIGS: gemma9b → llama3b (si aplica) |
| `src/models.py` | Ya corregido (completion_tokens) |
| `data/promise_nfr.csv` | Reemplazar con PROMISE_exp (969 muestras) |
| `data/ambiguity_dataset.csv` | Reemplazar con ReqEval (~200 muestras) |

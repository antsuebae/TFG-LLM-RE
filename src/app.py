"""
Frontend Streamlit para el TFG de Ingenieria de Requisitos con LLMs.

Paginas:
1. Pipeline Documento - Analisis completo multi-modelo sobre un documento
2. Clasificar Requisito (F/NF)
3. Analizar Calidad (ambiguedad, completitud, testabilidad)
4. Validar Consistencia (pares de requisitos)
5. Resultados Experimentos (dashboard)
6. Comparar Modelos (tabla comparativa)
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os
import json
import time
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
os.chdir(Path(__file__).parent)

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from models import get_model
from prompts import build_prompt, parse_response, TASK_NAMES, STRATEGY_NAMES

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="RE-LLM: Analisis de Requisitos",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Model configs ────────────────────────────────────────────
MODEL_CONFIGS = {
    "qwen7b": {"name": "qwen2.5-coder:7b-instruct-q5_K_M", "type": "ollama", "short_name": "qwen7b"},
    "llama8b": {"name": "llama3.1:8b-instruct-q4_K_M", "type": "ollama", "short_name": "llama8b"},
    "nim_llama70b": {"name": "meta/llama-3.1-70b-instruct", "type": "nvidia_nim", "short_name": "nim_llama70b"},
    "nim_llama8b": {"name": "meta/llama-3.1-8b-instruct", "type": "nvidia_nim", "short_name": "nim_llama8b"},
    "nim_mistral": {"name": "mistralai/mistral-7b-instruct-v0.3", "type": "nvidia_nim", "short_name": "nim_mistral"},
}

MODEL_LABELS = {
    "qwen7b": "Qwen 7B (local)",
    "llama8b": "Llama 8B (local)",
    "nim_llama70b": "Llama 70B (NIM)",
    "nim_llama8b": "Llama 8B (NIM)",
    "nim_mistral": "Mistral 7B (NIM)",
}

STRATEGY_LABELS = {
    "question_refinement": "Question Refinement (QR)",
    "cognitive_verifier": "Cognitive Verifier (CV)",
    "persona_context": "Persona + Context (PC)",
    "few_shot": "Few-Shot",
    "chain_of_thought": "Chain of Thought (CoT)",
}

RESULTS_DIR = Path(__file__).parent.parent / "results"


def get_model_instance(model_key: str, temperature: float = 0.4):
    """Crea instancia de modelo."""
    return get_model(MODEL_CONFIGS[model_key], temperature)


# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("RE-LLM")
st.sidebar.markdown("Analisis de Requisitos con LLMs")

page = st.sidebar.radio(
    "Navegacion",
    ["Pipeline Documento", "Clasificar Requisito", "Analizar Calidad",
     "Validar Consistencia", "Resultados Experimentos", "Comparar Modelos"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Modelos locales:** Ollama")
st.sidebar.markdown("**Modelos API:** NVIDIA NIM")
nim_configured = bool(os.getenv("NVIDIA_API_KEY"))
st.sidebar.markdown(f"NVIDIA NIM: {'Configurado' if nim_configured else 'No configurado'}")


# ============================================================
# HELPER: Run pipeline for one model
# ============================================================
def run_pipeline_for_model(requirements: list[str], model_key: str,
                           strategy: str, skip_inconsistency: bool) -> dict:
    """Ejecuta el pipeline completo para un modelo. Returns dict with results."""
    from pipeline import analyze_requirement, analyze_inconsistencies, compute_quality_score

    model = get_model(MODEL_CONFIGS[model_key])
    start_time = time.time()

    rows = []
    for req in requirements:
        row = analyze_requirement(model, req, strategy)
        row['quality_score'] = compute_quality_score(row)
        rows.append(row)

    inconsistencies = []
    if not skip_inconsistency and len(requirements) > 1:
        inconsistencies = analyze_inconsistencies(
            model, requirements, strategy, max_pairs=30
        )

    elapsed = time.time() - start_time
    results_df = pd.DataFrame(rows)

    return {
        'model': model_key,
        'strategy': strategy,
        'results_df': results_df,
        'inconsistencies': inconsistencies,
        'elapsed': elapsed,
    }


# ============================================================
# PAGE 1: Pipeline Documento (MULTI-MODELO)
# ============================================================
if page == "Pipeline Documento":
    st.title("Pipeline Completo de Analisis de Requisitos")
    st.markdown("Sube un documento y ejecuta el analisis con uno o varios modelos.")

    # ── Upload / seleccionar archivo ─────────────────────────
    input_mode = st.radio("Origen del documento", ["Subir archivo", "Archivo existente"],
                          horizontal=True)

    requirements = []

    if input_mode == "Subir archivo":
        uploaded = st.file_uploader("Documento de requisitos",
                                     type=['txt', 'md', 'csv', 'pdf'])
        if uploaded:
            # Save to temp file for pipeline to load
            suffix = Path(uploaded.name).suffix
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded.read())
            tmp.close()

            from pipeline import load_requirements
            try:
                requirements = load_requirements(tmp.name)
                st.session_state['pipeline_doc_name'] = uploaded.name
                st.success(f"**{len(requirements)}** requisitos cargados desde {uploaded.name}")
            except Exception as e:
                st.error(f"Error al cargar: {e}")
            finally:
                os.unlink(tmp.name)
    else:
        data_dir = Path(__file__).parent.parent / "data"
        files = sorted(data_dir.glob("*"))
        doc_files = [f for f in files if f.suffix in ('.txt', '.md', '.csv', '.pdf')]
        if doc_files:
            selected = st.selectbox("Archivo", doc_files,
                                     format_func=lambda x: x.name)
            from pipeline import load_requirements
            try:
                requirements = load_requirements(str(selected))
                st.session_state['pipeline_doc_name'] = selected.name
                st.success(f"**{len(requirements)}** requisitos cargados desde {selected.name}")
            except Exception as e:
                st.error(f"Error al cargar: {e}")

    if requirements:
        # Preview
        with st.expander(f"Vista previa ({len(requirements)} requisitos)"):
            for i, r in enumerate(requirements, 1):
                st.markdown(f"**{i}.** {r[:200]}")

        st.markdown("---")

        # ── Configuracion ────────────────────────────────────
        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)

        with col_cfg1:
            selected_models = st.multiselect(
                "Modelos a usar",
                list(MODEL_CONFIGS.keys()),
                default=["qwen7b"],
                format_func=lambda x: MODEL_LABELS.get(x, x)
            )

        with col_cfg2:
            use_all_strategies = st.checkbox("Usar todas las estrategias", value=False)
            if use_all_strategies:
                selected_strategies = STRATEGY_NAMES
                st.info(f"{len(selected_strategies)} estrategias seleccionadas")
            else:
                selected_strategies = st.multiselect(
                    "Estrategias de prompt",
                    STRATEGY_NAMES,
                    default=["few_shot"],
                    format_func=lambda x: STRATEGY_LABELS.get(x, x),
                    key="pipe_strategies"
                )

        with col_cfg3:
            execution_mode = st.radio("Ejecucion", ["Secuencial", "Paralela"],
                                       horizontal=True)
            skip_inconsistency = st.checkbox("Saltar inconsistencias (mas rapido)", value=False)

        if not selected_models:
            st.warning("Selecciona al menos un modelo.")
        elif not selected_strategies:
            st.warning("Selecciona al menos una estrategia.")

        # ── Ejecutar ─────────────────────────────────────────
        elif st.button("Ejecutar Pipeline", type="primary"):

            # all_results: dict[(model_key, strategy)] -> result
            all_results = {}
            combos = [(m, s) for m in selected_models for s in selected_strategies]
            total_combos = len(combos)
            progress = st.progress(0, text="Iniciando...")
            status_container = st.container()

            if execution_mode == "Paralela" and total_combos > 1:
                # ── PARALELO ─────────────────────────────────
                status_container.info(
                    f"Ejecutando {total_combos} combinaciones (modelo x estrategia) en paralelo..."
                )
                with ThreadPoolExecutor(max_workers=min(total_combos, 4)) as executor:
                    futures = {}
                    for model_key, strat in combos:
                        f = executor.submit(
                            run_pipeline_for_model,
                            requirements, model_key, strat, skip_inconsistency
                        )
                        futures[f] = (model_key, strat)

                    done_count = 0
                    for future in as_completed(futures):
                        combo_key = futures[future]
                        done_count += 1
                        progress.progress(
                            done_count / total_combos,
                            text=f"Completado: {combo_key[0]} + {STRATEGY_LABELS.get(combo_key[1], combo_key[1])} ({done_count}/{total_combos})"
                        )
                        try:
                            all_results[combo_key] = future.result()
                        except Exception as e:
                            st.error(f"Error con {combo_key[0]} + {combo_key[1]}: {e}")
            else:
                # ── SECUENCIAL ───────────────────────────────
                for i, (model_key, strat) in enumerate(combos):
                    progress.progress(
                        i / total_combos,
                        text=f"Ejecutando {MODEL_LABELS.get(model_key, model_key)} + {STRATEGY_LABELS.get(strat, strat)}... ({i+1}/{total_combos})"
                    )
                    try:
                        all_results[(model_key, strat)] = run_pipeline_for_model(
                            requirements, model_key, strat, skip_inconsistency
                        )
                    except Exception as e:
                        st.error(f"Error con {model_key} + {strat}: {e}")

            progress.progress(1.0, text="Completado")

            if not all_results:
                st.error("No se obtuvieron resultados.")
            else:
                # Guardar en session_state para que persista
                st.session_state['pipeline_results'] = all_results
                st.session_state['pipeline_reqs'] = requirements
                st.session_state['pipeline_strategies'] = selected_strategies

        # ── Mostrar resultados ───────────────────────────────
        if 'pipeline_results' in st.session_state:
            all_results = st.session_state['pipeline_results']
            requirements = st.session_state['pipeline_reqs']

            st.markdown("---")
            st.header("Resultados")

            # Extraer modelos y estrategias unicos de los resultados
            result_models = sorted(set(k[0] for k in all_results.keys()))
            result_strategies = sorted(set(k[1] for k in all_results.keys()))

            # ── Tabla resumen comparativa ─────────────────────
            if len(all_results) > 1:
                st.subheader("Comparacion Modelo x Estrategia")

                summary_rows = []
                for (model_key, strat), res in all_results.items():
                    df = res['results_df']
                    n_amb = len(df[df['is_ambiguous'] == True])
                    n_inc = len(df[df['is_complete'] == False])
                    n_nt = len(df[df['is_testable'] == False])
                    n_f = len(df[df['classification'] == 'F'])
                    summary_rows.append({
                        'Modelo': MODEL_LABELS.get(model_key, model_key),
                        'Estrategia': STRATEGY_LABELS.get(strat, strat),
                        'Funcionales': n_f,
                        'No Funcionales': len(df) - n_f,
                        'Ambiguos': n_amb,
                        'Incompletos': n_inc,
                        'No Testables': n_nt,
                        'Calidad Media': f"{df['quality_score'].mean():.0f}%",
                        'Inconsistencias': len(res['inconsistencies']),
                        'Tiempo (s)': f"{res['elapsed']:.1f}",
                    })
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

                # Heatmap de calidad: modelo x estrategia
                if len(result_models) > 1 or len(result_strategies) > 1:
                    st.subheader("Calidad Media: Modelo x Estrategia")
                    heatmap_data = {}
                    for (model_key, strat), res in all_results.items():
                        model_label = MODEL_LABELS.get(model_key, model_key)
                        strat_label = STRATEGY_LABELS.get(strat, strat)
                        if model_label not in heatmap_data:
                            heatmap_data[model_label] = {}
                        heatmap_data[model_label][strat_label] = round(res['results_df']['quality_score'].mean(), 1)
                    heatmap_df = pd.DataFrame(heatmap_data).T
                    heatmap_df.index.name = "Modelo"
                    st.dataframe(heatmap_df.style.background_gradient(cmap='RdYlGn', vmin=0, vmax=100),
                                 use_container_width=True)

            # ── Tabs por modelo, sub-tabs por estrategia ─────
            model_tabs = st.tabs([MODEL_LABELS.get(m, m) for m in result_models])

            for model_tab, model_key in zip(model_tabs, result_models):
                with model_tab:
                    # Estrategias para este modelo
                    model_strategies = [s for s in result_strategies if (model_key, s) in all_results]

                    if len(model_strategies) > 1:
                        strat_tabs = st.tabs([STRATEGY_LABELS.get(s, s) for s in model_strategies])
                    else:
                        strat_tabs = [model_tab]  # Single tab = the model tab itself

                    for strat_tab, strat in zip(strat_tabs, model_strategies):
                        with strat_tab:
                            res = all_results[(model_key, strat)]
                            df = res['results_df']
                            inconsistencies = res['inconsistencies']

                            if len(model_strategies) <= 1:
                                st.caption(f"Estrategia: {STRATEGY_LABELS.get(strat, strat)}")

                            # Metricas resumen
                            total = len(df)
                            col1, col2, col3, col4, col5 = st.columns(5)
                            col1.metric("Requisitos", total)
                            col2.metric("Ambiguos", len(df[df['is_ambiguous'] == True]))
                            col3.metric("Incompletos", len(df[df['is_complete'] == False]))
                            col4.metric("No Testables", len(df[df['is_testable'] == False]))
                            col5.metric("Calidad Media", f"{df['quality_score'].mean():.0f}%")

                            # Tabla detallada
                            st.subheader("Detalle por Requisito")
                            display_df = df[['text', 'classification', 'is_ambiguous',
                                             'is_complete', 'is_testable', 'quality_score']].copy()
                            display_df.columns = ['Requisito', 'Tipo', 'Ambiguo',
                                                  'Completo', 'Testable', 'Calidad']
                            display_df['Requisito'] = display_df['Requisito'].str[:100]
                            display_df.index = range(1, len(display_df) + 1)
                            st.dataframe(display_df, use_container_width=True, height=400)

                            # Requisitos con problemas
                            problems = df[
                                (df['is_ambiguous'] == True) |
                                (df['is_complete'] == False) |
                                (df['is_testable'] == False)
                            ]
                            if not problems.empty:
                                with st.expander(f"Requisitos con problemas ({len(problems)})"):
                                    for _, row in problems.iterrows():
                                        issues = []
                                        if row.get('is_ambiguous'):
                                            issues.append(f"Ambiguo ({row.get('ambiguity_type', '')})")
                                            if row.get('ambiguous_words'):
                                                issues.append(f"Palabras: {row['ambiguous_words']}")
                                        if not row.get('is_complete'):
                                            issues.append(f"Incompleto - Falta: {row.get('missing_elements', '')}")
                                        if not row.get('is_testable'):
                                            issues.append(f"No testable ({row.get('testability_reason', '')})")
                                        st.markdown(f"**{row['text'][:120]}**")
                                        for issue in issues:
                                            st.markdown(f"  - {issue}")
                                        st.markdown("---")

                            # Inconsistencias
                            if inconsistencies:
                                with st.expander(f"Inconsistencias detectadas ({len(inconsistencies)})"):
                                    for inc in inconsistencies:
                                        st.warning(
                                            f"**Req #{inc['req_a_idx']}** vs **Req #{inc['req_b_idx']}**: "
                                            f"{inc['description']}"
                                        )

                            # Exportar
                            csv_data = df.to_csv(index=False)
                            st.download_button(
                                f"Descargar CSV",
                                csv_data,
                                f"pipeline_{model_key}_{strat}_{datetime.now():%Y%m%d_%H%M}.csv",
                                "text/csv",
                                key=f"download_{model_key}_{strat}"
                            )

            # ── Acciones finales ──────────────────────────────
            if len(all_results) >= 1:
                st.markdown("---")
                col_save, col_correct = st.columns(2)

                with col_save:
                    if st.button("Guardar Informes", type="primary"):
                        from pipeline import save_pipeline_results
                        output_dir = RESULTS_DIR / "pipeline"
                        output_dir.mkdir(parents=True, exist_ok=True)

                        saved_dirs = []
                        for (model_key, strat), res in all_results.items():
                            doc_name = st.session_state.get('pipeline_doc_name', 'documento')
                            run_dir = save_pipeline_results(
                                res['results_df'], res['inconsistencies'],
                                model_key, strat, doc_name, output_dir
                            )
                            saved_dirs.append(run_dir)
                        st.success(f"{len(saved_dirs)} ejecuciones guardadas en: {output_dir}")

                with col_correct:
                    # Select which model/strategy to use for rewriting
                    combo_labels = {
                        k: f"{MODEL_LABELS.get(k[0], k[0])} + {STRATEGY_LABELS.get(k[1], k[1])}"
                        for k in all_results.keys()
                    }
                    rewrite_combo = st.selectbox(
                        "Modelo para correccion",
                        list(all_results.keys()),
                        format_func=lambda x: combo_labels[x],
                        key="rewrite_combo"
                    )
                    if st.button("Generar Documento Corregido"):
                        from pipeline import generate_corrected_document, rewrite_requirement
                        res = all_results[rewrite_combo]
                        model_key, strat = rewrite_combo
                        model = get_model_instance(model_key)

                        with st.spinner("Reescribiendo requisitos con problemas..."):
                            corrections = []
                            progress_rewrite = st.progress(0)
                            df = res['results_df']
                            for i, (_, row) in enumerate(df.iterrows()):
                                has_problems = (
                                    row.get('is_ambiguous') is True or
                                    row.get('is_complete') is False or
                                    row.get('is_testable') is False
                                )
                                if has_problems:
                                    result = rewrite_requirement(model, row['text'], row.to_dict())
                                else:
                                    result = {'original': row['text'], 'corrected': row['text'], 'changes_made': []}
                                corrections.append(result)
                                progress_rewrite.progress((i + 1) / len(df))

                        st.session_state['corrections'] = corrections

                # ── Mostrar documento corregido ──────────────
                if 'corrections' in st.session_state:
                    corrections = st.session_state['corrections']
                    n_corrected = sum(1 for c in corrections if c['changes_made'])
                    st.subheader(f"Documento Corregido ({n_corrected} requisitos modificados)")

                    for i, c in enumerate(corrections, 1):
                        if c['changes_made']:
                            with st.expander(f"Requisito {i} - Corregido", expanded=False):
                                st.markdown(f"**Original:** ~~{c['original'][:200]}~~")
                                st.markdown(f"**Corregido:** :green[{c['corrected'][:200]}]")
                                st.markdown("**Cambios:**")
                                for change in c['changes_made']:
                                    st.markdown(f"- {change}")

                    # Generar fichero descargable
                    doc_lines = ["# Documento de Requisitos Corregido\n"]
                    doc_lines.append(f"{n_corrected} de {len(corrections)} requisitos corregidos.\n---\n")
                    for i, c in enumerate(corrections, 1):
                        if c['changes_made']:
                            doc_lines.append(f"### Requisito {i} (corregido)\n")
                            doc_lines.append(f"**Original:** ~~{c['original']}~~\n")
                            doc_lines.append(f"**Corregido:** {c['corrected']}\n")
                            doc_lines.append("**Cambios:**")
                            for change in c['changes_made']:
                                doc_lines.append(f"- {change}")
                            doc_lines.append("")
                        else:
                            doc_lines.append(f"### Requisito {i}\n")
                            doc_lines.append(f"{c['corrected']}\n")
                    doc_md = '\n'.join(doc_lines)

                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        st.download_button(
                            "Descargar documento corregido (MD)",
                            doc_md,
                            f"requisitos_corregidos_{datetime.now():%Y%m%d_%H%M}.md",
                            "text/markdown",
                            key="dl_corrected_md"
                        )
                    with col_dl2:
                        # Plain text: just the corrected requirements
                        txt_lines = [c['corrected'] for c in corrections]
                        st.download_button(
                            "Descargar requisitos corregidos (TXT)",
                            '\n'.join(txt_lines),
                            f"requisitos_corregidos_{datetime.now():%Y%m%d_%H%M}.txt",
                            "text/plain",
                            key="dl_corrected_txt"
                        )


# ============================================================
# PAGE 2: Clasificar Requisito
# ============================================================
elif page == "Clasificar Requisito":
    st.title("Clasificacion de Requisitos (F / NF)")
    st.markdown("Clasifica un requisito como **Funcional (F)** o **No Funcional (NF)**.")

    col1, col2 = st.columns([2, 1])

    with col1:
        req_text = st.text_area(
            "Requisito a clasificar",
            value="The system shall allow users to reset their password via email.",
            height=100
        )

    with col2:
        model_key = st.selectbox("Modelo", list(MODEL_CONFIGS.keys()),
                                  format_func=lambda x: MODEL_LABELS.get(x, x))
        strategy = st.selectbox("Estrategia", STRATEGY_NAMES,
                                format_func=lambda x: STRATEGY_LABELS.get(x, x))
        temperature = st.slider("Temperatura", 0.0, 1.0, 0.4, 0.1)

    if st.button("Clasificar", type="primary"):
        with st.spinner("Clasificando..."):
            try:
                model = get_model_instance(model_key, temperature)
                prompt = build_prompt("classification", strategy, requirement=req_text)
                response = model.generate(prompt, max_tokens=512)

                if response['success']:
                    prediction = parse_response("classification", response['content'])

                    col_r1, col_r2, col_r3 = st.columns(3)
                    with col_r1:
                        label = {"F": "Funcional", "NF": "No Funcional"}.get(prediction, "No determinado")
                        st.metric("Clasificacion", f"{prediction} - {label}")
                    with col_r2:
                        st.metric("Tiempo", f"{response['time_seconds']:.2f}s")
                    with col_r3:
                        st.metric("Tokens/s", f"{response.get('tokens_per_second', 0):.1f}")

                    with st.expander("Respuesta completa del modelo"):
                        st.text(response['content'])
                else:
                    st.error(f"Error: {response['error']}")
            except Exception as e:
                st.error(f"Error: {e}")


# ============================================================
# PAGE 3: Analizar Calidad
# ============================================================
elif page == "Analizar Calidad":
    st.title("Analisis de Calidad de Requisitos")

    analysis_type = st.radio(
        "Tipo de analisis",
        ["Ambiguedad", "Completitud", "Testabilidad"],
        horizontal=True
    )

    task_map = {"Ambiguedad": "ambiguity", "Completitud": "completeness", "Testabilidad": "testability"}
    task = task_map[analysis_type]

    col1, col2 = st.columns([2, 1])
    with col1:
        req_text = st.text_area("Requisito a analizar", height=100,
                                value="The system should be fast and user-friendly.")
    with col2:
        model_key = st.selectbox("Modelo", list(MODEL_CONFIGS.keys()), key="quality_model",
                                  format_func=lambda x: MODEL_LABELS.get(x, x))
        strategy = st.selectbox("Estrategia", STRATEGY_NAMES, key="quality_strategy",
                                format_func=lambda x: STRATEGY_LABELS.get(x, x))

    if st.button("Analizar", type="primary"):
        with st.spinner("Analizando..."):
            try:
                model = get_model_instance(model_key)
                prompt = build_prompt(task, strategy, requirement=req_text)
                response = model.generate(prompt, max_tokens=512)

                if response['success']:
                    parsed = parse_response(task, response['content'])

                    if task == "ambiguity":
                        status = "Ambiguo" if parsed['is_ambiguous'] else "No ambiguo"
                        color = "red" if parsed['is_ambiguous'] else "green"
                        st.markdown(f"### Resultado: :{color}[{status}]")
                        if parsed['is_ambiguous']:
                            st.markdown(f"**Tipo:** {parsed['ambiguity_type']}")
                            if parsed['ambiguous_words']:
                                st.markdown(f"**Palabras ambiguas:** {', '.join(parsed['ambiguous_words'])}")

                    elif task == "completeness":
                        status = "Completo" if parsed['is_complete'] else "Incompleto"
                        color = "green" if parsed['is_complete'] else "red"
                        st.markdown(f"### Resultado: :{color}[{status}]")
                        if parsed['missing_elements']:
                            st.markdown("**Elementos faltantes:**")
                            for elem in parsed['missing_elements']:
                                st.markdown(f"- {elem}")

                    elif task == "testability":
                        status = "Testable" if parsed['is_testable'] else "No testable"
                        color = "green" if parsed['is_testable'] else "red"
                        st.markdown(f"### Resultado: :{color}[{status}]")
                        st.markdown(f"**Razon:** {parsed['reason']}")

                    st.metric("Tiempo", f"{response['time_seconds']:.2f}s")

                    with st.expander("Respuesta completa"):
                        st.text(response['content'])
                else:
                    st.error(f"Error: {response['error']}")
            except Exception as e:
                st.error(f"Error: {e}")


# ============================================================
# PAGE 4: Validar Consistencia
# ============================================================
elif page == "Validar Consistencia":
    st.title("Validacion de Consistencia entre Requisitos")
    st.markdown("Detecta inconsistencias entre pares de requisitos.")

    col1, col2 = st.columns(2)
    with col1:
        req_a = st.text_area("Requisito A", height=100,
                             value="The system shall store all data locally on the user's device.")
    with col2:
        req_b = st.text_area("Requisito B", height=100,
                             value="All user data must be stored in a cloud-based database.")

    model_key = st.selectbox("Modelo", list(MODEL_CONFIGS.keys()), key="consist_model",
                              format_func=lambda x: MODEL_LABELS.get(x, x))
    strategy = st.selectbox("Estrategia", STRATEGY_NAMES, key="consist_strategy",
                            format_func=lambda x: STRATEGY_LABELS.get(x, x))

    if st.button("Validar Consistencia", type="primary"):
        with st.spinner("Validando..."):
            try:
                model = get_model_instance(model_key)
                prompt = build_prompt("inconsistency", strategy,
                                      requirement_a=req_a, requirement_b=req_b)
                response = model.generate(prompt, max_tokens=512)

                if response['success']:
                    parsed = parse_response("inconsistency", response['content'])

                    if parsed['is_inconsistent']:
                        st.markdown("### :red[Inconsistencia detectada]")
                        st.markdown(f"**Descripcion:** {parsed['description']}")
                    else:
                        st.markdown("### :green[Requisitos consistentes]")

                    st.metric("Tiempo", f"{response['time_seconds']:.2f}s")

                    with st.expander("Respuesta completa"):
                        st.text(response['content'])
                else:
                    st.error(f"Error: {response['error']}")
            except Exception as e:
                st.error(f"Error: {e}")


# ============================================================
# PAGE 5: Resultados de Experimentos
# ============================================================
elif page == "Resultados Experimentos":
    st.title("Dashboard de Resultados")

    view_mode = st.radio("Ver", ["Pipeline (analisis de documentos)", "Experimentos (benchmark)"],
                         horizontal=True)

    if view_mode == "Pipeline (analisis de documentos)":
        # ── Ejecuciones del pipeline con metadata ─────────
        from pipeline import load_pipeline_runs
        pipeline_dir = RESULTS_DIR / "pipeline"
        runs = load_pipeline_runs(pipeline_dir)

        if not runs:
            st.warning("No hay ejecuciones del pipeline guardadas.")
            st.info("Ejecuta el pipeline desde 'Pipeline Documento' y pulsa 'Guardar Informes'.")
        else:
            st.subheader(f"{len(runs)} ejecuciones encontradas")

            # Tabla resumen de ejecuciones
            runs_summary = []
            for r in runs:
                s = r.get('summary', {})
                runs_summary.append({
                    'Fecha': r.get('timestamp', '')[:16],
                    'Documento': r.get('document', ''),
                    'Modelo': MODEL_LABELS.get(r.get('model', ''), r.get('model', '')),
                    'Estrategia': STRATEGY_LABELS.get(r.get('strategy', ''), r.get('strategy', '')),
                    'Requisitos': r.get('n_requirements', 0),
                    'Ambiguos': s.get('ambiguous', 0),
                    'Incompletos': s.get('incomplete', 0),
                    'No Testables': s.get('not_testable', 0),
                    'Calidad Media': f"{s.get('avg_quality', 0):.0f}%",
                    'run_dir': r.get('run_dir', ''),
                })
            runs_df = pd.DataFrame(runs_summary)
            display_runs_df = runs_df.drop(columns=['run_dir'])
            st.dataframe(display_runs_df, use_container_width=True)

            # Seleccionar ejecuciones para comparar
            run_labels = [f"{r['Fecha']} | {r['Modelo']} | {r['Estrategia']} | {r['Documento']}" for r in runs_summary]
            selected_runs = st.multiselect("Seleccionar ejecuciones para comparar", run_labels)

            if selected_runs:
                selected_indices = [run_labels.index(label) for label in selected_runs]

                if len(selected_indices) >= 2:
                    st.subheader("Comparacion de ejecuciones seleccionadas")
                    compare_rows = [runs_summary[i] for i in selected_indices]
                    compare_df = pd.DataFrame(compare_rows).drop(columns=['run_dir'])
                    st.dataframe(compare_df, use_container_width=True)

                # Detalle de cada ejecucion seleccionada
                for idx in selected_indices:
                    run = runs_summary[idx]
                    run_path = Path(run['run_dir'])
                    csv_path = run_path / "results.csv"
                    if csv_path.exists():
                        with st.expander(f"Detalle: {run['Modelo']} + {run['Estrategia']} ({run['Fecha']})"):
                            run_df = pd.read_csv(csv_path)
                            display_df = run_df[['text', 'classification', 'is_ambiguous',
                                                 'is_complete', 'is_testable', 'quality_score']].copy()
                            display_df.columns = ['Requisito', 'Tipo', 'Ambiguo',
                                                  'Completo', 'Testable', 'Calidad']
                            display_df['Requisito'] = display_df['Requisito'].str[:100]
                            display_df.index = range(1, len(display_df) + 1)
                            st.dataframe(display_df, use_container_width=True)

                            # Download HTML report
                            html_path = run_path / "informe.html"
                            if html_path.exists():
                                st.download_button(
                                    "Descargar informe HTML",
                                    html_path.read_text(encoding='utf-8'),
                                    f"informe_{run_path.name}.html",
                                    "text/html",
                                    key=f"dl_html_{idx}"
                                )

    else:
        # ── Resultados de experimentos (benchmark) ────────
        result_files = sorted(RESULTS_DIR.glob("results_*.csv"))

        if not result_files:
            st.warning("No hay resultados de experimentos. Ejecuta el pipeline experimental primero.")
            st.code("python experiment.py --dry-run --task classification --models qwen7b", language="bash")
        else:
            selected_file = st.selectbox(
                "Archivo de resultados",
                result_files,
                format_func=lambda x: x.name
            )

            df = pd.read_csv(selected_file)
            strategy_col = 'strategy' if 'strategy' in df.columns else 'pattern'
            task = df['task'].iloc[0] if 'task' in df.columns else 'classification'

            st.markdown(f"**Tarea:** {task} | **Registros:** {len(df)} | "
                        f"**Modelos:** {df['model'].nunique() if 'model' in df.columns else 'N/A'}")

            if 'model' in df.columns and strategy_col in df.columns and 'iteration' in df.columns:
                from analysis import compute_metrics_per_config
                metrics_df = compute_metrics_per_config(df, task)

                st.subheader("Resumen de Metricas")
                summary = metrics_df.groupby(['model', 'strategy']).agg({
                    'f1': ['mean', 'std'],
                    'accuracy': ['mean', 'std'],
                    'precision': 'mean',
                    'recall': 'mean',
                }).round(3)
                st.dataframe(summary, use_container_width=True)

                st.subheader("Visualizaciones")
                chart_tabs = st.tabs(["Barras", "Boxplot", "Heatmap", "Radar", "Velocidad"])

                import matplotlib.pyplot as plt
                with chart_tabs[0]:
                    from analysis import plot_grouped_bars
                    fig = plot_grouped_bars(metrics_df, 'f1')
                    st.pyplot(fig)
                    plt.close(fig)
                with chart_tabs[1]:
                    from analysis import plot_boxplots
                    fig = plot_boxplots(metrics_df, 'f1')
                    st.pyplot(fig)
                    plt.close(fig)
                with chart_tabs[2]:
                    from analysis import plot_heatmap
                    fig = plot_heatmap(metrics_df, 'f1')
                    st.pyplot(fig)
                    plt.close(fig)
                with chart_tabs[3]:
                    from analysis import plot_radar
                    fig = plot_radar(metrics_df)
                    st.pyplot(fig)
                    plt.close(fig)
                with chart_tabs[4]:
                    from analysis import plot_speed_comparison
                    fig = plot_speed_comparison(metrics_df)
                    if fig:
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.info("No hay datos de velocidad disponibles.")
            else:
                st.subheader("Datos")
                st.dataframe(df, use_container_width=True)


# ============================================================
# PAGE 6: Comparar Modelos
# ============================================================
elif page == "Comparar Modelos":
    st.title("Comparacion de Modelos")

    result_files = sorted(RESULTS_DIR.glob("results_*.csv"))

    if not result_files:
        st.warning("No hay resultados disponibles. Ejecuta experimentos primero.")
    else:
        selected_file = st.selectbox(
            "Archivo de resultados",
            result_files,
            format_func=lambda x: x.name,
            key="compare_file"
        )

        df = pd.read_csv(selected_file)
        task = df['task'].iloc[0] if 'task' in df.columns else 'classification'

        from analysis import compute_metrics_per_config
        metrics_df = compute_metrics_per_config(df, task)

        st.subheader("Rendimiento por Modelo")
        model_summary = metrics_df.groupby('model').agg({
            'f1': ['mean', 'std'],
            'accuracy': ['mean', 'std'],
            'precision': 'mean',
            'recall': 'mean',
            'avg_time_seconds': 'mean',
            'avg_tokens_per_second': 'mean',
        }).round(3)
        model_summary.columns = ['F1 Mean', 'F1 Std', 'Acc Mean', 'Acc Std',
                                  'Prec Mean', 'Rec Mean', 'Avg Time (s)', 'Tokens/s']
        st.dataframe(model_summary, use_container_width=True)

        st.subheader("Mejor Estrategia por Modelo")
        best_strategy = metrics_df.groupby(['model', 'strategy'])['f1'].mean().reset_index()
        best_per_model = best_strategy.loc[best_strategy.groupby('model')['f1'].idxmax()]
        st.dataframe(best_per_model.rename(columns={'f1': 'F1 Mean'}).set_index('model'),
                     use_container_width=True)

        st.subheader("Local vs API")
        local_models = ['qwen7b', 'llama8b']
        api_models = ['nim_llama70b', 'nim_llama8b', 'nim_mistral']

        local_metrics = metrics_df[metrics_df['model'].isin(local_models)]
        api_metrics = metrics_df[metrics_df['model'].isin(api_models)]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Modelos Locales (Ollama)**")
            if not local_metrics.empty:
                st.metric("F1 Promedio", f"{local_metrics['f1'].mean():.3f}")
                st.metric("Accuracy Promedio", f"{local_metrics['accuracy'].mean():.3f}")
                st.metric("Tiempo Promedio", f"{local_metrics['avg_time_seconds'].mean():.2f}s")
            else:
                st.info("Sin datos")

        with col2:
            st.markdown("**Modelos API (NVIDIA NIM)**")
            if not api_metrics.empty:
                st.metric("F1 Promedio", f"{api_metrics['f1'].mean():.3f}")
                st.metric("Accuracy Promedio", f"{api_metrics['accuracy'].mean():.3f}")
                st.metric("Tiempo Promedio", f"{api_metrics['avg_time_seconds'].mean():.2f}s")
            else:
                st.info("Sin datos")

        st.subheader("Trade-offs: Rendimiento vs Privacidad vs Coste")
        st.markdown("""
        | Aspecto | Modelos Locales (Ollama) | Modelos API (NVIDIA NIM) |
        |---------|-------------------------|--------------------------|
        | **Privacidad** | Total (datos no salen del equipo) | Parcial (datos enviados a API) |
        | **Coste** | Solo hardware (GPU) | Gratuito con limites / Pay-per-use |
        | **Latencia** | Dependiente de GPU local | Dependiente de red |
        | **Escalabilidad** | Limitada por hardware | Alta |
        | **Disponibilidad** | Siempre (offline) | Requiere conexion |
        """)

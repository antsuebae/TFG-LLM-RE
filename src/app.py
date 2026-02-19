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
from warnings_analysis import generate_all_warnings, check_input_warnings

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="RE-LLM: Analisis de Requisitos",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    section[data-testid="stSidebar"] {
        border-right: 1px solid #D5DBDB;
    }
    section[data-testid="stSidebar"] .stMarkdown h1 {
        color: #1B4F72;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #D5DBDB;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    div[data-testid="stMetric"] label {
        color: #566573;
        font-size: 0.85rem;
    }

    /* Tables */
    .stDataFrame {
        border-radius: 6px;
        overflow: hidden;
    }

    /* Primary buttons */
    .stButton > button[kind="primary"] {
        background-color: #1B4F72;
        border: none;
        border-radius: 6px;
        font-weight: 600;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #154360;
    }

    /* Status containers */
    details[data-testid="stExpander"] {
        border-radius: 6px;
        border: 1px solid #D5DBDB;
    }

    /* Download buttons */
    .stDownloadButton > button {
        border-radius: 6px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ── Model configs ────────────────────────────────────────────
MODEL_CONFIGS = {
    "qwen7b": {"name": "qwen2.5:7b-instruct-q5_K_M", "type": "ollama", "short_name": "qwen7b"},
    "llama8b": {"name": "llama3.1:8b-instruct-q4_K_M", "type": "ollama", "short_name": "llama8b"},
    "gemma9b": {"name": "gemma2:9b-instruct-q4_K_M", "type": "ollama", "short_name": "gemma9b"},
    "nim_llama70b": {"name": "meta/llama-3.1-70b-instruct", "type": "nvidia_nim", "short_name": "nim_llama70b"},
    "nim_llama8b": {"name": "meta/llama-3.1-8b-instruct", "type": "nvidia_nim", "short_name": "nim_llama8b"},
    "nim_mistral": {"name": "mistralai/mistral-7b-instruct-v0.3", "type": "nvidia_nim", "short_name": "nim_mistral"},
}

MODEL_LABELS = {
    "qwen7b": "Qwen 2.5 7B (local)",
    "llama8b": "Llama 8B (local)",
    "gemma9b": "Gemma 2 9B (local)",
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
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"


def _render_warnings(warnings: list[dict], title: str = "Advertencias de calidad"):
    """Muestra advertencias en un expander de Streamlit."""
    if not warnings:
        return
    n = len(warnings)
    with st.expander(f"{title} ({n})", expanded=False):
        for w in warnings:
            level = w.get('level', 'warning')
            msg = w.get('message', '')
            details = w.get('details', [])
            detail_text = "\n".join(f"  {d}" for d in details) if details else ""
            full_msg = f"{msg}\n{detail_text}" if detail_text else msg
            if level == 'error':
                st.error(full_msg)
            elif level == 'warning':
                st.warning(full_msg)
            else:
                st.info(full_msg)


def get_model_instance(model_key: str, temperature: float = 0.4):
    """Crea instancia de modelo."""
    return get_model(MODEL_CONFIGS[model_key], temperature)


# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("RE-LLM")
st.sidebar.markdown("Analisis de Requisitos con LLMs")

_PAGES = ["Pipeline Documento", "Clasificar Requisito", "Analizar Calidad",
           "Validar Consistencia", "Resultados Experimentos", "Comparar Modelos",
           "Progreso Experimentos"]

page = st.sidebar.radio(
    "Navegacion",
    _PAGES,
    key="nav_page",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Modelos locales:** Ollama")
st.sidebar.markdown("**Modelos API:** NVIDIA NIM")
nim_configured = bool(os.getenv("NVIDIA_API_KEY"))
st.sidebar.markdown(f"NVIDIA NIM: {'Configurado' if nim_configured else 'No configurado'}")


# ============================================================
# HELPER: Run pipeline for one model (uses DAG)
# ============================================================
def run_pipeline_for_model(requirements: list[str], model_key: str,
                           strategy: str, skip_inconsistency: bool,
                           filepath: str = None,
                           use_llm_extraction: bool = False,
                           doc_name: str = 'documento',
                           progress_callback=None) -> dict:
    """Ejecuta el pipeline DAG para un modelo. Returns dict with results."""
    from dag import run_dag_pipeline, run_dag_pipeline_from_requirements

    start_time = time.time()

    if filepath and use_llm_extraction:
        # Full DAG: load + LLM extraction + analysis
        ctx = run_dag_pipeline(
            filepath=filepath,
            model_key=model_key,
            strategy=strategy,
            use_llm_extraction=True,
            skip_inconsistency=skip_inconsistency,
            max_pairs=30,
            progress_callback=progress_callback,
        )
    else:
        # DAG from requirements (skip Nodes 1-2)
        ctx = run_dag_pipeline_from_requirements(
            requirements=requirements,
            model_key=model_key,
            strategy=strategy,
            skip_inconsistency=skip_inconsistency,
            max_pairs=30,
            doc_name=doc_name,
            progress_callback=progress_callback,
        )

    elapsed = time.time() - start_time
    results_df = ctx.get('results_df', pd.DataFrame())
    inconsistencies = ctx.get('inconsistencies', [])

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

    # Track filepath for LLM extraction
    doc_filepath = None
    doc_suffix = ''

    if input_mode == "Subir archivo":
        uploaded = st.file_uploader("Documento de requisitos",
                                     type=['txt', 'md', 'csv', 'pdf'])
        if uploaded:
            # Save to temp file (keep it for potential LLM extraction)
            suffix = Path(uploaded.name).suffix
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded.read())
            tmp.close()
            doc_filepath = tmp.name
            doc_suffix = suffix.lower()

            from pipeline import load_requirements
            try:
                requirements = load_requirements(tmp.name)
                st.session_state['pipeline_doc_name'] = uploaded.name
                st.session_state['pipeline_doc_filepath'] = tmp.name
                st.session_state['pipeline_doc_suffix'] = doc_suffix
                st.success(f"**{len(requirements)}** requisitos cargados desde {uploaded.name}")
            except Exception as e:
                st.error(f"Error al cargar: {e}")
    else:
        data_dir = Path(__file__).parent.parent / "data"
        files = sorted(data_dir.glob("*"))
        doc_files = [f for f in files if f.suffix in ('.txt', '.md', '.csv', '.pdf')]
        if doc_files:
            selected = st.selectbox("Archivo", doc_files,
                                     format_func=lambda x: x.name)
            doc_filepath = str(selected)
            doc_suffix = selected.suffix.lower()

            from pipeline import load_requirements
            try:
                requirements = load_requirements(str(selected))
                st.session_state['pipeline_doc_name'] = selected.name
                st.session_state['pipeline_doc_filepath'] = str(selected)
                st.session_state['pipeline_doc_suffix'] = doc_suffix
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
                default=["llama8b"],
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
            skip_inconsistency = st.checkbox("Saltar inconsistencias (mas rapido)", value=False)
            # LLM extraction: default True for PDF/TXT/MD, False for CSV
            current_suffix = st.session_state.get('pipeline_doc_suffix', '')
            default_llm_extract = current_suffix in ('.pdf', '.txt', '.md')
            use_llm_extraction = st.checkbox(
                "Usar LLM para extraer requisitos",
                value=default_llm_extract,
                help="Usa el LLM para limpiar y extraer requisitos del texto bruto. Recomendado para PDF/TXT."
            )

        if not selected_models:
            st.warning("Selecciona al menos un modelo.")
        elif not selected_strategies:
            st.warning("Selecciona al menos una estrategia.")

        # ── Ejecutar ─────────────────────────────────────────
        elif st.button("Ejecutar Pipeline", type="primary"):

            from dag import NODE_LABELS as DAG_NODE_LABELS

            # all_results: dict[(model_key, strategy)] -> result
            all_results = {}
            combos = [(m, s) for m in selected_models for s in selected_strategies]
            total_combos = len(combos)
            progress = st.progress(0, text="Iniciando...")

            current_filepath = st.session_state.get('pipeline_doc_filepath')
            current_doc_name = st.session_state.get('pipeline_doc_name', 'documento')

            # ── Extraccion unica de requisitos ────────────────
            extracted_requirements = None
            if use_llm_extraction and current_filepath:
                first_model_key = combos[0][0]
                first_strat = combos[0][1]
                combo_label = f"{MODEL_LABELS.get(first_model_key, first_model_key)} + {STRATEGY_LABELS.get(first_strat, first_strat)}"
                progress.progress(0, text=f"Combo 1/{total_combos}: {combo_label}")

                with st.status(f"Combo 1/{total_combos}: {combo_label}", expanded=True) as status_container:
                    def _extraction_cb(node_name, current, total, skipped=False):
                        label = DAG_NODE_LABELS.get(node_name, node_name)
                        if skipped:
                            status_container.write(f"~~{label}~~ (omitido)")
                        else:
                            status_container.write(f"**Paso {current}/{total}:** {label}...")

                    try:
                        from dag import run_dag_pipeline
                        ctx = run_dag_pipeline(
                            filepath=current_filepath,
                            model_key=first_model_key,
                            strategy=first_strat,
                            use_llm_extraction=True,
                            skip_inconsistency=skip_inconsistency,
                            max_pairs=30,
                            progress_callback=_extraction_cb,
                        )
                        extracted_requirements = ctx.get('requirements', [])
                        results_df = ctx.get('results_df', pd.DataFrame())
                        inconsistencies = ctx.get('inconsistencies', [])
                        all_results[(first_model_key, first_strat)] = {
                            'model': first_model_key,
                            'strategy': first_strat,
                            'results_df': results_df,
                            'inconsistencies': inconsistencies,
                            'elapsed': 0,
                        }
                        status_container.update(label=f"Combo 1/{total_combos}: {combo_label} - Completada", state="complete", expanded=False)
                        st.info(f"Requisitos extraidos por LLM: {len(extracted_requirements)} (se reutilizaran)")
                    except Exception as e:
                        status_container.update(label=f"Combo 1/{total_combos}: {combo_label} - Error", state="error", expanded=False)
                        st.error(f"Error en extraccion LLM: {e}")
                        extracted_requirements = requirements

            for i, (model_key, strat) in enumerate(combos):
                if use_llm_extraction and extracted_requirements and i == 0:
                    progress.progress(1 / total_combos, text=f"Combo 1/{total_combos} completada")
                    continue

                combo_label = f"{MODEL_LABELS.get(model_key, model_key)} + {STRATEGY_LABELS.get(strat, strat)}"
                progress.progress(
                    i / total_combos,
                    text=f"Combo {i+1}/{total_combos}: {combo_label}"
                )

                with st.status(f"Combo {i+1}/{total_combos}: {combo_label}", expanded=True) as status_container:
                    def _make_dag_cb(sc):
                        def _cb(node_name, current, total, skipped=False):
                            label = DAG_NODE_LABELS.get(node_name, node_name)
                            if skipped:
                                sc.write(f"~~{label}~~ (omitido)")
                            else:
                                sc.write(f"**Paso {current}/{total}:** {label}...")
                        return _cb

                    try:
                        reqs_to_use = extracted_requirements if extracted_requirements else requirements
                        all_results[(model_key, strat)] = run_pipeline_for_model(
                            reqs_to_use, model_key, strat, skip_inconsistency,
                            filepath=None,
                            use_llm_extraction=False,
                            doc_name=current_doc_name,
                            progress_callback=_make_dag_cb(status_container),
                        )
                        status_container.update(label=f"Combo {i+1}/{total_combos}: {combo_label} - Completada", state="complete", expanded=False)
                    except Exception as e:
                        status_container.update(label=f"Combo {i+1}/{total_combos}: {combo_label} - Error", state="error", expanded=False)
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

            # Filter out results with empty DataFrames
            all_results = {k: v for k, v in all_results.items()
                          if not v['results_df'].empty}
            if not all_results:
                st.error("Todas las ejecuciones fallaron. Revisa los logs del modelo.")
                st.stop()

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
                    n_amb = len(df[df['is_ambiguous'] == True]) if 'is_ambiguous' in df.columns else 0
                    n_inc = len(df[df['is_complete'] == False]) if 'is_complete' in df.columns else 0
                    n_nt = len(df[df['is_testable'] == False]) if 'is_testable' in df.columns else 0
                    n_f = len(df[df['classification'] == 'F']) if 'classification' in df.columns else 0
                    avg_q = df['quality_score'].mean() if 'quality_score' in df.columns else 0
                    summary_rows.append({
                        'Modelo': MODEL_LABELS.get(model_key, model_key),
                        'Estrategia': STRATEGY_LABELS.get(strat, strat),
                        'Funcionales': n_f,
                        'No Funcionales': len(df) - n_f,
                        'Ambiguos': n_amb,
                        'Incompletos': n_inc,
                        'No Testables': n_nt,
                        'Calidad Media': f"{avg_q:.0f}%",
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
                        q = res['results_df']['quality_score'].mean() if 'quality_score' in res['results_df'].columns else 0
                        heatmap_data[model_label][strat_label] = round(q, 1)
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
                            col2.metric("Ambiguos", len(df[df['is_ambiguous'] == True]) if 'is_ambiguous' in df.columns else 0)
                            col3.metric("Incompletos", len(df[df['is_complete'] == False]) if 'is_complete' in df.columns else 0)
                            col4.metric("No Testables", len(df[df['is_testable'] == False]) if 'is_testable' in df.columns else 0)
                            col5.metric("Calidad Media", f"{df['quality_score'].mean():.0f}%" if 'quality_score' in df.columns else "N/A")

                            # Tabla detallada
                            st.subheader("Detalle por Requisito")
                            display_df = df[['text', 'classification', 'is_ambiguous',
                                             'is_complete', 'is_testable', 'quality_score']].copy()
                            display_df.columns = ['Requisito', 'Tipo', 'Ambiguo',
                                                  'Completo', 'Testable', 'Calidad']
                            display_df['Requisito'] = display_df['Requisito'].str[:100]
                            display_df.index = range(1, len(display_df) + 1)
                            st.dataframe(display_df, use_container_width=True, height=400)

                            # Advertencias de calidad automaticas
                            pipeline_warnings = generate_all_warnings(df)
                            _render_warnings(pipeline_warnings, "Advertencias de calidad")

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
                                                issues.append(f"Palabras: {row.get('ambiguous_words', '')}")
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

                        df = res['results_df']
                        n_problems = sum(1 for _, row in df.iterrows() if (
                            row.get('is_ambiguous') is True or
                            row.get('is_complete') is False or
                            row.get('is_testable') is False
                        ))

                        with st.status(f"Reescribiendo {n_problems} requisitos con problemas...", expanded=True) as rewrite_status:
                            corrections = []
                            progress_rewrite = st.progress(0)
                            for i, (_, row) in enumerate(df.iterrows()):
                                has_problems = (
                                    row.get('is_ambiguous') is True or
                                    row.get('is_complete') is False or
                                    row.get('is_testable') is False
                                )
                                if has_problems:
                                    short_text = row['text'][:80] + ('...' if len(row['text']) > 80 else '')
                                    rewrite_status.write(f"**Requisito {i+1}/{len(df)}:** {short_text}")
                                    result = rewrite_requirement(model, row['text'], row.to_dict())
                                else:
                                    result = {'original': row['text'], 'corrected': row['text'], 'changes_made': []}
                                corrections.append(result)
                                progress_rewrite.progress((i + 1) / len(df))
                            rewrite_status.update(label=f"Reescritura completada ({n_problems} requisitos corregidos)", state="complete", expanded=False)

                        st.session_state['corrections'] = corrections

                # ── Mostrar documento corregido ──────────────
                if 'corrections' in st.session_state:
                    corrections = st.session_state['corrections']
                    n_corrected = sum(1 for c in corrections if c['changes_made'])
                    st.subheader(f"Documento Corregido ({n_corrected} requisitos modificados)")

                    # Advertencias de reescritura
                    rewrite_key = st.session_state.get('rewrite_combo', list(all_results.keys())[0])
                    rewrite_df = all_results[rewrite_key]['results_df']
                    rewrite_warnings = generate_all_warnings(rewrite_df, corrections)
                    _render_warnings(rewrite_warnings, "Advertencias de reescritura")

                    for i, c in enumerate(corrections, 1):
                        if c['changes_made']:
                            with st.expander(f"Requisito {i} - Corregido", expanded=False):
                                st.markdown(f"**Original:** ~~{c['original'][:200]}~~")
                                st.markdown(f"**Corregido:** :green[{c['corrected'][:200]}]")
                                st.markdown("**Cambios:**")
                                for change in c['changes_made']:
                                    st.markdown(f"- {change}")

                    # Generar documento SRS profesional
                    from pipeline import generate_srs_document, generate_srs_html
                    rewrite_key = st.session_state.get('rewrite_combo', list(all_results.keys())[0])
                    res = all_results[rewrite_key]
                    srs_metadata = {
                        'doc_name': st.session_state.get('pipeline_doc_name', 'documento'),
                        'model': MODEL_LABELS.get(rewrite_key[0], rewrite_key[0]),
                        'strategy': STRATEGY_LABELS.get(rewrite_key[1], rewrite_key[1]),
                        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    }
                    srs_md = generate_srs_document(
                        res['results_df'], corrections,
                        res['inconsistencies'], srs_metadata
                    )
                    srs_html = generate_srs_html(srs_md, srs_metadata)

                    col_dl1, col_dl2, col_dl3 = st.columns(3)
                    with col_dl1:
                        st.download_button(
                            "Descargar SRS (Markdown)",
                            srs_md,
                            f"SRS_{datetime.now():%Y%m%d_%H%M}.md",
                            "text/markdown",
                            key="dl_corrected_md"
                        )
                    with col_dl2:
                        st.download_button(
                            "Descargar SRS (HTML)",
                            srs_html,
                            f"SRS_{datetime.now():%Y%m%d_%H%M}.html",
                            "text/html",
                            key="dl_corrected_html"
                        )
                    with col_dl3:
                        # Plain text: just the corrected requirements
                        txt_lines = [c['corrected'] for c in corrections]
                        st.download_button(
                            "Descargar requisitos (TXT)",
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

    # Advertencias de entrada
    for w in check_input_warnings(req_text):
        st.caption(f"{'⚠️' if w['level'] == 'warning' else 'ℹ️'} {w['message']}")

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

    # Advertencias de entrada
    for w in check_input_warnings(req_text):
        st.caption(f"{'⚠️' if w['level'] == 'warning' else 'ℹ️'} {w['message']}")

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

    # Advertencias de entrada
    for w in check_input_warnings(req_a) + check_input_warnings(req_b):
        st.caption(f"{'⚠️' if w['level'] == 'warning' else 'ℹ️'} {w['message']}")

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
        result_files = sorted(EXPERIMENTS_DIR.glob("results_*.csv"))

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
                summary = metrics_df.groupby(['model', strategy_col]).agg({
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
                    try:
                        from analysis import plot_grouped_bars
                        fig = plot_grouped_bars(metrics_df, 'f1')
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Error generando grafica: {e}")
                with chart_tabs[1]:
                    try:
                        from analysis import plot_boxplots
                        fig = plot_boxplots(metrics_df, 'f1')
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Error generando grafica: {e}")
                with chart_tabs[2]:
                    try:
                        from analysis import plot_heatmap
                        fig = plot_heatmap(metrics_df, 'f1')
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Error generando grafica: {e}")
                with chart_tabs[3]:
                    try:
                        from analysis import plot_radar
                        fig = plot_radar(metrics_df)
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Error generando grafica: {e}")
                with chart_tabs[4]:
                    try:
                        from analysis import plot_speed_comparison
                        fig = plot_speed_comparison(metrics_df)
                        if fig:
                            st.pyplot(fig)
                            plt.close(fig)
                        else:
                            st.info("No hay datos de velocidad disponibles.")
                    except Exception as e:
                        st.error(f"Error generando grafica: {e}")
            else:
                st.subheader("Datos")
                st.dataframe(df, use_container_width=True)


# ============================================================
# PAGE 6: Comparar Modelos
# ============================================================
elif page == "Comparar Modelos":
    st.title("Comparacion de Modelos")

    result_files = sorted(EXPERIMENTS_DIR.glob("results_*.csv"))

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
        local_models = [k for k, v in MODEL_CONFIGS.items() if v['type'] == 'ollama']
        api_models = [k for k, v in MODEL_CONFIGS.items() if v['type'] == 'nvidia_nim']

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


# ============================================================
# PAGE 7: Progreso Experimentos
# ============================================================
elif page == "Progreso Experimentos":
    st.title("Progreso de Experimentos")
    st.markdown("Monitoriza el avance de los experimentos en tiempo real.")

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # ── Configuracion esperada ────────────────────────────────
    EXPECTED_TASKS = ['classification', 'ambiguity', 'completeness', 'inconsistency', 'testability']
    EXPECTED_MODELS = list(MODEL_CONFIGS.keys())
    EXPECTED_STRATEGIES = list(STRATEGY_LABELS.keys())
    EXPECTED_ITERATIONS = 5
    EXPECTED_SEEDS = [42, 123, 456, 789, 1024]
    CONFIGS_PER_TASK = len(EXPECTED_MODELS) * len(EXPECTED_STRATEGIES) * EXPECTED_ITERATIONS  # 150
    TOTAL_CONFIGS = len(EXPECTED_TASKS) * CONFIGS_PER_TASK  # 750
    STALE_SECONDS = 600  # 10 min

    task_labels = {
        'classification': 'Clasificacion F/NF',
        'ambiguity': 'Deteccion Ambiguedad',
        'completeness': 'Eval. Completitud',
        'inconsistency': 'Deteccion Inconsistencias',
        'testability': 'Eval. Testabilidad',
    }

    # ── Funciones de carga ────────────────────────────────────
    def _scan_all_files():
        """Escanea checkpoints y resultados, clasifica por frescura."""
        now = time.time()
        files = []
        if CHECKPOINTS_DIR.exists():
            for f in sorted(CHECKPOINTS_DIR.glob("checkpoint_*.json")):
                task_name = f.stem.replace("checkpoint_", "").rsplit("_", 2)[0]
                if task_name not in EXPECTED_TASKS:
                    continue
                mtime = f.stat().st_mtime
                files.append({
                    'path': f, 'name': f.name, 'task': task_name,
                    'type': 'checkpoint', 'mtime': mtime,
                    'active': (now - mtime) <= STALE_SECONDS,
                })
        if EXPERIMENTS_DIR.exists():
            for f in sorted(EXPERIMENTS_DIR.glob("results_*.csv")):
                task_name = f.stem.replace("results_", "").rsplit("_", 2)[0]
                if task_name not in EXPECTED_TASKS:
                    continue
                mtime = f.stat().st_mtime
                files.append({
                    'path': f, 'name': f.name, 'task': task_name,
                    'type': 'result', 'mtime': mtime,
                    'active': (now - mtime) <= STALE_SECONDS,
                })
        return files

    def _load_file_records(file_info):
        """Carga registros de un checkpoint o resultado."""
        if file_info['type'] == 'checkpoint':
            try:
                with open(file_info['path'], 'r') as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, OSError):
                return []
        else:
            try:
                return pd.read_csv(file_info['path']).to_dict('records')
            except (pd.errors.EmptyDataError, OSError):
                return []

    def _build_all_progress(files):
        """Carga TODOS los archivos y devuelve configs por tarea + tiempos."""
        task_configs = {task: set() for task in EXPECTED_TASKS}
        # Tiempos: model -> [time_seconds, ...]
        time_by_model = {m: [] for m in EXPECTED_MODELS}
        all_records_flat = []
        for f in files:
            for r in _load_file_records(f):
                strategy = r.get('strategy', r.get('pattern', ''))
                key = (r.get('model', ''), strategy, r.get('iteration', 0), r.get('seed', 0))
                task_configs[f['task']].add(key)
                t = r.get('time_seconds', 0)
                m = r.get('model', '')
                if t and m in time_by_model:
                    time_by_model[m].append(t)
                all_records_flat.append(r)
        return task_configs, time_by_model, all_records_flat

    def _select_display_files(files):
        """Un archivo por tarea: el mas relevante.

        Prioridad por tarea:
        1. Checkpoint activo mas reciente (indica proceso en curso)
        2. Resultado CSV mas reciente (experimento terminado)
        3. Checkpoint inactivo mas reciente (experimento interrumpido)
        """
        best_per_task = {}
        for f in files:
            t = f['task']
            prev = best_per_task.get(t)
            if prev is None:
                best_per_task[t] = f
                continue
            # Prioridad: checkpoint activo > resultado > checkpoint inactivo
            def _priority(fi):
                if fi['type'] == 'checkpoint' and fi['active']:
                    return (2, fi['mtime'])
                elif fi['type'] == 'result':
                    return (1, fi['mtime'])
                else:
                    return (0, fi['mtime'])
            if _priority(f) > _priority(prev):
                best_per_task[t] = f
        return list(best_per_task.values())

    all_files_raw = _scan_all_files()

    if not all_files_raw:
        st.warning("No hay checkpoints ni resultados en el directorio de resultados.")
        st.info("Ejecuta experimentos con `python experiment.py` para generar checkpoints.")
        st.stop()

    # ── Bloque auto-actualizable ──────────────────────────────
    @st.fragment(run_every=30)
    def _render_progress():
        # Rescanear archivos para detectar activos en tiempo real
        fresh_files = _select_display_files(_scan_all_files())
        # (el resumen se muestra tras construir proc_list)
        t_palette_map = dict(zip(EXPECTED_TASKS,
                                 ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']))

        # ── Preparar datos POR ARCHIVO (= proceso) ─────────────
        proc_list = []
        for f in fresh_files:
            records = _load_file_records(f)
            configs = set()
            for r in records:
                strategy = r.get('strategy', r.get('pattern', ''))
                key = (r.get('model', ''), strategy,
                       r.get('iteration', 0), r.get('seed', 0))
                configs.add(key)
            # Detectar si es local, API o mixto
            models_in = set(k[0] for k in configs)
            local_models = {k for k, v in MODEL_CONFIGS.items() if v['type'] == 'ollama'}
            nim_models = {k for k, v in MODEL_CONFIGS.items() if v['type'] == 'nvidia_nim'}
            has_local = bool(models_in & local_models)
            has_nim = bool(models_in & nim_models)
            if has_local and has_nim:
                origin = "Local+API"
            elif has_nim:
                origin = "API"
            elif has_local:
                origin = "Local"
            else:
                origin = ""
            # Configs esperadas segun modelos presentes en este proceso
            n_models_in = len(models_in) if models_in else len(EXPECTED_MODELS)
            expected_configs = n_models_in * len(EXPECTED_STRATEGIES) * EXPECTED_ITERATIONS
            # Activo = checkpoint reciente que no ha completado todas las configs
            # Un resultado CSV (type=result) nunca es "en curso"
            is_active = (f['type'] == 'checkpoint' and f['active']
                         and len(configs) < expected_configs)
            # ETA estimado por proceso
            cfg_times = {}
            for r in records:
                s = r.get('strategy', r.get('pattern', ''))
                k = (r.get('model', ''), s, r.get('iteration', 0), r.get('seed', 0))
                cfg_times[k] = cfg_times.get(k, 0) + (r.get('time_seconds', 0) or 0)
            avg_t = (sum(cfg_times.values()) / len(cfg_times)) if cfg_times else 0
            eta = (expected_configs - len(configs)) * avg_t
            proc_list.append({
                'file': f, 'configs': configs, 'records': records,
                'active': is_active, 'task': f['task'], 'origin': origin,
                'eta_sec': eta, 'avg_time': avg_t,
                'expected_configs': expected_configs,
            })

        # Ordenar: activos primero, luego por tarea
        proc_list.sort(key=lambda p: (not p['active'],
                                       EXPECTED_TASKS.index(p['task'])
                                       if p['task'] in EXPECTED_TASKS else 99))

        if not proc_list:
            st.info("No hay datos de experimentos todavia.")
            return

        # ── Resumen rapido ─────────────────────────────────────
        n_active = sum(1 for p in proc_list if p['active'])
        if n_active:
            st.success(f"{n_active} proceso(s) activo(s) de {len(proc_list)} total(es)")
        else:
            st.info(f"{len(proc_list)} proceso(s) encontrado(s) — ninguno activo")

        # ── Selector de proceso ───────────────────────────────
        def _fmt_eta(secs):
            if secs < 60:
                return f'{secs:.0f}s'
            elif secs < 3600:
                return f'{secs/60:.0f}min'
            else:
                return f'{secs/3600:.1f}h'

        options = []
        for i, p in enumerate(proc_list):
            lbl = task_labels.get(p['task'], p['task'])
            n_done = len(p['configs'])
            pct = n_done / p['expected_configs'] * 100
            tag = p['origin']
            if p['active']:
                eta_txt = f" | ETA ~{_fmt_eta(p['eta_sec'])}" if p['eta_sec'] > 0 else ""
                options.append(f"🟢 {lbl} [{tag}] — {pct:.0f}% en curso{eta_txt}")
            elif p['file']['type'] == 'result':
                options.append(f"✅ {lbl} [{tag}] — {pct:.0f}% completada")
            else:
                options.append(f"⚪ {lbl} [{tag}] — {pct:.0f}% interrumpida")

        selected_idx = st.selectbox(
            "Proceso", range(len(options)),
            format_func=lambda i: options[i],
            key='_progress_proc_select',
        )

        td = proc_list[selected_idx]
        task_name = td['task']
        proc_id = f"{task_name}_{selected_idx}"
        color = t_palette_map.get(task_name, '#3498db')
        n_done = len(td['configs'])
        expected = td['expected_configs']
        pct_done = n_done / expected

        # ── Calcular datos del proceso seleccionado ─────────
        model_counts = {m: 0 for m in EXPECTED_MODELS}
        for cfg in td['configs']:
            if cfg[0] in model_counts:
                model_counts[cfg[0]] += 1
        max_per_model_task = len(EXPECTED_STRATEGIES) * EXPECTED_ITERATIONS  # 25

        # Tokens/s por modelo (solo este proceso)
        tps_by_model = {}
        for r in td['records']:
            m = r.get('model', '')
            tps = r.get('tokens_per_second', 0)
            if m in EXPECTED_MODELS and tps:
                tps_by_model.setdefault(m, []).append(tps)

        # ETA ya precalculado en proc_list
        avg_time = td['avg_time']
        eta_sec = td['eta_sec']
        eta_str = _fmt_eta(eta_sec)

        models_seen = set(k[0] for k in td['configs'])
        strats_seen = set(k[1] for k in td['configs'])
        status_parts = [
            f"**{n_done}/{expected}** configs",
            f"{len(models_seen)} modelos",
            f"{len(strats_seen)}/{len(EXPECTED_STRATEGIES)} estrategias",
        ]
        if td['active']:
            status_parts.append(f"ETA: **~{eta_str}** (~{avg_time:.1f}s/config)")
        st.caption(" | ".join(status_parts))

        # ── Graficas (2 columnas) ─────────────────────────
        col_left, col_right = st.columns(2)

        # Izquierda: Progreso por modelo
        with col_left:
            m_active = [m for m in EXPECTED_MODELS if model_counts[m] > 0]
            m_all = m_active if m_active else EXPECTED_MODELS
            m_labels = [MODEL_LABELS.get(m, m) for m in m_all]
            m_values = [model_counts[m] for m in m_all]
            m_colors = ['#2ecc71' if m.startswith('nim_') else '#3498db' for m in m_all]
            m_pcts = [v / max_per_model_task * 100 for v in m_values]

            fig1 = go.Figure()
            fig1.add_trace(go.Bar(
                y=m_labels, x=m_values, orientation='h',
                marker_color=m_colors,
                text=[f'{p:.0f}%' for p in m_pcts],
                textposition='outside', textfont=dict(size=10),
                hovertemplate='%{y}<br>Configs: %{x}/' +
                              str(max_per_model_task) +
                              '<br>Progreso: %{text}<extra></extra>',
            ))
            fig1.add_vline(x=max_per_model_task, line_dash="dot",
                          line_color="#555", opacity=0.5)
            fig1.update_layout(
                title=dict(text='Progreso por modelo', font=dict(size=13)),
                xaxis_title='Configuraciones completadas',
                height=250, margin=dict(l=10, r=10, t=35, b=30),
                font=dict(size=10),
                showlegend=False,
            )
            fig1.update_xaxes(gridcolor='rgba(136,136,136,0.2)',
                              zeroline=False, showline=False)
            fig1.update_yaxes(gridcolor='rgba(136,136,136,0.2)',
                              zeroline=False, showline=False)
            st.plotly_chart(fig1, use_container_width=True,
                           config={'displayModeBar': False},
                           key=f'prog_model_{proc_id}')

        # Derecha: Velocidad (tokens/s)
        with col_right:
            tps_models = [m for m in EXPECTED_MODELS if m in tps_by_model]
            fig2 = go.Figure()
            if tps_models:
                tps_labels = [MODEL_LABELS.get(m, m) for m in tps_models]
                tps_vals = [sum(tps_by_model[m]) / len(tps_by_model[m])
                            for m in tps_models]
                tps_colors = ['#2ecc71' if m.startswith('nim_') else '#3498db'
                              for m in tps_models]
                tps_counts = [len(tps_by_model[m]) for m in tps_models]
                fig2.add_trace(go.Bar(
                    y=tps_labels, x=tps_vals, orientation='h',
                    marker_color=tps_colors,
                    text=[f'{v:.0f}' for v in tps_vals],
                    textposition='outside', textfont=dict(size=10),
                    hovertemplate='%{y}<br>Media: %{x:.1f} tok/s<br>'
                                  'Muestras: %{customdata}<extra></extra>',
                    customdata=tps_counts,
                ))
            fig2.update_layout(
                title=dict(text='Velocidad media', font=dict(size=13)),
                xaxis_title='Tokens/segundo',
                height=250, margin=dict(l=10, r=10, t=35, b=30),
                font=dict(size=10),
                showlegend=False,
            )
            fig2.update_xaxes(gridcolor='rgba(136,136,136,0.2)',
                              zeroline=False, showline=False)
            fig2.update_yaxes(gridcolor='rgba(136,136,136,0.2)',
                              zeroline=False, showline=False)
            if not tps_models:
                fig2.add_annotation(text='Sin datos de velocidad',
                                    x=0.5, y=0.5, xref='paper', yref='paper',
                                    showarrow=False, font=dict(size=12, color='#888'))
            st.plotly_chart(fig2, use_container_width=True,
                           config={'displayModeBar': False},
                           key=f'prog_tps_{proc_id}')

        # ── Tabla detalle modelo x estrategia ─────────────
        proc_models = [m for m in EXPECTED_MODELS if model_counts[m] > 0]
        if not proc_models:
            proc_models = list(models_seen) if models_seen else EXPECTED_MODELS

        with st.expander("Detalle modelo x estrategia", expanded=td['active']):
            matrix = {}
            time_matrix = {}
            for model in proc_models:
                matrix[model] = {}
                time_matrix[model] = {}
                for strat in EXPECTED_STRATEGIES:
                    count = sum(
                        1 for it in range(1, EXPECTED_ITERATIONS + 1)
                        for seed in EXPECTED_SEEDS
                        if (model, strat, it, seed) in td['configs']
                    )
                    matrix[model][strat] = count
                    # Tiempo total acumulado para esta celda
                    total_t = sum(
                        r.get('time_seconds', 0) or 0
                        for r in td['records']
                        if r.get('model') == model
                        and r.get('strategy', r.get('pattern', '')) == strat
                    )
                    time_matrix[model][strat] = total_t

            st.caption("Iteraciones completadas")
            matrix_df = pd.DataFrame(matrix).T
            matrix_df.columns = [STRATEGY_LABELS.get(s, s) for s in EXPECTED_STRATEGIES]
            matrix_df.index = [MODEL_LABELS.get(m, m) for m in proc_models]
            matrix_df.index.name = "Modelo"

            st.dataframe(
                matrix_df.style.background_gradient(
                    cmap='RdYlGn', vmin=0, vmax=EXPECTED_ITERATIONS
                ).format("{:.0f}/" + str(EXPECTED_ITERATIONS)),
                use_container_width=True
            )

            st.caption("Tiempo acumulado por modelo x estrategia")
            time_df = pd.DataFrame(time_matrix).T
            time_df.columns = [STRATEGY_LABELS.get(s, s) for s in EXPECTED_STRATEGIES]
            time_df.index = [MODEL_LABELS.get(m, m) for m in proc_models]
            time_df.index.name = "Modelo"

            def _fmt_time(v):
                if v < 60:
                    return f"{v:.0f}s"
                elif v < 3600:
                    return f"{v/60:.1f}m"
                else:
                    return f"{v/3600:.1f}h"

            # vmax = mayor tiempo observado o al menos 1s para evitar todo-rojo con un solo modelo
            _t_max = max(time_df.values.max(), 1)
            st.dataframe(
                time_df.style.background_gradient(
                    cmap='YlOrRd', vmin=0, vmax=_t_max
                ).format(_fmt_time),
                use_container_width=True
            )

        # ── Detalle de archivos ───────────────────────────────
        with st.expander("Detalle de archivos monitorizados"):
            file_rows = []
            for p in proc_list:
                f = p['file']
                file_rows.append({
                    'Tarea': task_labels.get(f['task'], f['task']),
                    'Tipo': 'Checkpoint' if f['type'] == 'checkpoint' else 'Resultado final',
                    'Archivo': f['name'],
                    'Registros': len(p['records']),
                    'Configs': len(p['configs']),
                    'Origen': p['origin'],
                    'Estado': 'En curso' if p['active'] else ('Completado' if f['type'] == 'result' else 'Interrumpido'),
                    'Ultima modificacion': datetime.fromtimestamp(f['mtime']).strftime('%Y-%m-%d %H:%M:%S'),
                })
            st.dataframe(pd.DataFrame(file_rows), use_container_width=True)

        st.caption(f"Ultima actualizacion: {datetime.now().strftime('%H:%M:%S')}")

    st.markdown("---")
    _render_progress()

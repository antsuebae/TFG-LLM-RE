"""
Frontend Streamlit para el TFG de Ingenieria de Requisitos con LLMs.

Paginas:
1. Análisis de Documento - Analisis completo multi-modelo sobre un documento
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

# Plotly en tema claro de forma global
try:
    import plotly.io as pio
    pio.templates.default = "plotly_white"
except ImportError:
    pass

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
    page_title="Verificación de Requisitos con IA",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Acento global: eliminar rojo de Streamlit, usar azul acero ── */
    :root {
        --primary-color: #2471A3 !important;
    }
    /* Cualquier elemento que use el rojo de acento de Streamlit */
    a, a:visited { color: #2471A3 !important; }
    [style*="color: rgb(255, 75, 75)"],
    [style*="color: #ff4b4b"],
    [style*="color:#ff4b4b"] {
        color: #2471A3 !important;
    }
    /* Spinner / status activo */
    [data-testid="stSpinner"] svg circle,
    [data-testid="stSpinner"] svg path { stroke: #2471A3 !important; }

    /* ── Base: fondo blanco, texto oscuro ── */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stApp"], .stApp, .main,
    [data-testid="stMain"], [data-testid="stMainBlockContainer"] {
        background-color: #F7F9FC !important;
        color: #1C2833 !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stHeader"] {
        background-color: #FFFFFF !important;
        border-bottom: 1px solid #E0E6ED;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    /* Eliminar fondo oscuro residual en el cuerpo Streamlit */
    .block-container {
        background-color: transparent !important;
        padding-top: 2rem;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1B3A5C 0%, #1B4F72 100%) !important;
        border-right: none;
    }
    section[data-testid="stSidebar"] * {
        color: #E8EEF4 !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] .stMarkdown h1 {
        color: #FFFFFF !important;
        font-size: 1.3rem !important;
        letter-spacing: 0.04em;
        font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #C5D8E8 !important;
        font-size: 0.9rem;
    }
    section[data-testid="stSidebar"] .stRadio [aria-checked="true"] ~ span {
        color: #FFFFFF !important;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.15) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stCaption"] {
        color: #8AABC5 !important;
        font-size: 0.78rem;
    }

    /* ── Encabezados ── */
    h1 { color: #1B3A5C !important; font-weight: 700 !important; font-size: 1.8rem !important; }
    h2 { color: #1B3A5C !important; font-weight: 600 !important; }
    h3 { color: #1B4F72 !important; font-weight: 600 !important; }
    h4, h5, h6 { color: #2C3E50 !important; }
    p, li, span, label { color: #1C2833; }

    /* ── Métricas en tarjeta ── */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E0E6ED;
        border-left: 4px solid #1B4F72;
        border-radius: 8px;
        padding: 14px 16px !important;
        box-shadow: 0 1px 6px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricLabel"] > div {
        color: #5D6D7E !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    [data-testid="stMetricValue"] > div {
        color: #1B3A5C !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricDelta"] { color: #2E86C1 !important; }

    /* ── Botones primarios ── */
    .stButton > button[kind="primary"],
    .stButton > button[kind="primaryFormSubmit"],
    button[data-testid="baseButton-primary"],
    button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #2E86C1, #85C1E9) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.55rem 1.4rem !important;
        box-shadow: 0 2px 8px rgba(46,134,193,0.3) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"]:hover,
    button[data-testid="baseButton-primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover {
        background: linear-gradient(135deg, #1A6FA0, #5DADE2) !important;
        box-shadow: 0 4px 14px rgba(46,134,193,0.4) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button[kind="secondary"],
    button[data-testid="baseButton-secondary"],
    button[data-testid="stBaseButton-secondary"] {
        border: 1px solid #2E86C1 !important;
        color: #2E86C1 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        background-color: #FFFFFF !important;
    }
    .stButton > button[kind="secondary"]:hover,
    button[data-testid="stBaseButton-secondary"]:hover {
        background-color: #EAF4FB !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #EAF0F6 !important;
        border-radius: 10px;
        padding: 4px !important;
        gap: 2px;
        border-bottom: none !important;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #5D6D7E !important;
        font-weight: 500;
        border-radius: 8px;
        padding: 8px 16px !important;
        font-size: 0.875rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2E86C1 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 6px rgba(46,134,193,0.25);
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none !important; }
    .stTabs [data-baseweb="tab-border"]    { display: none !important; }

    /* ── Inputs y textareas ── */
    [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input {
        background-color: #FFFFFF !important;
        border: 1px solid #D5DBDB !important;
        border-radius: 8px !important;
        color: #1C2833 !important;
        font-size: 0.9rem;
    }
    [data-testid="stTextArea"] textarea:focus,
    [data-testid="stTextInput"] input:focus {
        border-color: #2E86C1 !important;
        box-shadow: 0 0 0 2px rgba(46,134,193,0.15) !important;
    }

    /* ── Selectbox y multiselect ── */
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stMultiSelect"] > div > div {
        background-color: #FFFFFF !important;
        border: 1px solid #D5DBDB !important;
        border-radius: 8px !important;
        color: #1C2833 !important;
    }

    /* ── Expanders ── */
    details[data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #D5DBDB !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        overflow: hidden;
    }
    details[data-testid="stExpander"] summary,
    [data-testid="stExpanderHeader"],
    details[data-testid="stExpander"] > summary {
        background-color: #FFFFFF !important;
        color: #1B3A5C !important;
        font-weight: 600 !important;
        padding: 12px 16px !important;
    }
    details[data-testid="stExpander"] summary *,
    [data-testid="stExpanderHeader"] * {
        color: #1B3A5C !important;
        background-color: transparent !important;
    }
    details[data-testid="stExpander"] summary:hover,
    [data-testid="stExpanderHeader"]:hover {
        background-color: #F0F5FA !important;
    }
    details[data-testid="stExpander"] summary svg {
        fill: #5D6D7E !important;
    }

    /* ── Alertas/info/warning ── */
    [data-testid="stAlert"] {
        border-radius: 8px;
        border-left-width: 4px;
    }

    /* ── DataFrames ── */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #E0E6ED;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }

    /* ── Progress bars ── */
    [data-testid="stProgressBar"] > div {
        background-color: #E0E6ED;
        border-radius: 4px;
    }
    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #1B4F72, #2E86C1);
        border-radius: 4px;
    }

    /* ── Download buttons ── */
    .stDownloadButton > button {
        border-radius: 8px;
        border: 1px solid #1B4F72 !important;
        color: #1B4F72 !important;
        font-weight: 500;
    }
    .stDownloadButton > button:hover {
        background-color: #EAF0F6 !important;
    }

    /* ── Divider ── */
    hr {
        border-color: #E0E6ED !important;
        margin: 1.2rem 0;
    }

    /* ── Checkboxes y radios: label ── */
    [data-testid="stCheckbox"] label,
    [data-testid="stRadio"] label {
        color: #1C2833 !important;
        font-size: 0.9rem;
    }
    /* Radio: borde y punto interior del elemento seleccionado */
    [data-testid="stRadio"] [role="radio"][aria-checked="true"] {
        border-color: #2E86C1 !important;
    }
    [data-testid="stRadio"] [role="radio"][aria-checked="true"] > div {
        background-color: #2E86C1 !important;
    }
    /* Checkbox: fondo cuando está marcado */
    [data-testid="stCheckbox"] [role="checkbox"][aria-checked="true"],
    [data-baseweb="checkbox"] [role="checkbox"][aria-checked="true"] {
        background-color: #2E86C1 !important;
        border-color: #2E86C1 !important;
    }
    /* Selectbox y multiselect: borde de foco y chips seleccionados */
    [data-testid="stSelectbox"] [data-baseweb="select"]:focus-within,
    [data-testid="stMultiSelect"] [data-baseweb="select"]:focus-within {
        border-color: #2E86C1 !important;
        box-shadow: 0 0 0 2px rgba(46,134,193,0.15) !important;
    }
    [data-baseweb="tag"] {
        background-color: #2E86C1 !important;
        border-color: #2E86C1 !important;
    }

    /* ── Slider ── */
    [data-testid="stSlider"] [role="slider"] {
        background-color: #2E86C1 !important;
    }

    /* ── Status container (spinner) ── */
    [data-testid="stStatusWidget"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E0E6ED;
        border-radius: 10px;
    }

    /* ── File uploader: fondo claro ── */
    [data-testid="stFileUploader"] {
        background-color: #FFFFFF !important;
        border-radius: 10px !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: #F7F9FC !important;
        border: 2px dashed #B0C4D8 !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        background-color: #EAF0F6 !important;
        border-color: #2E86C1 !important;
    }
    [data-testid="stFileUploaderDropzone"] *,
    [data-testid="stFileUploaderDropzoneInstructions"] * {
        color: #5D6D7E !important;
    }
    [data-testid="stFileUploaderFile"] {
        background-color: #EAF0F6 !important;
        border-radius: 6px !important;
        border: 1px solid #D5DBDB !important;
    }
    [data-testid="stFileUploaderFile"] * {
        color: #1C2833 !important;
    }
    /* Barra de progreso de subida */
    [data-testid="stFileUploaderDropzone"] > div:last-child {
        background-color: #EAF0F6 !important;
    }

    /* ── Multiselect tags: acero-azul en vez de rojo ── */
    [data-baseweb="tag"] {
        background-color: #2471A3 !important;
        border-color: #2471A3 !important;
        border-radius: 6px !important;
    }
    [data-baseweb="tag"] span,
    [data-baseweb="tag"] * {
        color: #FFFFFF !important;
    }
    [data-baseweb="tag"] svg path { fill: #FFFFFF !important; }

    /* ── Radio buttons: punto azul acero en vez de rojo ── */
    [data-baseweb="radio"] [data-checked="true"] > div > div {
        background-color: #2471A3 !important;
        border-color: #2471A3 !important;
    }
    [data-baseweb="radio"]:hover > div > div {
        border-color: #2471A3 !important;
    }
    /* Sidebar: dot de nav activo en azul claro (sobre fondo oscuro) */
    section[data-testid="stSidebar"] [data-baseweb="radio"] [data-checked="true"] > div > div {
        background-color: #A8D8EA !important;
        border-color: #A8D8EA !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="radio"] > div > div {
        border-color: rgba(168,216,234,0.5) !important;
    }

    /* ── Checkboxes: acento azul ── */
    [data-baseweb="checkbox"] [data-checked="true"] > div > div {
        background-color: #2471A3 !important;
        border-color: #2471A3 !important;
    }
    [data-baseweb="checkbox"]:hover > div > div {
        border-color: #2471A3 !important;
    }

    /* selector adicional para versiones antiguas de Streamlit */
    [data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #2E86C1, #85C1E9) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(46,134,193,0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Model configs ────────────────────────────────────────────
MODEL_CONFIGS = {
    "qwen7b": {"name": "qwen2.5:7b-instruct-q5_K_M", "type": "ollama", "short_name": "qwen7b"},
    "qwen9b": {"name": "qwen3.5:9b", "type": "ollama", "short_name": "qwen9b"},
    "llama8b": {"name": "llama3.1:8b-instruct-q4_K_M", "type": "ollama", "short_name": "llama8b"},
    "llama3b": {"name": "llama3.2:3b-instruct-q4_K_M", "type": "ollama", "short_name": "llama3b"},
    "nim_llama70b": {"name": "meta/llama-3.1-70b-instruct", "type": "nvidia_nim", "short_name": "nim_llama70b"},
    "nim_llama8b": {"name": "meta/llama-3.1-8b-instruct", "type": "nvidia_nim", "short_name": "nim_llama8b"},
    "nim_mistral": {"name": "mistralai/mistral-7b-instruct-v0.3", "type": "nvidia_nim", "short_name": "nim_mistral"},
}

MODEL_LABELS = {
    "qwen7b": "Qwen 2.5 7B (local)",
    "qwen9b": "Qwen 3.5 9B (local)",
    "llama8b": "Llama 8B (local)",
    "llama3b": "Llama 3.2 3B (local)",
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
st.sidebar.markdown("Verificación de calidad de requisitos asistida por IA")

_PAGES = ["Análisis de Documento", "Características de Calidad",
           "Resultados", "Progreso Experimentos"]

# Restaurar página desde URL al hacer F5
if "nav_page" not in st.session_state:
    _qp = st.query_params.get("page", _PAGES[0])
    st.session_state["nav_page"] = _qp if _qp in _PAGES else _PAGES[0]

def _on_nav_change():
    st.query_params["page"] = st.session_state["nav_page"]

page = st.sidebar.radio(
    "Navegación",
    _PAGES,
    key="nav_page",
    on_change=_on_nav_change,
)

st.sidebar.markdown("---")
nim_configured = bool(os.getenv("NVIDIA_API_KEY"))
_nim_status = "✓ Configurada" if nim_configured else "✗ No configurada"
st.sidebar.caption(f"NVIDIA NIM API: {_nim_status}")


# ============================================================
# HELPER: Run pipeline for one model (uses DAG)
# ============================================================
def run_pipeline_for_model(requirements: list[str], model_key: str,
                           strategy: str, skip_inconsistency: bool,
                           filepath: str = None,
                           use_llm_extraction: bool = False,
                           doc_name: str = 'documento',
                           context_prompt: str = '',
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
            context_prompt=context_prompt,
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
            context_prompt=context_prompt,
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
# PAGE 1: Análisis de Documento (MULTI-MODELO)
# ============================================================
if page == "Análisis de Documento":
    st.title("Verificación de Calidad de Requisitos Asistida por IA")
    st.markdown("Sube un documento y ejecuta el análisis con uno o varios modelos.")

    # ── Subida de documento ───────────────────────────────────
    # Leer de session_state para que persistan entre reruns
    requirements = st.session_state.get('pipeline_requirements', [])
    doc_filepath = st.session_state.get('pipeline_doc_filepath')
    doc_suffix = st.session_state.get('pipeline_doc_suffix', '')

    uploaded = st.file_uploader(
        "Documento de requisitos",
        type=['txt', 'md', 'csv', 'pdf'],
        help="Formatos admitidos: TXT, Markdown, CSV (una columna de texto) y PDF.",
        key="pipeline_file_uploader",
    )
    if uploaded:
        # Solo recargar si cambió el archivo
        if uploaded.name != st.session_state.get('pipeline_doc_name'):
            suffix = Path(uploaded.name).suffix
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded.read())
            tmp.close()
            from pipeline import load_requirements
            try:
                requirements = load_requirements(tmp.name)
                st.session_state['pipeline_requirements'] = requirements
                st.session_state['pipeline_doc_name'] = uploaded.name
                st.session_state['pipeline_doc_filepath'] = tmp.name
                st.session_state['pipeline_doc_suffix'] = suffix.lower()
                st.session_state['pipeline_load_source'] = 'uploader'
                doc_filepath = tmp.name
                doc_suffix = suffix.lower()
            except Exception as e:
                st.error(f"Error al cargar: {e}")
        if requirements:
            doc_name = st.session_state.get('pipeline_doc_name', uploaded.name)
            st.success(f"**{len(requirements)}** requisitos cargados desde **{doc_name}**")
    elif not uploaded and st.session_state.get('pipeline_load_source') == 'uploader':
        # Solo limpiar si el archivo vino del uploader y el usuario lo quitó
        for k in ['pipeline_requirements', 'pipeline_doc_name', 'pipeline_doc_filepath',
                  'pipeline_doc_suffix', 'pipeline_load_source']:
            st.session_state.pop(k, None)
        requirements = []

    # Mostrar el nombre del dataset cargado desde ejemplo (persiste entre reruns)
    if not uploaded and st.session_state.get('pipeline_doc_name'):
        st.success(f"**{len(requirements)}** requisitos cargados desde **{st.session_state['pipeline_doc_name']}**")

    # Datasets de ejemplo (colapsado por defecto)
    with st.expander("Cargar dataset de ejemplo", expanded=False):
        data_dir = Path(__file__).parent.parent / "data"
        doc_files = sorted(
            f for f in data_dir.glob("*") if f.suffix in ('.txt', '.md', '.csv', '.pdf')
        )
        if doc_files:
            selected_ex = st.selectbox(
                "Archivo de ejemplo",
                doc_files,
                format_func=lambda x: x.name,
                key="pipeline_example_file"
            )
            if st.button("Cargar archivo", type="primary", key="btn_load_example"):
                from pipeline import load_requirements
                try:
                    requirements = load_requirements(str(selected_ex))
                    st.session_state['pipeline_requirements'] = requirements
                    st.session_state['pipeline_doc_name'] = selected_ex.name
                    st.session_state['pipeline_doc_filepath'] = str(selected_ex)
                    st.session_state['pipeline_doc_suffix'] = selected_ex.suffix.lower()
                    st.session_state['pipeline_load_source'] = 'example'
                except Exception as e:
                    st.error(f"Error al cargar: {e}")
        else:
            st.info("No hay archivos de ejemplo disponibles en /data.")

    if requirements:
        # Preview
        with st.expander(f"Vista previa ({len(requirements)} requisitos)"):
            for i, r in enumerate(requirements, 1):
                st.markdown(f"**{i}.** {r[:200]}")

        st.markdown("---")

        # ── Contexto del dominio (prompt previo) ─────────────
        context_prompt = st.text_area(
            "Contexto del dominio (opcional)",
            height=80,
            placeholder=(
                "Ej: Los casos de uso son siempre requisitos funcionales. "
                "En este dominio, 'rápido' significa menos de 2 segundos y no se considera ambiguo."
            ),
            help=(
                "Este contexto se añade al inicio de cada prompt para guiar el análisis "
                "con información específica del dominio. No reemplaza los criterios de análisis, "
                "los complementa."
            ),
            key="pipeline_context_prompt",
        )

        if context_prompt.strip():
            try:
                from langdetect import detect, LangDetectException
                sample_text = " ".join(requirements[:5])
                req_lang = detect(sample_text)
                ctx_lang = detect(context_prompt)
                if req_lang != ctx_lang:
                    st.warning(
                        "El idioma del contexto no coincide con el de los requisitos. "
                        "El modelo puede ignorar el contexto si no están en el mismo idioma."
                    )
            except Exception:
                pass

        st.markdown("---")

        # ── Configuración ─────────────────────────────────────
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
            skip_inconsistency = st.checkbox("Saltar inconsistencias (más rápido)", value=False)
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

            st.info("El análisis se está ejecutando. **No cambies de página** hasta que termine — navegar interrumpe el progreso visual.")
            st.session_state.pop('pipeline_results', None)  # limpiar resultados anteriores

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
                            context_prompt=context_prompt,
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
                        st.info(f"Requisitos extraídos por LLM: {len(extracted_requirements)} (se reutilizarán)")
                    except Exception as e:
                        status_container.update(label=f"Combo 1/{total_combos}: {combo_label} - Error", state="error", expanded=False)
                        st.error(f"Error en extracción LLM: {e}")
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
                            context_prompt=context_prompt,
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

            all_results = {k: v for k, v in all_results.items() if not v['results_df'].empty}
            if not all_results:
                st.error("Todas las ejecuciones fallaron. Revisa los logs del modelo.")
                st.stop()

            import plotly.graph_objects as go

            st.markdown("---")
            st.header("Resultados del análisis")

            combo_keys = list(all_results.keys())

            # ── Comparación si hay varios combos ─────────────
            if len(combo_keys) > 1:
                cmp_labels, cmp_quality, cmp_colors = [], [], []
                for (mk, sk), res in all_results.items():
                    avg_q = res['results_df']['quality_score'].mean() if 'quality_score' in res['results_df'].columns else 0
                    cmp_labels.append(f"{MODEL_LABELS.get(mk, mk)}<br>{STRATEGY_LABELS.get(sk, sk)}")
                    cmp_quality.append(avg_q)
                order = sorted(range(len(cmp_quality)), key=lambda i: cmp_quality[i], reverse=True)
                cmp_labels = [cmp_labels[i] for i in order]
                cmp_quality = [cmp_quality[i] for i in order]
                cmp_colors = ['#1B4F72' if i == 0 else '#5D9BD5' for i in range(len(cmp_labels))]
                fig_cmp = go.Figure(go.Bar(
                    x=cmp_labels, y=cmp_quality,
                    marker_color=cmp_colors,
                    text=[f"{q:.0f}%" for q in cmp_quality],
                    textposition='outside',
                ))
                fig_cmp.update_layout(
                    title=dict(text="Calidad media por modelo y estrategia", font=dict(size=14)),
                    yaxis=dict(title="Calidad media (%)", range=[0, 115]),
                    xaxis_title="", height=320,
                    margin=dict(l=40, r=40, t=50, b=40),
                    paper_bgcolor='white', plot_bgcolor='#F7F9FC',
                )
                fig_cmp.update_yaxes(gridcolor='rgba(128,128,128,0.15)', zeroline=False)
                st.plotly_chart(fig_cmp, use_container_width=True, config={'displayModeBar': False}, key="fig_cmp_overview")

            # ── Tabs: una por combo ───────────────────────────
            if len(combo_keys) > 1:
                _tab_labels = [
                    f"{MODEL_LABELS.get(k[0], k[0])} · {STRATEGY_LABELS.get(k[1], k[1])}"
                    for k in combo_keys
                ]
                result_tabs = st.tabs(_tab_labels)
            else:
                result_tabs = [st.container()]

            for result_tab, combo_key in zip(result_tabs, combo_keys):
                with result_tab:
                    res = all_results[combo_key]
                    df = res['results_df']
                    inconsistencies = res['inconsistencies']
                    model_key, strat = combo_key

                    if len(combo_keys) == 1:
                        st.caption(
                            f"Modelo: **{MODEL_LABELS.get(model_key, model_key)}** · "
                            f"Estrategia: **{STRATEGY_LABELS.get(strat, strat)}** · "
                            f"Tiempo: {res['elapsed']:.1f}s"
                        )

                    total = len(df)
                    n_amb = int((df['is_ambiguous'] == True).sum()) if 'is_ambiguous' in df.columns else 0
                    n_inc = int((df['is_complete'] == False).sum()) if 'is_complete' in df.columns else 0
                    n_nt  = int((df['is_testable'] == False).sum()) if 'is_testable' in df.columns else 0
                    n_f   = int((df['classification'] == 'F').sum()) if 'classification' in df.columns else 0
                    avg_q = df['quality_score'].mean() if 'quality_score' in df.columns else 0

                    _ok_mask = pd.Series(True, index=df.index)
                    if 'is_ambiguous' in df.columns: _ok_mask &= (df['is_ambiguous'] != True)
                    if 'is_complete'  in df.columns: _ok_mask &= (df['is_complete']  != False)
                    if 'is_testable'  in df.columns: _ok_mask &= (df['is_testable']  != False)
                    n_ok = int(_ok_mask.sum())

                    # ── KPI cards ─────────────────────────────
                    k1, k2, k3, k4, k5 = st.columns(5)
                    k1.metric("Calidad media", f"{avg_q:.0f}%")
                    k2.metric("Sin problemas", f"{n_ok} / {total}")
                    k3.metric("Ambiguos",     n_amb, delta=f"{n_amb/total*100:.0f}%" if total else None, delta_color="inverse")
                    k4.metric("Incompletos",  n_inc, delta=f"{n_inc/total*100:.0f}%" if total else None, delta_color="inverse")
                    k5.metric("No testables", n_nt,  delta=f"{n_nt/total*100:.0f}%"  if total else None, delta_color="inverse")

                    # ── Gráfico: perfil de calidad ─────────────
                    cats = ['F/NF — Funcionales', 'A2 — Sin ambigüedad', 'A3 — Completos', 'V2 — Testables']
                    ok_vals  = [n_f, total - n_amb, total - n_inc, total - n_nt]
                    bad_vals = [total - v for v in ok_vals]

                    fig_prof = go.Figure()
                    fig_prof.add_trace(go.Bar(
                        name='Correcto', y=cats, x=ok_vals, orientation='h',
                        marker_color='#27AE60', opacity=0.85,
                        text=[str(v) for v in ok_vals],
                        textposition='inside', insidetextanchor='middle',
                    ))
                    fig_prof.add_trace(go.Bar(
                        name='Con problema', y=cats, x=bad_vals, orientation='h',
                        marker_color='#E74C3C', opacity=0.75,
                        text=[str(v) for v in bad_vals],
                        textposition='inside', insidetextanchor='middle',
                    ))
                    fig_prof.update_layout(
                        barmode='stack',
                        title=dict(text='Perfil de calidad del documento', font=dict(size=14)),
                        xaxis=dict(title=f'Requisitos (total: {total})', range=[0, total + 1]),
                        yaxis_title='',
                        height=250,
                        margin=dict(l=20, r=20, t=50, b=20),
                        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='right', x=1),
                        paper_bgcolor='white', plot_bgcolor='#F7F9FC',
                    )
                    fig_prof.update_xaxes(gridcolor='rgba(128,128,128,0.15)', zeroline=False)
                    st.plotly_chart(fig_prof, use_container_width=True, config={'displayModeBar': False}, key=f"fig_prof_{combo_key[0]}_{combo_key[1]}")

                    # ── Tabla color-coded ─────────────────────
                    st.subheader("Tabla de requisitos")
                    _col_map = {
                        'text': 'Requisito', 'classification': 'F/NF',
                        'is_ambiguous': 'Ambiguo', 'is_complete': 'Completo',
                        'is_testable': 'Testable', 'quality_score': 'Calidad',
                    }
                    _avail = [c for c in _col_map if c in df.columns]
                    disp = df[_avail].rename(columns=_col_map).copy()
                    if 'Requisito' in disp.columns:
                        disp['Requisito'] = disp['Requisito'].str[:120]
                    if 'Calidad' in disp.columns:
                        disp['Calidad'] = disp['Calidad'].apply(lambda x: f"{x:.0f}%")
                    for bool_col in ('Ambiguo', 'Completo', 'Testable'):
                        if bool_col in disp.columns:
                            disp[bool_col] = disp[bool_col].map({True: 'Sí', False: 'No'}).fillna('')
                    disp.index = range(1, len(disp) + 1)

                    def _row_bg(row):
                        n_issues = (
                            int(row.get('Ambiguo') == 'Sí') +
                            int(row.get('Completo') == 'No') +
                            int(row.get('Testable') == 'No')
                        )
                        color = '#eafaf1' if n_issues == 0 else ('#fef9e7' if n_issues == 1 else '#fde8e4')
                        return [f'background-color: {color}'] * len(row)

                    st.dataframe(
                        disp.style.apply(_row_bg, axis=1),
                        use_container_width=True, height=380, hide_index=False,
                    )
                    st.caption("Verde = sin problemas · Amarillo = 1 problema · Rojo = 2 o más problemas")

                    # ── Problemas agrupados ───────────────────
                    total_issues = n_amb + n_inc + n_nt + len(inconsistencies)
                    if total_issues > 0:
                        st.subheader("Problemas detectados")
                        pc1, pc2 = st.columns(2)

                        with pc1:
                            if n_amb > 0 and 'is_ambiguous' in df.columns:
                                amb_rows = df[df['is_ambiguous'] == True]
                                with st.expander(f"Requisitos ambiguos — {n_amb}", expanded=n_amb <= 5):
                                    for _, row in amb_rows.iterrows():
                                        st.markdown(f"· {row['text'][:150]}")
                                        parts = []
                                        if row.get('ambiguity_type'): parts.append(f"Tipo: {row['ambiguity_type']}")
                                        if row.get('ambiguous_words'): parts.append(f"Palabras: *{row['ambiguous_words']}*")
                                        if parts: st.caption(" · ".join(parts))

                            if n_nt > 0 and 'is_testable' in df.columns:
                                nt_rows = df[df['is_testable'] == False]
                                _reason_map = {'measurable': 'Medible', 'vague': 'Vago', 'subjective': 'Subjetivo'}
                                with st.expander(f"Requisitos no testables — {n_nt}", expanded=n_nt <= 5):
                                    for _, row in nt_rows.iterrows():
                                        st.markdown(f"· {row['text'][:150]}")
                                        r = _reason_map.get(row.get('testability_reason', ''), row.get('testability_reason', ''))
                                        if r: st.caption(f"Motivo: {r}")

                        with pc2:
                            if n_inc > 0 and 'is_complete' in df.columns:
                                inc_rows = df[df['is_complete'] == False]
                                with st.expander(f"Requisitos incompletos — {n_inc}", expanded=n_inc <= 5):
                                    for _, row in inc_rows.iterrows():
                                        st.markdown(f"· {row['text'][:150]}")
                                        if row.get('missing_elements'): st.caption(f"Falta: {row['missing_elements']}")

                            if len(inconsistencies) > 0:
                                with st.expander(f"Inconsistencias entre requisitos — {len(inconsistencies)}", expanded=len(inconsistencies) <= 3):
                                    for inc in inconsistencies:
                                        st.warning(f"**Req #{inc['req_a_idx']}** vs **Req #{inc['req_b_idx']}**: {inc['description']}")

                    # ── Advertencias automáticas ──────────────
                    pipeline_warnings = generate_all_warnings(df)
                    _render_warnings(pipeline_warnings, "Advertencias de calidad")

                    # ── Descarga CSV ──────────────────────────
                    st.download_button(
                        "Descargar CSV",
                        df.to_csv(index=False),
                        f"pipeline_{model_key}_{strat}_{datetime.now():%Y%m%d_%H%M}.csv",
                        "text/csv",
                        key=f"download_{model_key}_{strat}",
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
                                model_key, strat, doc_name, output_dir,
                                context_prompt=context_prompt,
                                skip_inconsistency=skip_inconsistency,
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
# PAGE 2: Características de Calidad — 5 pestañas
# ============================================================
elif page == "Características de Calidad":
    st.title("Características de Calidad de Requisitos")
    st.markdown(
        "Analiza un requisito individualmente según las cinco características de calidad del experimento."
    )

    (tab_a1, tab_a2, tab_a3, tab_v1, tab_v2) = st.tabs([
        "A1 · Clasificación F/NF",
        "A2 · Ambigüedad",
        "A3 · Completitud",
        "V1 · Inconsistencias",
        "V2 · Testabilidad",
    ])

    # ── Helper: widget de configuración reutilizable ──────────
    def _cq_config(key_prefix: str, with_temperature: bool = False):
        col1, col2 = st.columns([2, 1])
        with col2:
            mk = st.selectbox("Modelo", list(MODEL_CONFIGS.keys()),
                              format_func=lambda x: MODEL_LABELS.get(x, x),
                              key=f"{key_prefix}_model")
            sk = st.selectbox("Estrategia", STRATEGY_NAMES,
                              format_func=lambda x: STRATEGY_LABELS.get(x, x),
                              key=f"{key_prefix}_strat")
            temp = 0.4
            if with_temperature:
                temp = st.slider("Temperatura", 0.0, 1.0, 0.4, 0.1,
                                 key=f"{key_prefix}_temp")
        return col1, mk, sk, temp

    def _cq_timing(response):
        c1, c2 = st.columns(2)
        c1.metric("Tiempo", f"{response['time_seconds']:.2f}s")
        c2.metric("Tokens/s", f"{response.get('tokens_per_second', 0):.1f}")

    # ── A1: Clasificación F/NF ────────────────────────────────
    with tab_a1:
        st.caption("Determina si un requisito es **Funcional (F)** — describe *qué* hace el sistema — "
                   "o **No Funcional (NF)** — describe *cómo* se comporta.")
        col1, mk, sk, temp = _cq_config("a1", with_temperature=True)
        with col1:
            req_a1 = st.text_area(
                "Requisito",
                value="El sistema deberá permitir a los usuarios restablecer su contraseña "
                      "mediante correo electrónico.",
                height=110, key="a1_req"
            )
        for w in check_input_warnings(req_a1):
            st.caption(f"{'⚠️' if w['level'] == 'warning' else 'ℹ️'} {w['message']}")

        if st.button("Clasificar", type="primary", key="btn_a1"):
            with st.spinner("Clasificando..."):
                try:
                    model = get_model_instance(mk, temp)
                    response = model.generate(
                        build_prompt("classification", sk, requirement=req_a1), max_tokens=512
                    )
                    if response['success']:
                        pred = parse_response("classification", response['content'])
                        label = {"F": "Funcional", "NF": "No Funcional"}.get(pred, "No determinado")
                        color = "green" if pred == "F" else ("orange" if pred == "NF" else "red")
                        st.markdown(f"### Resultado: :{color}[{pred} — {label}]")
                        _cq_timing(response)
                        with st.expander("Respuesta completa del modelo"):
                            st.text(response['content'])
                    else:
                        st.error(f"Error: {response['error']}")
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── A2: Ambigüedad ────────────────────────────────────────
    with tab_a2:
        st.caption("Detecta si el requisito contiene **términos vagos**, **pronombres sin referencia** "
                   "o **cuantificadores imprecisos** que impidan una única interpretación.")
        col1, mk, sk, _ = _cq_config("a2")
        with col1:
            req_a2 = st.text_area(
                "Requisito",
                value="El sistema deberá ser rápido y manejar bien los datos del usuario.",
                height=110, key="a2_req"
            )
        for w in check_input_warnings(req_a2):
            st.caption(f"{'⚠️' if w['level'] == 'warning' else 'ℹ️'} {w['message']}")

        if st.button("Detectar ambigüedad", type="primary", key="btn_a2"):
            with st.spinner("Analizando..."):
                try:
                    model = get_model_instance(mk)
                    response = model.generate(
                        build_prompt("ambiguity", sk, requirement=req_a2), max_tokens=512
                    )
                    if response['success']:
                        parsed = parse_response("ambiguity", response['content'])
                        is_amb = parsed.get('is_ambiguous', False)
                        color = "red" if is_amb else "green"
                        st.markdown(f"### Resultado: :{color}[{'Ambiguo' if is_amb else 'No ambiguo'}]")
                        if is_amb:
                            amb_type = parsed.get('ambiguity_type', '')
                            words = parsed.get('ambiguous_words', [])
                            if amb_type:
                                st.markdown(f"**Tipo:** `{amb_type}`")
                            if words:
                                items = words if isinstance(words, list) else [words]
                                st.markdown(f"**Palabras ambiguas:** {', '.join(f'`{w}`' for w in items)}")
                        _cq_timing(response)
                        with st.expander("Respuesta completa del modelo"):
                            st.text(response['content'])
                    else:
                        st.error(f"Error: {response['error']}")
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── A3: Completitud ───────────────────────────────────────
    with tab_a3:
        st.caption("Evalúa si el requisito especifica todos los elementos necesarios: "
                   "manejo de errores, condiciones límite, criterios de aceptación y precondiciones.")
        col1, mk, sk, _ = _cq_config("a3")
        with col1:
            req_a3 = st.text_area(
                "Requisito",
                value="El sistema deberá procesar los pagos de los usuarios.",
                height=110, key="a3_req"
            )
        for w in check_input_warnings(req_a3):
            st.caption(f"{'⚠️' if w['level'] == 'warning' else 'ℹ️'} {w['message']}")

        if st.button("Evaluar completitud", type="primary", key="btn_a3"):
            with st.spinner("Analizando..."):
                try:
                    model = get_model_instance(mk)
                    response = model.generate(
                        build_prompt("completeness", sk, requirement=req_a3), max_tokens=512
                    )
                    if response['success']:
                        parsed = parse_response("completeness", response['content'])
                        is_comp = parsed.get('is_complete', False)
                        color = "green" if is_comp else "red"
                        st.markdown(f"### Resultado: :{color}[{'Completo' if is_comp else 'Incompleto'}]")
                        missing = parsed.get('missing_elements', [])
                        if missing:
                            st.markdown("**Elementos faltantes:**")
                            items = missing if isinstance(missing, list) else [missing]
                            for elem in items:
                                st.markdown(f"- `{elem}`")
                        _cq_timing(response)
                        with st.expander("Respuesta completa del modelo"):
                            st.text(response['content'])
                    else:
                        st.error(f"Error: {response['error']}")
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── V1: Inconsistencias ───────────────────────────────────
    with tab_v1:
        st.caption("Detecta contradicciones entre pares de requisitos. "
                   "Introduce uno por línea; se evaluarán todos los pares posibles.")

        _CONSIST_PLACEHOLDER = (
            "El sistema almacenará todos los datos del usuario localmente en su dispositivo.\n"
            "Todos los datos de usuario deben almacenarse en una base de datos en la nube.\n"
            "El sistema responderá a cualquier consulta en menos de 1 segundo.\n"
            "El sistema procesará consultas complejas en un máximo de 10 segundos."
        )
        reqs_raw = st.text_area("Requisitos (uno por línea)", height=180,
                                 placeholder=_CONSIST_PLACEHOLDER, key="consist_raw")

        _cc1, _cc2 = st.columns(2)
        with _cc1:
            model_key_v1 = st.selectbox("Modelo", list(MODEL_CONFIGS.keys()),
                                         key="consist_model",
                                         format_func=lambda x: MODEL_LABELS.get(x, x))
        with _cc2:
            strategy_v1 = st.selectbox("Estrategia", STRATEGY_NAMES,
                                        key="consist_strategy",
                                        format_func=lambda x: STRATEGY_LABELS.get(x, x))

        MAX_REQS = 15
        reqs = [r.strip() for r in reqs_raw.strip().splitlines() if r.strip()]
        if len(reqs) > MAX_REQS:
            st.warning(f"Máximo {MAX_REQS} requisitos para evitar tiempos excesivos. "
                       f"Se usarán los primeros {MAX_REQS}.")
            reqs = reqs[:MAX_REQS]
        if reqs:
            import itertools
            pairs = list(itertools.combinations(range(len(reqs)), 2))
            st.info(f"{len(reqs)} requisitos → **{len(pairs)} pares** a evaluar")

        if st.button("Detectar inconsistencias", type="primary",
                     disabled=(len(reqs) < 2), key="btn_consist"):
            st.session_state.pop("consist_results", None)
            _model = get_model_instance(model_key_v1)
            _results = []
            _pbar = st.progress(0, text="Evaluando pares...")
            _total = len(pairs)
            for _idx, (i, j) in enumerate(pairs):
                try:
                    _prompt = build_prompt("inconsistency", strategy_v1,
                                           requirement_a=reqs[i], requirement_b=reqs[j])
                    _resp = _model.generate(_prompt, max_tokens=512)
                    _parsed = parse_response("inconsistency", _resp['content']) if _resp['success'] else {}
                    _results.append({
                        'i': i, 'j': j, 'req_a': reqs[i], 'req_b': reqs[j],
                        'is_inconsistent': bool(_parsed.get('is_inconsistent', False)),
                        'description': _parsed.get('description', ''),
                        'success': _resp['success'],
                        'time': _resp.get('time_seconds', 0),
                    })
                except Exception as _e:
                    _results.append({
                        'i': i, 'j': j, 'req_a': reqs[i], 'req_b': reqs[j],
                        'is_inconsistent': False, 'description': str(_e),
                        'success': False, 'time': 0,
                    })
                _pbar.progress((_idx + 1) / _total,
                               text=f"Evaluando par {_idx + 1}/{_total}...")
            _pbar.empty()
            st.session_state["consist_results"] = {"reqs": reqs, "results": _results}

        if "consist_results" in st.session_state:
            _data = st.session_state["consist_results"]
            _reqs = _data["reqs"]
            _results = _data["results"]
            _n = len(_reqs)
            _inconsistent = [r for r in _results if r['is_inconsistent']]
            _errors = [r for r in _results if not r['success']]

            st.divider()
            _k1, _k2, _k3, _k4 = st.columns(4)
            _k1.metric("Requisitos", _n)
            _k2.metric("Pares evaluados", len(_results))
            _k3.metric("Inconsistencias", len(_inconsistent))
            _k4.metric("Consistencia", f"{(1 - len(_inconsistent)/max(len(_results),1))*100:.0f}%")
            if _errors:
                st.warning(f"{len(_errors)} pares no pudieron evaluarse.")

            st.divider()
            import plotly.graph_objects as go
            _labels = [f"R{i+1}" for i in range(_n)]
            _matrix = [[None]*_n for _ in range(_n)]
            for r in _results:
                _matrix[r['i']][r['j']] = 1 if r['is_inconsistent'] else 0
                _matrix[r['j']][r['i']] = 1 if r['is_inconsistent'] else 0
            for i in range(_n):
                _matrix[i][i] = -1
            _hover = [["" ]*_n for _ in range(_n)]
            for r in _results:
                _txt = "Inconsistentes" if r['is_inconsistent'] else "Consistentes"
                if r['description']:
                    _txt += f"<br>{r['description'][:120]}"
                _hover[r['i']][r['j']] = _txt
                _hover[r['j']][r['i']] = _txt
            for i in range(_n):
                _hover[i][i] = f"R{i+1}: {_reqs[i][:60]}..."
            _fig_mat = go.Figure(go.Heatmap(
                z=_matrix, x=_labels, y=_labels, text=_hover,
                hovertemplate="%{text}<extra></extra>",
                colorscale=[[0, '#2ecc71'], [0.4, '#f39c12'], [0.6, '#e74c3c'], [1, '#e74c3c']],
                zmin=-1, zmax=1, showscale=False, xgap=2, ygap=2,
            ))
            _fig_mat.update_layout(
                title=dict(text="Matriz de consistencia (rojo = inconsistente, verde = consistente)",
                           font=dict(size=13, color='#1B3A5C')),
                height=max(300, _n * 45 + 100),
                margin=dict(l=40, r=20, t=50, b=40),
                paper_bgcolor='white', plot_bgcolor='#F7F9FC',
                yaxis=dict(autorange='reversed'),
            )
            st.plotly_chart(_fig_mat, width='stretch', config={'displayModeBar': False})
            if _inconsistent:
                st.subheader(f"Inconsistencias detectadas ({len(_inconsistent)})")
                for r in _inconsistent:
                    with st.expander(f"R{r['i']+1} vs R{r['j']+1}"):
                        _ic1, _ic2 = st.columns(2)
                        _ic1.markdown(f"**R{r['i']+1}:** {r['req_a']}")
                        _ic2.markdown(f"**R{r['j']+1}:** {r['req_b']}")
                        if r['description']:
                            st.error(r['description'])
            else:
                st.success("No se detectaron inconsistencias entre los requisitos.")
            _csv_rows = [{
                'req_a_idx': r['i']+1, 'req_b_idx': r['j']+1,
                'req_a': r['req_a'], 'req_b': r['req_b'],
                'inconsistente': r['is_inconsistent'], 'descripcion': r['description'],
            } for r in _results]
            st.download_button(
                "Descargar resultados CSV",
                pd.DataFrame(_csv_rows).to_csv(index=False),
                "inconsistencias.csv", "text/csv", key="dl_consist_csv",
            )

    # ── V2: Testabilidad ──────────────────────────────────────
    with tab_v2:
        st.caption("Evalúa si el requisito es **verificable objetivamente**: "
                   "debe contener métricas o criterios concretos que permitan diseñar "
                   "un caso de prueba con resultado pass/fail.")
        col1, mk, sk, _ = _cq_config("v2")
        with col1:
            req_v2 = st.text_area(
                "Requisito",
                value="El sistema deberá ser fácil de usar y agradable para los usuarios.",
                height=110, key="v2_req"
            )
        for w in check_input_warnings(req_v2):
            st.caption(f"{'⚠️' if w['level'] == 'warning' else 'ℹ️'} {w['message']}")

        if st.button("Evaluar testabilidad", type="primary", key="btn_v2"):
            with st.spinner("Analizando..."):
                try:
                    model = get_model_instance(mk)
                    response = model.generate(
                        build_prompt("testability", sk, requirement=req_v2), max_tokens=512
                    )
                    if response['success']:
                        parsed = parse_response("testability", response['content'])
                        is_test = parsed.get('is_testable', False)
                        color = "green" if is_test else "red"
                        st.markdown(f"### Resultado: :{color}[{'Testable' if is_test else 'No testable'}]")
                        reason = parsed.get('reason', '')
                        if reason:
                            _reason_labels = {
                                'measurable': 'Medible — tiene criterios cuantificables',
                                'vague': 'Vago — sin métricas precisas',
                                'subjective': 'Subjetivo — depende de interpretación personal',
                            }
                            st.markdown(f"**Motivo:** {_reason_labels.get(reason, reason)}")
                        _cq_timing(response)
                        with st.expander("Respuesta completa del modelo"):
                            st.text(response['content'])
                    else:
                        st.error(f"Error: {response['error']}")
                except Exception as e:
                    st.error(f"Error: {e}")


# ============================================================
# PAGE 5: Resultados de Experimentos
# ============================================================
elif page == "Resultados":
    st.title("Dashboard de Resultados")

    tab_pipeline, tab_bench = st.tabs(
        ["Pipeline (documentos)", "Benchmark y Comparativa"]
    )

    with tab_pipeline:
        # ── Ejecuciones del pipeline con metadata ─────────
        from pipeline import load_pipeline_runs
        pipeline_dir = RESULTS_DIR / "pipeline"
        runs = load_pipeline_runs(pipeline_dir)

        if not runs:
            st.warning("No hay ejecuciones del pipeline guardadas.")
            st.info("Ejecuta el pipeline desde 'Análisis de Documento' y pulsa 'Guardar Informes'.")
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
            st.dataframe(display_runs_df, width='stretch')

            # Seleccionar ejecuciones para comparar
            run_labels = [f"{r['Fecha']} | {r['Modelo']} | {r['Estrategia']} | {r['Documento']}" for r in runs_summary]
            selected_runs = st.multiselect("Seleccionar ejecuciones para comparar", run_labels)

            if selected_runs:
                selected_indices = [run_labels.index(label) for label in selected_runs]

                if len(selected_indices) >= 2:
                    st.subheader("Comparación de ejecuciones seleccionadas")
                    compare_rows = [runs_summary[i] for i in selected_indices]
                    compare_df = pd.DataFrame(compare_rows).drop(columns=['run_dir'])
                    st.dataframe(compare_df, width='stretch')

                # Detalle de cada ejecucion seleccionada
                for idx in selected_indices:
                    run = runs_summary[idx]
                    run_path = Path(run['run_dir'])
                    csv_path = run_path / "results.csv"
                    if csv_path.exists():
                        with st.expander(f"Detalle: {run['Modelo']} + {run['Estrategia']} ({run['Fecha']})"):
                            run_df = pd.read_csv(csv_path)
                            _col_map = {
                                'text': 'Requisito', 'classification': 'Tipo',
                                'is_ambiguous': 'Ambiguo', 'is_complete': 'Completo',
                                'is_testable': 'Testable', 'quality_score': 'Calidad',
                            }
                            avail = [c for c in _col_map if c in run_df.columns]
                            display_df = run_df[avail].rename(columns=_col_map).copy()
                            if 'Requisito' in display_df.columns:
                                display_df['Requisito'] = display_df['Requisito'].str[:100]
                            display_df.index = range(1, len(display_df) + 1)
                            st.dataframe(display_df, width='stretch')

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

    with tab_bench:
        # ── Resultados de experimentos (benchmark) ────────
        import matplotlib.pyplot as plt
        from analysis import compute_metrics_per_config

        TASK_DISPLAY = {
            'classification':  'Clasificación F/NF',
            'ambiguity':       'Detección de Ambigüedad',
            'completeness':    'Evaluación de Completitud',
            'inconsistency':   'Detección de Inconsistencias',
            'testability':     'Evaluación de Testabilidad',
        }

        def _load_benchmark(task: str, version: str) -> pd.DataFrame:
            """Carga y concatena todos los CSVs de una tarea+version."""
            dirs = []
            if version in ("v1", "Ambas"):
                dirs.append(EXPERIMENTS_DIR / "v1")
            if version in ("v2", "Ambas"):
                dirs.append(EXPERIMENTS_DIR / "v2")
            dfs = []
            for d in dirs:
                if d.exists():
                    for f in sorted(d.glob(f"results_{task}_*.csv")):
                        try:
                            _df = pd.read_csv(f)
                            _df['_version'] = d.name
                            dfs.append(_df)
                        except Exception:
                            pass
            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        # ── Controles ─────────────────────────────────────
        ctrl_col1, ctrl_col2 = st.columns([1, 2])
        with ctrl_col1:
            version_sel = st.radio(
                "Versión",
                ["v2"],
                horizontal=True,
                key="bench_version",
            )
        with ctrl_col2:
            # Solo mostrar tareas que tienen datos en la version seleccionada
            available_tasks = []
            for t in TASK_DISPLAY:
                dirs_to_check = []
                if version_sel in ("v1", "Ambas"):
                    dirs_to_check.append(EXPERIMENTS_DIR / "v1")
                if version_sel in ("v2", "Ambas"):
                    dirs_to_check.append(EXPERIMENTS_DIR / "v2")
                has_data = any(
                    list(d.glob(f"results_{t}_*.csv"))
                    for d in dirs_to_check if d.exists()
                )
                if has_data:
                    available_tasks.append(t)

            if not available_tasks:
                st.warning("No hay resultados para la versión seleccionada.")
                st.stop()

            task_sel = st.selectbox(
                "Tarea",
                available_tasks,
                format_func=lambda x: TASK_DISPLAY.get(x, x),
                key="bench_task",
            )

        df = _load_benchmark(task_sel, version_sel)

        if df.empty:
            st.info("No hay datos para esta selección. Ejecuta los experimentos primero.")
        else:
            strategy_col = 'strategy' if 'strategy' in df.columns else 'pattern'

            if 'model' in df.columns and strategy_col in df.columns and 'iteration' in df.columns:
                metrics_df = compute_metrics_per_config(df, task_sel)
                summary = metrics_df.groupby(['model', strategy_col]).agg(
                    f1_mean=('f1', 'mean'), f1_std=('f1', 'std'),
                    accuracy=('accuracy', 'mean'),
                    precision=('precision', 'mean'),
                    recall=('recall', 'mean'),
                ).round(3).reset_index()

                # ── KPI cards ─────────────────────────────
                best_row = summary.loc[summary['f1_mean'].idxmax()]
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Mejor F1", f"{best_row['f1_mean']:.3f}",
                          delta=f"±{best_row['f1_std']:.3f}")
                k2.metric("Mejor modelo",
                          MODEL_LABELS.get(best_row['model'], best_row['model']))
                k3.metric("Mejor estrategia",
                          STRATEGY_LABELS.get(best_row[strategy_col], best_row[strategy_col]))
                k4.metric("Configuraciones", len(summary))

                st.divider()

                # ── Visualizaciones ───────────────────────
                chart_tabs = st.tabs(["Heatmap F1", "Barras", "Boxplot", "Radar", "Velocidad",
                                      "F1 vs Tiempo", "Ranking Estrategias", "Local vs API"])
                with chart_tabs[0]:
                    try:
                        from analysis import plot_heatmap
                        fig = plot_heatmap(metrics_df, 'f1')
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Error generando gráfica: {e}")
                with chart_tabs[1]:
                    try:
                        from analysis import plot_grouped_bars
                        fig = plot_grouped_bars(metrics_df, 'f1')
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Error generando gráfica: {e}")
                with chart_tabs[2]:
                    try:
                        from analysis import plot_boxplots
                        fig = plot_boxplots(metrics_df, 'f1')
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Error generando gráfica: {e}")
                with chart_tabs[3]:
                    try:
                        from analysis import plot_radar
                        fig = plot_radar(metrics_df)
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Error generando gráfica: {e}")
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
                        st.error(f"Error generando gráfica: {e}")

                with chart_tabs[5]:
                    try:
                        import plotly.graph_objects as go
                        # Agregar tiempo medio por modelo+estrategia
                        _time_agg = metrics_df.groupby(['model', 'strategy']).agg(
                            avg_time=('avg_time_seconds', 'mean'),
                            avg_tps=('avg_tokens_per_second', 'mean'),
                        ).reset_index()
                        _scatter_df = summary.merge(_time_agg, left_on=['model', strategy_col],
                                                    right_on=['model', 'strategy'], how='left')

                        if _scatter_df['avg_time'].isna().all():
                            st.info("No hay datos de tiempo disponibles.")
                        else:
                            _model_colors = {
                                'qwen7b': '#2980b9', 'qwen9b': '#1abc9c', 'llama8b': '#1a5276', 'llama3b': '#5dade2',
                                'nim_llama70b': '#c0392b', 'nim_llama8b': '#922b21', 'nim_mistral': '#f1948a',
                            }
                            _strat_symbols = {s: sym for sym, s in enumerate(
                                ['circle', 'square', 'diamond', 'cross', 'x'],
                            )}  # reutilizamos idx
                            _strat_list = list(STRATEGY_LABELS.keys())
                            _strat_sym_map = {s: i for i, s in enumerate(_strat_list)}

                            fig_sc = go.Figure()
                            for _, r in _scatter_df.iterrows():
                                if pd.isna(r.get('avg_time')):
                                    continue
                                _mlabel = MODEL_LABELS.get(r['model'], r['model'])
                                _slabel = STRATEGY_LABELS.get(r[strategy_col], r[strategy_col])
                                fig_sc.add_trace(go.Scatter(
                                    x=[r['avg_time']],
                                    y=[r['f1_mean']],
                                    mode='markers',
                                    marker=dict(
                                        size=14,
                                        color=_model_colors.get(r['model'], '#888'),
                                        symbol=_strat_sym_map.get(r[strategy_col], 0),
                                        line=dict(width=1.5, color='white'),
                                    ),
                                    name=f"{_mlabel} · {_slabel}",
                                    hovertemplate=(
                                        f"<b>{_mlabel}</b><br>"
                                        f"Estrategia: {_slabel}<br>"
                                        f"F1: {r['f1_mean']:.3f} ± {r['f1_std']:.3f}<br>"
                                        f"Tiempo medio: {r['avg_time']:.2f} s/req<br>"
                                        + (f"Velocidad: {r['avg_tps']:.1f} tok/s" if r.get('avg_tps') else "")
                                        + "<extra></extra>"
                                    ),
                                ))

                            fig_sc.update_layout(
                                title=dict(text="F1 vs Tiempo de inferencia por configuración",
                                           font=dict(size=14)),
                                xaxis_title="Tiempo medio por requisito (s)",
                                yaxis_title="F1-score",
                                yaxis=dict(range=[0, 1.05]),
                                height=480,
                                legend=dict(orientation='v', yanchor='top', y=1,
                                            xanchor='left', x=1.02, font=dict(size=10)),
                                margin=dict(l=50, r=200, t=50, b=50),
                            )
                            fig_sc.update_xaxes(gridcolor='rgba(128,128,128,0.2)', zeroline=False)
                            fig_sc.update_yaxes(gridcolor='rgba(128,128,128,0.2)', zeroline=False)
                            st.plotly_chart(fig_sc, width='stretch',
                                            config={'displayModeBar': False})
                            st.caption("Cada punto = modelo + estrategia. "
                                       "Esquina superior izquierda = mejor rendimiento en menos tiempo.")
                    except Exception as e:
                        st.error(f"Error generando gráfica: {e}")

                with chart_tabs[6]:
                    # Ranking de estrategias por modelo
                    try:
                        import plotly.graph_objects as go
                        _rank_df = metrics_df.groupby(['model', 'strategy'])['f1']\
                            .mean().reset_index()
                        _rank_df['rank'] = _rank_df.groupby('model')['f1']\
                            .rank(ascending=False, method='min').astype(int)
                        _rank_df['model_label'] = _rank_df['model'].map(
                            lambda x: MODEL_LABELS.get(x, x))
                        _strat_colors_r = {
                            'question_refinement': '#3498db', 'cognitive_verifier': '#e74c3c',
                            'persona_context': '#2ecc71', 'few_shot': '#f39c12',
                            'chain_of_thought': '#9b59b6',
                        }
                        _model_order_r = [m for m in MODEL_CONFIGS
                                          if m in _rank_df['model'].unique()]
                        _model_pos = {m: i for i, m in enumerate(_model_order_r)}
                        fig_rank = go.Figure()
                        for _s in _rank_df['strategy'].unique():
                            _sub = _rank_df[_rank_df['strategy'] == _s].copy()
                            _sub['_pos'] = _sub['model'].map(_model_pos)
                            _sub = _sub.sort_values('_pos')
                            _sl = STRATEGY_LABELS.get(_s, _s)
                            fig_rank.add_trace(go.Scatter(
                                x=_sub['model_label'], y=_sub['rank'],
                                mode='lines+markers', name=_sl,
                                line=dict(color=_strat_colors_r.get(_s, '#888'), width=2.5),
                                marker=dict(size=12, color=_strat_colors_r.get(_s, '#888'),
                                            line=dict(width=1.5, color='white')),
                                hovertemplate=(
                                    f"<b>{_sl}</b><br>Modelo: %{{x}}<br>"
                                    "Ranking: #%{y}<br>F1 medio: %{customdata:.3f}<extra></extra>"
                                ),
                                customdata=_sub['f1'],
                            ))
                        fig_rank.update_layout(
                            title=dict(text="Ranking de estrategias por modelo (1 = mejor F1)",
                                       font=dict(size=14)),
                            xaxis_title="Modelo",
                            yaxis=dict(title="Ranking", tickmode='linear', dtick=1,
                                       autorange='reversed', range=[5.4, 0.6]),
                            height=460,
                            legend=dict(orientation='h', yanchor='bottom', y=1.02,
                                        xanchor='right', x=1),
                            margin=dict(l=50, r=30, t=80, b=50),
                        )
                        fig_rank.update_xaxes(gridcolor='rgba(128,128,128,0.2)', zeroline=False)
                        fig_rank.update_yaxes(gridcolor='rgba(128,128,128,0.2)', zeroline=False)
                        st.plotly_chart(fig_rank, width='stretch',
                                        config={'displayModeBar': False})
                        st.caption("Lineas horizontales = la estrategia funciona igual en todos los modelos. "
                                   "Cruces entre lineas = el mejor prompt depende del modelo (RQ2).")
                    except Exception as e:
                        st.error(f"Error generando gráfica: {e}")

                with chart_tabs[7]:
                    # Local vs API distribución de F1
                    try:
                        import plotly.graph_objects as go
                        _cmp_df = metrics_df.copy()
                        _cmp_df['tipo'] = _cmp_df['model'].map(
                            lambda m: 'API (NIM)'
                            if MODEL_CONFIGS.get(m, {}).get('type') == 'nvidia_nim'
                            else 'Local (Ollama)'
                        )
                        _tipo_colors = {'Local (Ollama)': '#2980b9', 'API (NIM)': '#c0392b'}
                        fig_viol = go.Figure()
                        for _tipo in ['Local (Ollama)', 'API (NIM)']:
                            _sub = _cmp_df[_cmp_df['tipo'] == _tipo]
                            if _sub.empty:
                                continue
                            fig_viol.add_trace(go.Violin(
                                x=_sub['tipo'], y=_sub['f1'],
                                name=_tipo,
                                box_visible=True, meanline_visible=True,
                                fillcolor=_tipo_colors[_tipo], opacity=0.65,
                                line_color=_tipo_colors[_tipo],
                                points='all', pointpos=0, jitter=0.3,
                                marker=dict(size=5, opacity=0.5),
                                hovertemplate='F1: %{y:.3f}<extra>' + _tipo + '</extra>',
                            ))
                        fig_viol.update_layout(
                            title=dict(text="Distribución de F1: Modelos Locales vs API",
                                       font=dict(size=14)),
                            yaxis=dict(title="F1-score", range=[0, 1.05]),
                            xaxis_title="",
                            height=440,
                            showlegend=False,
                            violingap=0.3,
                            margin=dict(l=50, r=30, t=50, b=50),
                        )
                        fig_viol.update_xaxes(gridcolor='rgba(128,128,128,0.2)', zeroline=False)
                        fig_viol.update_yaxes(gridcolor='rgba(128,128,128,0.2)', zeroline=False)
                        st.plotly_chart(fig_viol, width='stretch',
                                        config={'displayModeBar': False})
                        _sc1, _sc2 = st.columns(2)
                        _local_f1 = _cmp_df[_cmp_df['tipo'] == 'Local (Ollama)']['f1']
                        _api_f1   = _cmp_df[_cmp_df['tipo'] == 'API (NIM)']['f1']
                        if not _local_f1.empty:
                            _sc1.metric("Local — F1 medio", f"{_local_f1.mean():.3f}",
                                        delta=f"σ={_local_f1.std():.3f}")
                        if not _api_f1.empty:
                            _sc2.metric("API — F1 medio", f"{_api_f1.mean():.3f}",
                                        delta=f"σ={_api_f1.std():.3f}")
                        st.caption("Responde RQ1: diferencia de rendimiento entre modelos locales y API. "
                                   "La caja interior muestra el rango intercuartilico; "
                                   "la linea central es la mediana.")
                    except Exception as e:
                        st.error(f"Error generando gráfica: {e}")

                st.divider()

                # ── Tabla de metricas ─────────────────────
                st.subheader("Tabla de métricas")
                display_summary = summary.copy()
                display_summary['modelo'] = display_summary['model'].map(
                    lambda x: MODEL_LABELS.get(x, x))
                display_summary['estrategia'] = display_summary[strategy_col].map(
                    lambda x: STRATEGY_LABELS.get(x, x))
                display_summary = display_summary[
                    ['modelo', 'estrategia', 'f1_mean', 'f1_std', 'accuracy', 'precision', 'recall']
                ].rename(columns={
                    'f1_mean': 'F1', 'f1_std': 'F1 ±',
                    'accuracy': 'Accuracy', 'precision': 'Precision', 'recall': 'Recall',
                }).sort_values('F1', ascending=False).reset_index(drop=True)

                st.dataframe(
                    display_summary.style.background_gradient(
                        subset=['F1', 'Accuracy'], cmap='RdYlGn', vmin=0.0, vmax=1.0
                    ),
                    width='stretch',
                    hide_index=True,
                )
            else:
                st.dataframe(df, width='stretch')

        # ── Comparativa Local vs API ──────────────────────────────
        st.divider()
        st.subheader("Comparativa Local vs API")
        import matplotlib.pyplot as plt
        from analysis import compute_metrics_per_config

        TASK_DISPLAY_CMP = {
            'classification':  'Clasificación F/NF',
            'ambiguity':       'Detección de Ambigüedad',
            'completeness':    'Evaluación de Completitud',
            'inconsistency':   'Detección de Inconsistencias',
            'testability':     'Evaluación de Testabilidad',
        }

        # ── Controles ─────────────────────────────────────────────
        ctrl1, ctrl2 = st.columns([1, 2])
        with ctrl1:
            version_cmp = st.radio("Versión", ["v2"], horizontal=True, key="cmp_version")
        with ctrl2:
            dirs_cmp = [v for v in ["v1", "v2"] if version_cmp in (v, "Ambas")]
            avail_tasks_cmp = [
                t for t in TASK_DISPLAY_CMP
                if any(list((EXPERIMENTS_DIR / v).glob(f"results_{t}_*.csv"))
                       for v in dirs_cmp if (EXPERIMENTS_DIR / v).exists())
            ]
            if not avail_tasks_cmp:
                st.warning("No hay resultados para la versión seleccionada.")
                st.stop()
            task_cmp = st.selectbox(
                "Tarea", avail_tasks_cmp,
                format_func=lambda x: TASK_DISPLAY_CMP.get(x, x),
                key="cmp_task",
            )

        df_cmp = _load_benchmark(task_cmp, version_cmp)

        if df_cmp.empty:
            st.info("No hay datos para esta selección. Ejecuta los experimentos primero.")
        else:
            strategy_col_cmp = 'strategy' if 'strategy' in df_cmp.columns else 'pattern'
            local_models = [k for k, v in MODEL_CONFIGS.items() if v['type'] == 'ollama']
            api_models   = [k for k, v in MODEL_CONFIGS.items() if v['type'] == 'nvidia_nim']

            if 'model' not in df_cmp.columns or strategy_col_cmp not in df_cmp.columns:
                st.dataframe(df_cmp, width='stretch')
            else:
                metrics_cmp = compute_metrics_per_config(df_cmp, task_cmp)
                local_m = metrics_cmp[metrics_cmp['model'].isin(local_models)]
                api_m   = metrics_cmp[metrics_cmp['model'].isin(api_models)]

                # ── KPI cards ──────────────────────────────
                _f1_local  = local_m['f1'].mean() if not local_m.empty else None
                _f1_api    = api_m['f1'].mean()   if not api_m.empty   else None
                _tps_local = local_m['avg_tokens_per_second'].mean() if (not local_m.empty and 'avg_tokens_per_second' in local_m.columns) else None
                _tps_api   = api_m['avg_tokens_per_second'].mean()   if (not api_m.empty   and 'avg_tokens_per_second' in api_m.columns)   else None

                if any([_f1_local, _f1_api, _tps_local, _tps_api]):
                    _cols = st.columns(sum([x is not None for x in [_f1_local, _f1_api, _tps_local, _tps_api]]))
                    _ci = 0
                    if _f1_local is not None:
                        _delta = f"{_f1_local - _f1_api:+.3f} vs API" if _f1_api is not None else None
                        _cols[_ci].metric("F1 Local (Ollama)", f"{_f1_local:.3f}", delta=_delta)
                        _ci += 1
                    if _f1_api is not None:
                        _delta = f"{_f1_api - _f1_local:+.3f} vs Local" if _f1_local is not None else None
                        _cols[_ci].metric("F1 API (NIM)", f"{_f1_api:.3f}", delta=_delta)
                        _ci += 1
                    if _tps_local is not None:
                        _cols[_ci].metric("Tokens/s Local", f"{_tps_local:.1f}")
                        _ci += 1
                    if _tps_api is not None:
                        _cols[_ci].metric("Tokens/s API", f"{_tps_api:.1f}")

                st.divider()

                # ── F1 por modelo ──────────────────────────
                st.subheader("F1 por modelo")
                model_agg = metrics_cmp.groupby('model').agg(
                    f1_mean=('f1', 'mean'), f1_std=('f1', 'std'),
                    accuracy=('accuracy', 'mean'),
                    precision=('precision', 'mean'),
                    recall=('recall', 'mean'),
                ).round(3).reset_index()
                model_agg['modelo'] = model_agg['model'].map(lambda x: MODEL_LABELS.get(x, x))
                model_agg = model_agg[['modelo', 'f1_mean', 'f1_std', 'accuracy', 'precision', 'recall']]\
                    .rename(columns={'f1_mean': 'F1', 'f1_std': 'F1 ±',
                                     'accuracy': 'Accuracy', 'precision': 'Precision', 'recall': 'Recall'})\
                    .sort_values('F1', ascending=False).reset_index(drop=True)
                st.dataframe(
                    model_agg.style.background_gradient(subset=['F1', 'Accuracy'], cmap='RdYlGn', vmin=0, vmax=1),
                    width='stretch', hide_index=True,
                )

                st.divider()

                # ── Mejor estrategia por modelo ────────────
                st.subheader("Mejor estrategia por modelo")
                best_s = metrics_cmp.groupby(['model', strategy_col_cmp])['f1'].mean().reset_index()
                best_pm = best_s.loc[best_s.groupby('model')['f1'].idxmax()].copy()
                best_pm['modelo']     = best_pm['model'].map(lambda x: MODEL_LABELS.get(x, x))
                best_pm['estrategia'] = best_pm[strategy_col_cmp].map(lambda x: STRATEGY_LABELS.get(x, x))
                best_pm = best_pm[['modelo', 'estrategia', 'f1']]\
                    .rename(columns={'f1': 'F1 medio'})\
                    .sort_values('F1 medio', ascending=False).reset_index(drop=True)
                st.dataframe(
                    best_pm.style.background_gradient(subset=['F1 medio'], cmap='RdYlGn', vmin=0, vmax=1),
                    width='stretch', hide_index=True,
                )

                st.divider()

                # ── Velocidad ──────────────────────────────
                st.subheader("Velocidad: tokens/s por modelo")
                if 'avg_tokens_per_second' in metrics_cmp.columns:
                    try:
                        from analysis import plot_speed_comparison
                        fig = plot_speed_comparison(metrics_cmp)
                        if fig:
                            st.pyplot(fig)
                            plt.close(fig)
                        else:
                            st.info("No hay datos de velocidad disponibles.")
                    except Exception as e:
                        st.error(f"Error generando gráfica: {e}")
                else:
                    st.info("No hay datos de velocidad en los resultados cargados.")

                st.divider()

                # ── Trade-offs ─────────────────────────────
                st.subheader("Trade-offs: Rendimiento vs Privacidad vs Coste")
                st.markdown("""
                | Aspecto | Modelos Locales (Ollama) | Modelos API (NVIDIA NIM) |
                |---------|-------------------------|--------------------------|
                | **Privacidad** | Total (datos no salen del equipo) | Parcial (datos enviados a API) |
                | **Coste** | Solo hardware (GPU) | Gratuito con limites / Pay-per-use |
                | **Latencia** | Dependiente de GPU local | Dependiente de red |
                | **Escalabilidad** | Limitada por hardware | Alta |
                | **Disponibilidad** | Siempre (offline) | Requiere conexión |
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
    STALE_SECONDS = 7200  # 2 horas — configs lentos (CoT con modelos locales) pueden tardar >10min

    task_labels = {
        'classification': 'Clasificación F/NF',
        'ambiguity': 'Detección Ambigüedad',
        'completeness': 'Eval. Completitud',
        'inconsistency': 'Detección Inconsistencias',
        'testability': 'Eval. Testabilidad',
    }

    # ── Funciones de carga ────────────────────────────────────
    def _scan_all_files():
        """Escanea checkpoints y resultados, clasifica por frescura."""
        now = time.time()
        files = []
        if CHECKPOINTS_DIR.exists():
            for f in sorted(CHECKPOINTS_DIR.glob("checkpoint_*.json")):
                try:
                    task_name = f.stem.replace("checkpoint_", "").rsplit("_", 2)[0]
                except Exception:
                    continue
                if not task_name or task_name not in EXPECTED_TASKS:
                    continue
                mtime = f.stat().st_mtime
                files.append({
                    'path': f, 'name': f.name, 'task': task_name,
                    'type': 'checkpoint', 'mtime': mtime,
                    'active': (now - mtime) <= STALE_SECONDS,
                })
        if EXPERIMENTS_DIR.exists():
            for f in sorted(EXPERIMENTS_DIR.rglob("results_*.csv")):
                try:
                    task_name = f.stem.replace("results_", "").rsplit("_", 2)[0]
                except Exception:
                    continue
                if not task_name or task_name not in EXPECTED_TASKS:
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
        """Archivos a mostrar en progreso.

        - Checkpoints activos: se muestran TODOS (puede haber local + API a la vez)
        - Tareas sin checkpoint activo: un archivo por tarea (resultado o checkpoint inactivo)
        """
        active_checkpoints = [f for f in files if f['type'] == 'checkpoint' and f['active']]
        active_tasks = {f['task'] for f in active_checkpoints}

        # Para tareas sin checkpoint activo, elegir el mejor archivo inactivo
        best_inactive = {}
        for f in files:
            if f['task'] in active_tasks:
                continue
            t = f['task']
            prev = best_inactive.get(t)
            if prev is None:
                best_inactive[t] = f
                continue
            def _priority(fi):
                return (1, fi['mtime']) if fi['type'] == 'result' else (0, fi['mtime'])
            if _priority(f) > _priority(prev):
                best_inactive[t] = f

        return active_checkpoints + list(best_inactive.values())

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
            # Activo = checkpoint reciente sin completar, o CSV recién escrito (experimento recién terminado)
            is_active = (
                (f['type'] == 'checkpoint' and f['active'] and len(configs) < expected_configs)
                or (f['type'] == 'result' and f['active'])
            )
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
            st.info("No hay datos de experimentos todavía.")
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
            m_colors = ['#2ecc71' if MODEL_CONFIGS.get(m, {}).get('type') == 'nvidia_nim' else '#3498db' for m in m_all]
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
            st.plotly_chart(fig1, width='stretch',
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
                tps_colors = ['#2ecc71' if MODEL_CONFIGS.get(m, {}).get('type') == 'nvidia_nim' else '#3498db'
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
            st.plotly_chart(fig2, width='stretch',
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
                width='stretch'
            )

            try:
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

                _t_max = max(float(time_df.values.max()), 1.0)
                st.dataframe(
                    time_df.style.background_gradient(
                        cmap='YlOrRd', vmin=0, vmax=_t_max
                    ).format(_fmt_time),
                    width='stretch'
                )
            except Exception:
                st.dataframe(time_df if 'time_df' in dir() else pd.DataFrame(),
                             width='stretch')

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
                    'Última modificación': datetime.fromtimestamp(f['mtime']).strftime('%Y-%m-%d %H:%M:%S'),
                })
            st.dataframe(pd.DataFrame(file_rows), width='stretch')

        st.caption(f"Última actualización: {datetime.now().strftime('%H:%M:%S')}")

    st.markdown("---")
    _render_progress()

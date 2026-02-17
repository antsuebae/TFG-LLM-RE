"""
Frontend Streamlit para el TFG de Ingenieria de Requisitos con LLMs.

Paginas:
1. Clasificar Requisito (F/NF)
2. Analizar Calidad (ambiguedad, completitud, testabilidad)
3. Validar Consistencia (pares de requisitos)
4. Resultados de Experimentos (dashboard)
5. Comparar Modelos (tabla comparativa)
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import json
import glob

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
    ["Clasificar Requisito", "Analizar Calidad", "Validar Consistencia",
     "Resultados Experimentos", "Comparar Modelos"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Modelos locales:** Ollama")
st.sidebar.markdown("**Modelos API:** NVIDIA NIM")
nim_configured = bool(os.getenv("NVIDIA_API_KEY"))
st.sidebar.markdown(f"NVIDIA NIM: {'Configurado' if nim_configured else 'No configurado'}")


# ============================================================
# PAGE 1: Clasificar Requisito
# ============================================================
if page == "Clasificar Requisito":
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
        model_key = st.selectbox("Modelo", list(MODEL_CONFIGS.keys()))
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
                        color = "green" if prediction == "F" else ("blue" if prediction == "NF" else "red")
                        label = {"F": "Funcional", "NF": "No Funcional"}.get(prediction, "No determinado")
                        st.metric("Clasificacion", f"{prediction} - {label}")
                    with col_r2:
                        st.metric("Tiempo", f"{response['time_seconds']:.2f}s")
                    with col_r3:
                        st.metric("Tokens/s", f"{response.get('tokens_per_second', 0):.1f}")

                    with st.expander("Respuesta completa del modelo"):
                        st.text(response['content'])
                    with st.expander("Prompt enviado"):
                        st.text(prompt)
                else:
                    st.error(f"Error: {response['error']}")
            except Exception as e:
                st.error(f"Error: {e}")


# ============================================================
# PAGE 2: Analizar Calidad
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
        model_key = st.selectbox("Modelo", list(MODEL_CONFIGS.keys()), key="quality_model")
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
# PAGE 3: Validar Consistencia
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

    model_key = st.selectbox("Modelo", list(MODEL_CONFIGS.keys()), key="consist_model")
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
# PAGE 4: Resultados de Experimentos
# ============================================================
elif page == "Resultados Experimentos":
    st.title("Dashboard de Resultados Experimentales")

    # Find result files
    result_files = sorted(RESULTS_DIR.glob("results_*.csv"))

    if not result_files:
        st.warning("No hay resultados de experimentos. Ejecuta el pipeline primero.")
        st.code("cd proyecto/src && python experiment.py --dry-run", language="bash")
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
                    f"**Modelos:** {df['model'].nunique()} | **Estrategias:** {df[strategy_col].nunique()}")

        # Import analysis functions
        from analysis import compute_metrics_per_config

        metrics_df = compute_metrics_per_config(df, task)

        # Summary table
        st.subheader("Resumen de Metricas")
        summary = metrics_df.groupby(['model', 'strategy']).agg({
            'f1': ['mean', 'std'],
            'accuracy': ['mean', 'std'],
            'precision': 'mean',
            'recall': 'mean',
        }).round(3)
        st.dataframe(summary, use_container_width=True)

        # Charts
        st.subheader("Visualizaciones")
        chart_tabs = st.tabs(["Barras", "Boxplot", "Heatmap", "Radar", "Velocidad"])

        with chart_tabs[0]:
            import matplotlib.pyplot as plt
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

        # Statistical analysis
        st.subheader("Analisis Estadistico")
        if len(metrics_df) >= 4:
            from analysis import full_statistical_analysis, generate_stats_table
            try:
                stat_results = full_statistical_analysis(metrics_df)
                stats_table = generate_stats_table(stat_results)
                st.markdown(stats_table)
            except Exception as e:
                st.warning(f"No se pudo realizar el analisis estadistico: {e}")
        else:
            st.info("Se necesitan mas datos para el analisis estadistico.")

        # Export
        st.subheader("Exportar")
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            csv_data = metrics_df.to_csv(index=False)
            st.download_button("Descargar Metricas (CSV)", csv_data,
                              "metrics.csv", "text/csv")
        with col_e2:
            if st.button("Generar Reporte Completo"):
                from analysis import generate_full_report
                output_dir = RESULTS_DIR / "analysis"
                generate_full_report(str(selected_file), str(output_dir), task)
                st.success(f"Reporte generado en: {output_dir}")


# ============================================================
# PAGE 5: Comparar Modelos
# ============================================================
elif page == "Comparar Modelos":
    st.title("Comparacion de Modelos")

    result_files = sorted(RESULTS_DIR.glob("results_*.csv"))

    if not result_files:
        st.warning("No hay resultados disponibles.")
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

        # Model comparison table
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

        # Best strategy per model
        st.subheader("Mejor Estrategia por Modelo")
        best_strategy = metrics_df.groupby(['model', 'strategy'])['f1'].mean().reset_index()
        best_per_model = best_strategy.loc[best_strategy.groupby('model')['f1'].idxmax()]
        st.dataframe(best_per_model.rename(columns={'f1': 'F1 Mean'}).set_index('model'),
                     use_container_width=True)

        # Local vs API comparison
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

        # Cost/privacy analysis
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

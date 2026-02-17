"""
Modulo de analisis estadistico y visualizacion de resultados.

Genera graficos (barras, boxplots, heatmaps, radar) y analisis estadistico
(ANOVA, t-tests, Cohen's d) para los resultados experimentales.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

# Style
sns.set_theme(style="whitegrid", font_scale=1.1)
FIGSIZE = (12, 7)
DPI = 300
PALETTE = "Set2"


def load_results(results_path: str) -> pd.DataFrame:
    """Carga resultados desde CSV."""
    df = pd.read_csv(results_path)
    logger.info(f"Resultados cargados: {len(df)} filas desde {results_path}")
    return df


def compute_metrics_per_config(df: pd.DataFrame, task: str = "classification") -> pd.DataFrame:
    """Computa metricas por cada configuracion (modelo x estrategia x iteracion)."""
    from metrics import calculate_metrics, calculate_binary_metrics

    strategy_col = 'strategy' if 'strategy' in df.columns else 'pattern'
    rows = []

    for model in df['model'].unique():
        for strategy in df[strategy_col].unique():
            for iteration in df['iteration'].unique():
                subset = df[
                    (df['model'] == model) &
                    (df[strategy_col] == strategy) &
                    (df['iteration'] == iteration)
                ]
                if subset.empty:
                    continue

                if task == "classification":
                    y_true = subset['ground_truth'].tolist()
                    y_pred = subset['prediction'].tolist()
                    metrics = calculate_metrics(y_true, y_pred)
                else:
                    y_true = [str(v).lower() == 'true' for v in subset['ground_truth']]
                    y_pred = [str(v).lower() == 'true' for v in subset['prediction']]
                    metrics = calculate_binary_metrics(y_true, y_pred)

                avg_time = subset['time_seconds'].mean()
                avg_tps = subset.get('tokens_per_second', pd.Series([0])).mean()

                rows.append({
                    'model': model,
                    'strategy': strategy,
                    'iteration': iteration,
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'accuracy': metrics['accuracy'],
                    'invalid_rate': metrics.get('invalid_rate', 0),
                    'avg_time_seconds': avg_time,
                    'avg_tokens_per_second': avg_tps,
                })

    return pd.DataFrame(rows)


# ============================================================
# GRAFICOS
# ============================================================

def plot_grouped_bars(metrics_df: pd.DataFrame, metric: str = "f1",
                      output_path: Optional[str] = None, title: Optional[str] = None):
    """Barras agrupadas: metrica por modelo y estrategia."""
    pivot = metrics_df.groupby(['model', 'strategy'])[metric].mean().unstack()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    pivot.plot(kind='bar', ax=ax, colormap=PALETTE, edgecolor='black', linewidth=0.5)

    ax.set_title(title or f'{metric.upper()} por Modelo y Estrategia', fontsize=14, fontweight='bold')
    ax.set_xlabel('Modelo', fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(title='Estrategia', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
        logger.info(f"Grafico guardado: {output_path}")
    return fig


def plot_boxplots(metrics_df: pd.DataFrame, metric: str = "f1",
                  output_path: Optional[str] = None, title: Optional[str] = None):
    """Boxplots: variabilidad entre iteraciones por modelo."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    sns.boxplot(data=metrics_df, x='model', y=metric, hue='strategy',
                ax=ax, palette=PALETTE)

    ax.set_title(title or f'Variabilidad de {metric.upper()} entre Iteraciones', fontsize=14, fontweight='bold')
    ax.set_xlabel('Modelo', fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(title='Estrategia', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
        logger.info(f"Grafico guardado: {output_path}")
    return fig


def plot_heatmap(metrics_df: pd.DataFrame, metric: str = "f1",
                 output_path: Optional[str] = None, title: Optional[str] = None):
    """Heatmap: modelo x estrategia con metrica promedio."""
    pivot = metrics_df.groupby(['model', 'strategy'])[metric].mean().unstack()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                vmin=0, vmax=1, linewidths=0.5, cbar_kws={'label': metric.upper()})

    ax.set_title(title or f'{metric.upper()} - Modelo x Estrategia', fontsize=14, fontweight='bold')
    ax.set_xlabel('Estrategia', fontsize=12)
    ax.set_ylabel('Modelo', fontsize=12)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
        logger.info(f"Grafico guardado: {output_path}")
    return fig


def plot_radar(metrics_df: pd.DataFrame, models: Optional[List[str]] = None,
               output_path: Optional[str] = None, title: Optional[str] = None):
    """Radar chart: modelos comparados en multiples metricas."""
    metric_cols = ['precision', 'recall', 'f1', 'accuracy']
    model_means = metrics_df.groupby('model')[metric_cols].mean()

    if models:
        model_means = model_means.loc[model_means.index.isin(models)]

    categories = [m.upper() for m in metric_cols]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = sns.color_palette(PALETTE, len(model_means))

    for idx, (model, values) in enumerate(model_means.iterrows()):
        vals = values.tolist()
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', linewidth=2, label=model, color=colors[idx])
        ax.fill(angles, vals, alpha=0.1, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title(title or 'Comparacion de Modelos', fontsize=14, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
        logger.info(f"Grafico guardado: {output_path}")
    return fig


def plot_task_comparison(all_metrics: dict, metric: str = "f1",
                         output_path: Optional[str] = None):
    """Curvas comparativas: rendimiento por tarea para cada modelo."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for task_name, mdf in all_metrics.items():
        model_means = mdf.groupby('model')[metric].mean()
        ax.plot(model_means.index, model_means.values, 'o-', label=task_name, linewidth=2)

    ax.set_title(f'{metric.upper()} por Tarea y Modelo', fontsize=14, fontweight='bold')
    ax.set_xlabel('Modelo', fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(title='Tarea')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
    return fig


def plot_speed_comparison(metrics_df: pd.DataFrame, output_path: Optional[str] = None):
    """Barras: tokens/segundo por modelo."""
    if 'avg_tokens_per_second' not in metrics_df.columns:
        logger.warning("No hay datos de tokens/segundo")
        return None

    model_speed = metrics_df.groupby('model')['avg_tokens_per_second'].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#2ecc71' if 'nim' not in m else '#3498db' for m in model_speed.index]
    model_speed.plot(kind='bar', ax=ax, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_title('Velocidad de Inferencia (tokens/s)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Modelo', fontsize=12)
    ax.set_ylabel('Tokens/segundo', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Add value labels on bars
    for i, v in enumerate(model_speed.values):
        ax.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
    return fig


# ============================================================
# ANALISIS ESTADISTICO
# ============================================================

def anova_two_way(metrics_df: pd.DataFrame, metric: str = "f1") -> dict:
    """ANOVA de 2 factores (modelo x estrategia)."""
    models = metrics_df['model'].unique()
    strategies = metrics_df['strategy'].unique()

    # Group data by model
    model_groups = [metrics_df[metrics_df['model'] == m][metric].values for m in models]
    f_model, p_model = stats.f_oneway(*model_groups)

    # Group data by strategy
    strategy_groups = [metrics_df[metrics_df['strategy'] == s][metric].values for s in strategies]
    f_strategy, p_strategy = stats.f_oneway(*strategy_groups)

    result = {
        'metric': metric,
        'model_effect': {'F': float(f_model), 'p': float(p_model),
                         'significant': p_model < 0.05},
        'strategy_effect': {'F': float(f_strategy), 'p': float(p_strategy),
                            'significant': p_strategy < 0.05},
    }

    logger.info(f"ANOVA ({metric}): Model F={f_model:.3f} p={p_model:.4f}, "
                f"Strategy F={f_strategy:.3f} p={p_strategy:.4f}")
    return result


def paired_ttest_local_vs_api(metrics_df: pd.DataFrame, metric: str = "f1",
                               local_models: Optional[List[str]] = None,
                               api_models: Optional[List[str]] = None) -> dict:
    """T-test pareado: modelos locales vs API."""
    if local_models is None:
        local_models = ['qwen7b', 'llama8b']
    if api_models is None:
        api_models = ['nim_llama70b', 'nim_llama8b', 'nim_mistral']

    local_scores = metrics_df[metrics_df['model'].isin(local_models)][metric].values
    api_scores = metrics_df[metrics_df['model'].isin(api_models)][metric].values

    # Use independent t-test (groups may differ in size)
    t_stat, p_value = stats.ttest_ind(local_scores, api_scores)

    result = {
        'metric': metric,
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'local_mean': float(np.mean(local_scores)),
        'api_mean': float(np.mean(api_scores)),
        'cohens_d': float(cohens_d(local_scores, api_scores)),
    }

    logger.info(f"T-test Local vs API ({metric}): t={t_stat:.3f} p={p_value:.4f} "
                f"d={result['cohens_d']:.3f}")
    return result


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calcula el tamano del efecto Cohen's d."""
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0.0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def full_statistical_analysis(metrics_df: pd.DataFrame) -> dict:
    """Ejecuta analisis estadistico completo."""
    results = {}
    for metric in ['f1', 'accuracy', 'precision', 'recall']:
        results[metric] = {
            'anova': anova_two_way(metrics_df, metric),
            'local_vs_api': paired_ttest_local_vs_api(metrics_df, metric),
        }
    return results


# ============================================================
# TABLAS Y REPORTES
# ============================================================

def generate_summary_table(metrics_df: pd.DataFrame, fmt: str = "markdown") -> str:
    """Genera tabla resumen en Markdown o LaTeX."""
    summary = metrics_df.groupby(['model', 'strategy']).agg({
        'f1': ['mean', 'std'],
        'accuracy': ['mean', 'std'],
        'precision': 'mean',
        'recall': 'mean',
    }).round(3)

    summary.columns = ['F1 Mean', 'F1 Std', 'Acc Mean', 'Acc Std', 'Prec Mean', 'Rec Mean']
    summary = summary.reset_index()

    if fmt == "latex":
        return summary.to_latex(index=False, float_format="%.3f")
    return summary.to_markdown(index=False)


def generate_stats_table(stats_results: dict, fmt: str = "markdown") -> str:
    """Genera tabla de resultados estadisticos."""
    rows = []
    for metric, analysis in stats_results.items():
        anova = analysis['anova']
        ttest = analysis['local_vs_api']
        rows.append({
            'Metric': metric.upper(),
            'ANOVA Model F': f"{anova['model_effect']['F']:.2f}",
            'ANOVA Model p': f"{anova['model_effect']['p']:.4f}",
            'ANOVA Strat F': f"{anova['strategy_effect']['F']:.2f}",
            'ANOVA Strat p': f"{anova['strategy_effect']['p']:.4f}",
            'T-test t': f"{ttest['t_statistic']:.2f}",
            'T-test p': f"{ttest['p_value']:.4f}",
            "Cohen's d": f"{ttest['cohens_d']:.2f}",
            'Local Mean': f"{ttest['local_mean']:.3f}",
            'API Mean': f"{ttest['api_mean']:.3f}",
        })

    table_df = pd.DataFrame(rows)
    if fmt == "latex":
        return table_df.to_latex(index=False)
    return table_df.to_markdown(index=False)


# ============================================================
# GENERACION COMPLETA DE REPORTES
# ============================================================

def generate_full_report(results_path: str, output_dir: str, task: str = "classification"):
    """Genera todos los graficos y tablas para un archivo de resultados."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    df = load_results(results_path)
    metrics_df = compute_metrics_per_config(df, task)

    logger.info(f"Generando reporte completo en {output}")

    # Graficos
    plot_grouped_bars(metrics_df, 'f1', str(output / 'grouped_bars_f1.png'))
    plot_grouped_bars(metrics_df, 'accuracy', str(output / 'grouped_bars_accuracy.png'))
    plot_boxplots(metrics_df, 'f1', str(output / 'boxplot_f1.png'))
    plot_heatmap(metrics_df, 'f1', str(output / 'heatmap_f1.png'))
    plot_heatmap(metrics_df, 'accuracy', str(output / 'heatmap_accuracy.png'))
    plot_radar(metrics_df, output_path=str(output / 'radar_models.png'))
    plot_speed_comparison(metrics_df, str(output / 'speed_comparison.png'))

    # Tablas
    md_table = generate_summary_table(metrics_df, 'markdown')
    latex_table = generate_summary_table(metrics_df, 'latex')

    (output / 'summary_table.md').write_text(md_table)
    (output / 'summary_table.tex').write_text(latex_table)

    # Analisis estadistico
    stat_results = full_statistical_analysis(metrics_df)
    stats_md = generate_stats_table(stat_results, 'markdown')
    stats_latex = generate_stats_table(stat_results, 'latex')

    (output / 'statistics_table.md').write_text(stats_md)
    (output / 'statistics_table.tex').write_text(stats_latex)

    logger.info("Reporte completo generado exitosamente")
    plt.close('all')

    return {
        'metrics_df': metrics_df,
        'stats': stat_results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generar analisis y graficos')
    parser.add_argument('results', help='Ruta al CSV de resultados')
    parser.add_argument('--output', default='../results/analysis', help='Directorio de salida')
    parser.add_argument('--task', default='classification', help='Tipo de tarea')

    args = parser.parse_args()
    generate_full_report(args.results, args.output, args.task)

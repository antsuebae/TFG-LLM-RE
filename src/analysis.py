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
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

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
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

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
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

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
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

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
        local_models = ['qwen7b', 'llama8b', 'gemma9b']
    if api_models is None:
        api_models = ['nim_llama70b', 'nim_llama8b', 'nim_mistral']

    local_scores = metrics_df[metrics_df['model'].isin(local_models)][metric].values
    api_scores = metrics_df[metrics_df['model'].isin(api_models)][metric].values

    # Si falta uno de los grupos no se puede calcular el t-test
    if len(local_scores) < 2 or len(api_scores) < 2:
        logger.warning(f"T-test Local vs API ({metric}): datos insuficientes "
                       f"(local={len(local_scores)}, api={len(api_scores)}) — omitido")
        return {
            'metric': metric,
            't_statistic': float('nan'),
            'p_value': float('nan'),
            'significant': False,
            'local_mean': float(np.mean(local_scores)) if len(local_scores) > 0 else float('nan'),
            'api_mean': float(np.mean(api_scores)) if len(api_scores) > 0 else float('nan'),
            'cohens_d': float('nan'),
        }

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

    # Eficiencia (coste/rendimiento)
    try:
        fig = plot_efficiency_frontier(metrics_df, 'f1', str(output / 'efficiency_f1_vs_speed.png'))
        plt.close(fig)
        fig = plot_f1_vs_time(metrics_df, str(output / 'f1_vs_time.png'))
        plt.close(fig)
        eff_md = generate_efficiency_table(metrics_df, 'markdown')
        eff_latex = generate_efficiency_table(metrics_df, 'latex')
        (output / 'efficiency_table.md').write_text(eff_md)
        (output / 'efficiency_table.tex').write_text(eff_latex)
    except Exception as e:
        logger.warning(f"No se pudo generar analisis de eficiencia: {e}")

    # Analisis de errores
    try:
        generate_error_report(results_path, task, str(output / 'errors'))
    except Exception as e:
        logger.warning(f"No se pudo generar analisis de errores: {e}")

    logger.info("Reporte completo generado exitosamente")
    plt.close('all')

    return {
        'metrics_df': metrics_df,
        'stats': stat_results,
    }


# ============================================================
# ANALISIS CRUZADO ENTRE TAREAS
# ============================================================

def compute_cross_task_ranking(all_metrics: dict, metric: str = "f1") -> pd.DataFrame:
    """Ranking de modelos agregado a traves de todas las tareas.

    Args:
        all_metrics: dict[task_name] -> metrics_df
        metric: Metrica para el ranking (default: f1)

    Returns:
        DataFrame con modelo, media global, std global, y media por tarea.
    """
    rows = []
    for task_name, mdf in all_metrics.items():
        model_means = mdf.groupby('model')[metric].mean()
        for model, value in model_means.items():
            rows.append({'task': task_name, 'model': model, metric: value})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    pivot = df.pivot(index='model', columns='task', values=metric)

    ranking = pd.DataFrame({
        'model': pivot.index,
        f'{metric}_global_mean': pivot.mean(axis=1).values,
        f'{metric}_global_std': pivot.std(axis=1).values,
    })

    # Anadir columna por tarea
    for task in pivot.columns:
        ranking[f'{metric}_{task}'] = pivot[task].values

    ranking = ranking.sort_values(f'{metric}_global_mean', ascending=False).reset_index(drop=True)
    ranking.index = range(1, len(ranking) + 1)
    ranking.index.name = 'rank'

    return ranking


def plot_cross_task_heatmap(all_metrics: dict, metric: str = "f1",
                            output_path: Optional[str] = None) -> plt.Figure:
    """Heatmap: modelo x tarea con metrica promedio."""
    rows = []
    for task_name, mdf in all_metrics.items():
        model_means = mdf.groupby('model')[metric].mean()
        for model, value in model_means.items():
            rows.append({'task': task_name, 'model': model, metric: value})

    df = pd.DataFrame(rows)
    pivot = df.pivot(index='model', columns='task', values=metric)

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 2), max(5, len(pivot) * 0.8)))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                vmin=0, vmax=1, linewidths=0.5, cbar_kws={'label': metric.upper()})

    ax.set_title(f'{metric.upper()} por Modelo y Tarea (media sobre iteraciones)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Tarea', fontsize=12)
    ax.set_ylabel('Modelo', fontsize=12)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
        logger.info(f"Heatmap cruzado guardado: {output_path}")
    return fig


def plot_model_ranking(ranking_df: pd.DataFrame, metric: str = "f1",
                       output_path: Optional[str] = None) -> plt.Figure:
    """Barras horizontales: ranking global de modelos."""
    if ranking_df.empty:
        return plt.figure()

    col_mean = f'{metric}_global_mean'
    col_std = f'{metric}_global_std'

    fig, ax = plt.subplots(figsize=(10, max(4, len(ranking_df) * 0.7)))
    colors = ['#2ecc71' if 'nim' not in m else '#3498db' for m in ranking_df['model']]

    bars = ax.barh(ranking_df['model'], ranking_df[col_mean], xerr=ranking_df[col_std],
                   color=colors, edgecolor='black', linewidth=0.5, capsize=4)

    ax.set_xlabel(f'{metric.upper()} (media global)', fontsize=12)
    ax.set_title(f'Ranking Global de Modelos ({metric.upper()} medio en todas las tareas)',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()

    for bar, val in zip(bars, ranking_df[col_mean]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontweight='bold')

    # Leyenda local vs API
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', label='Local (Ollama)'),
                       Patch(facecolor='#3498db', label='API (NIM)')]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
        logger.info(f"Ranking guardado: {output_path}")
    return fig


# ============================================================
# ANALISIS DE ERRORES
# ============================================================

def analyze_errors(results_path: str, task: str = "classification") -> pd.DataFrame:
    """Identifica requisitos dificiles que fallan consistentemente.

    Returns:
        DataFrame con requirement_id, text, n_models_fail, n_total,
        fail_rate, models_that_fail.
    """
    df = load_results(results_path)
    strategy_col = 'strategy' if 'strategy' in df.columns else 'pattern'

    # Determinar si la prediccion es correcta
    if task == "classification":
        df['correct'] = df['prediction'] == df['ground_truth']
    else:
        df['correct'] = df['prediction'].astype(str).str.lower() == df['ground_truth'].astype(str).str.lower()

    # Agrupar por requisito
    req_col = 'requirement_id' if 'requirement_id' in df.columns else df.columns[0]
    text_col = 'requirement_text' if 'requirement_text' in df.columns else 'requirement_text'

    error_rows = []
    for req_id, group in df.groupby(req_col):
        n_total = len(group)
        n_correct = group['correct'].sum()
        n_fail = n_total - n_correct

        failed_models = group[~group['correct']]['model'].unique().tolist()
        text = group[text_col].iloc[0] if text_col in group.columns else str(req_id)

        error_rows.append({
            'requirement_id': req_id,
            'text': str(text)[:120],
            'n_fail': n_fail,
            'n_total': n_total,
            'fail_rate': n_fail / n_total if n_total > 0 else 0,
            'n_models_fail': len(failed_models),
            'models_that_fail': ', '.join(failed_models),
            'ground_truth': group['ground_truth'].iloc[0],
        })

    error_df = pd.DataFrame(error_rows).sort_values('fail_rate', ascending=False)
    return error_df


def plot_error_distribution(error_df: pd.DataFrame,
                            output_path: Optional[str] = None) -> plt.Figure:
    """Distribucion de dificultad de requisitos."""
    if error_df.empty:
        return plt.figure()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Histograma de fail_rate
    axes[0].hist(error_df['fail_rate'], bins=20, color='#e74c3c', edgecolor='black',
                 alpha=0.7)
    axes[0].set_xlabel('Tasa de fallo', fontsize=12)
    axes[0].set_ylabel('Numero de requisitos', fontsize=12)
    axes[0].set_title('Distribucion de dificultad de requisitos', fontsize=13, fontweight='bold')
    axes[0].axvline(x=0.5, color='orange', linestyle='--', label='50% fallo')
    axes[0].legend()

    # Top 15 mas dificiles
    top = error_df.head(15)
    colors = ['#e74c3c' if r >= 0.8 else '#f39c12' if r >= 0.5 else '#95a5a6'
              for r in top['fail_rate']]
    axes[1].barh(range(len(top)), top['fail_rate'], color=colors, edgecolor='black',
                 linewidth=0.5)
    axes[1].set_yticks(range(len(top)))
    labels = [f"REQ {row['requirement_id']}" for _, row in top.iterrows()]
    axes[1].set_yticklabels(labels, fontsize=9)
    axes[1].set_xlabel('Tasa de fallo', fontsize=12)
    axes[1].set_title('Top 15 requisitos mas dificiles', fontsize=13, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1.05)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
        logger.info(f"Distribucion de errores guardada: {output_path}")
    return fig


def generate_error_report(results_path: str, task: str, output_dir: str):
    """Genera informe completo de analisis de errores."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    error_df = analyze_errors(results_path, task)

    if error_df.empty:
        logger.warning(f"Sin datos de error para {task}")
        return error_df

    # Guardar CSV
    error_df.to_csv(output / 'error_analysis.csv', index=False)

    # Grafico
    fig = plot_error_distribution(error_df, str(output / 'error_distribution.png'))
    plt.close(fig)

    # Resumen
    n_always_fail = len(error_df[error_df['fail_rate'] == 1.0])
    n_always_pass = len(error_df[error_df['fail_rate'] == 0.0])
    n_hard = len(error_df[error_df['fail_rate'] >= 0.5])

    summary = (
        f"# Analisis de Errores - {task}\n\n"
        f"- Requisitos analizados: {len(error_df)}\n"
        f"- Siempre correctos (0% fallo): {n_always_pass}\n"
        f"- Dificiles (>=50% fallo): {n_hard}\n"
        f"- Siempre fallan (100%): {n_always_fail}\n\n"
        f"## Top 10 mas dificiles\n\n"
        f"| REQ ID | Texto | Tasa fallo | Modelos que fallan |\n"
        f"|--------|-------|------------|-------------------|\n"
    )
    for _, row in error_df.head(10).iterrows():
        text = str(row['text'])[:60].replace('|', '\\|')
        summary += f"| {row['requirement_id']} | {text} | {row['fail_rate']:.0%} | {row['models_that_fail']} |\n"

    (output / 'error_analysis.md').write_text(summary)
    logger.info(f"Analisis de errores guardado en {output}")

    return error_df


# ============================================================
# ANALISIS DE COSTE / EFICIENCIA
# ============================================================

def plot_efficiency_frontier(metrics_df: pd.DataFrame, metric: str = "f1",
                             output_path: Optional[str] = None) -> plt.Figure:
    """Scatter: F1 vs tokens/s por modelo. Muestra el trade-off rendimiento vs velocidad."""
    if 'avg_tokens_per_second' not in metrics_df.columns:
        logger.warning("No hay datos de tokens/segundo para grafico de eficiencia")
        return plt.figure()

    model_stats = metrics_df.groupby('model').agg({
        metric: 'mean',
        'avg_tokens_per_second': 'mean',
        'avg_time_seconds': 'mean',
    }).reset_index()

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = ['#2ecc71' if 'nim' not in m else '#3498db' for m in model_stats['model']]
    sizes = [120] * len(model_stats)

    scatter = ax.scatter(model_stats['avg_tokens_per_second'], model_stats[metric],
                         c=colors, s=sizes, edgecolors='black', linewidth=0.8, zorder=5)

    # Etiquetas
    for _, row in model_stats.iterrows():
        ax.annotate(row['model'],
                    (row['avg_tokens_per_second'], row[metric]),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=10, fontweight='bold')

    ax.set_xlabel('Velocidad (tokens/s)', fontsize=12)
    ax.set_ylabel(f'{metric.upper()} (media)', fontsize=12)
    ax.set_title(f'Eficiencia: {metric.upper()} vs Velocidad de Inferencia',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)

    # Leyenda
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', label='Local (Ollama)'),
                       Patch(facecolor='#3498db', label='API (NIM)')]
    ax.legend(handles=legend_elements, loc='lower right')

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
        logger.info(f"Grafico eficiencia guardado: {output_path}")
    return fig


def plot_f1_vs_time(metrics_df: pd.DataFrame, output_path: Optional[str] = None) -> plt.Figure:
    """Scatter: F1 vs tiempo medio por requisito, agrupado por modelo."""
    model_stats = metrics_df.groupby('model').agg({
        'f1': 'mean',
        'avg_time_seconds': 'mean',
    }).reset_index()

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = ['#2ecc71' if 'nim' not in m else '#3498db' for m in model_stats['model']]

    ax.scatter(model_stats['avg_time_seconds'], model_stats['f1'],
               c=colors, s=120, edgecolors='black', linewidth=0.8, zorder=5)

    for _, row in model_stats.iterrows():
        ax.annotate(row['model'],
                    (row['avg_time_seconds'], row['f1']),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=10, fontweight='bold')

    ax.set_xlabel('Tiempo medio por requisito (s)', fontsize=12)
    ax.set_ylabel('F1 (media)', fontsize=12)
    ax.set_title('Trade-off: Rendimiento vs Tiempo de Inferencia',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', label='Local (Ollama)'),
                       Patch(facecolor='#3498db', label='API (NIM)')]
    ax.legend(handles=legend_elements, loc='lower right')

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
        logger.info(f"Grafico F1 vs tiempo guardado: {output_path}")
    return fig


def generate_efficiency_table(metrics_df: pd.DataFrame, fmt: str = "markdown") -> str:
    """Tabla de eficiencia: modelo, F1, tokens/s, tiempo, ratio calidad/coste."""
    model_stats = metrics_df.groupby('model').agg({
        'f1': 'mean',
        'accuracy': 'mean',
        'avg_tokens_per_second': 'mean',
        'avg_time_seconds': 'mean',
    }).round(3).reset_index()

    model_stats.columns = ['Modelo', 'F1', 'Accuracy', 'Tokens/s', 'Tiempo (s)']
    model_stats['Tipo'] = ['Local' if 'nim' not in m else 'API' for m in model_stats['Modelo']]
    model_stats['Eficiencia (F1/s)'] = (model_stats['F1'] / model_stats['Tiempo (s)']).round(3)
    model_stats = model_stats.sort_values('Eficiencia (F1/s)', ascending=False)

    cols = ['Modelo', 'Tipo', 'F1', 'Accuracy', 'Tokens/s', 'Tiempo (s)', 'Eficiencia (F1/s)']
    model_stats = model_stats[cols]

    if fmt == "latex":
        return model_stats.to_latex(index=False, float_format="%.3f")
    return model_stats.to_markdown(index=False)


def generate_all_reports(results_dir: str = "../results/experiments", output_base: str = "../results/analysis"):
    """Descubre todos los CSVs de resultados y genera analisis para cada tarea.

    Busca archivos results_<task>_*.csv, agrupa por tarea y concatena todos
    los CSVs de la misma tarea (local + API pueden estar en ficheros distintos).
    """
    import re as re_mod
    results_path = Path(results_dir)
    output_path = Path(output_base)

    # Descubrir CSVs y agrupar TODOS los ficheros por tarea
    task_files: dict[str, list] = {}
    for csv_file in sorted(results_path.glob("results_*.csv")):
        m = re_mod.match(r'results_(.+)_\d{8}_\d{6}\.csv', csv_file.name)
        if m:
            task = m.group(1)
            task_files.setdefault(task, []).append(csv_file)

    if not task_files:
        logger.error(f"No se encontraron archivos results_*.csv en {results_dir}")
        return

    logger.info(f"Encontrados resultados para {len(task_files)} tareas: {list(task_files.keys())}")

    all_metrics = {}
    for task, csv_files in task_files.items():
        task_output = str(output_path / f"analysis_{task}")

        # Concatenar todos los CSVs de la tarea (local + API en ficheros separados)
        dfs = []
        for f in csv_files:
            try:
                dfs.append(pd.read_csv(f))
            except Exception as e:
                logger.warning(f"  No se pudo leer {f.name}: {e}")
        if not dfs:
            continue
        combined = pd.concat(dfs, ignore_index=True)

        # Eliminar duplicados exactos (mismo modelo+estrategia+iteracion+requisito)
        combined = combined.drop_duplicates()

        n_models = combined['model'].nunique() if 'model' in combined.columns else '?'
        logger.info(f"Generando analisis para '{task}': {len(csv_files)} fichero(s), "
                    f"{len(combined)} filas, {n_models} modelos...")

        # Guardar CSV combinado temporal para reutilizar generate_full_report
        tmp_path = output_path / f"_combined_{task}.csv"
        output_path.mkdir(parents=True, exist_ok=True)
        combined.to_csv(tmp_path, index=False)
        try:
            result = generate_full_report(str(tmp_path), task_output, task)
            all_metrics[task] = result['metrics_df']
            logger.info(f"  -> {task_output}/")
        except Exception as e:
            logger.error(f"  Error en '{task}': {e}")
        finally:
            tmp_path.unlink(missing_ok=True)

    # Analisis cruzado entre tareas si hay mas de una
    if len(all_metrics) > 1:
        cross_dir = output_path / 'cross_task'
        cross_dir.mkdir(parents=True, exist_ok=True)

        try:
            fig = plot_task_comparison(all_metrics, 'f1',
                                       str(cross_dir / 'task_comparison_f1.png'))
            plt.close(fig)
        except Exception as e:
            logger.warning(f"No se pudo generar comparativo de tareas: {e}")

        try:
            fig = plot_cross_task_heatmap(all_metrics, 'f1',
                                          str(cross_dir / 'cross_task_heatmap_f1.png'))
            plt.close(fig)
            fig = plot_cross_task_heatmap(all_metrics, 'accuracy',
                                          str(cross_dir / 'cross_task_heatmap_accuracy.png'))
            plt.close(fig)
        except Exception as e:
            logger.warning(f"No se pudo generar heatmap cruzado: {e}")

        try:
            ranking = compute_cross_task_ranking(all_metrics, 'f1')
            fig = plot_model_ranking(ranking, 'f1',
                                     str(cross_dir / 'model_ranking_f1.png'))
            plt.close(fig)
            ranking.to_csv(cross_dir / 'model_ranking.csv')
            (cross_dir / 'model_ranking.md').write_text(ranking.to_markdown())
            logger.info(f"Ranking global de modelos guardado en {cross_dir}")
        except Exception as e:
            logger.warning(f"No se pudo generar ranking: {e}")

    logger.info(f"Analisis completo. Resultados en {output_path}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generar analisis y graficos')
    parser.add_argument('results', nargs='?', default=None,
                        help='Ruta al CSV de resultados (omitir con --all)')
    parser.add_argument('--output', default='../results/analysis', help='Directorio de salida')
    parser.add_argument('--task', default='classification', help='Tipo de tarea')
    parser.add_argument('--all', action='store_true',
                        help='Analizar todos los CSVs de resultados automaticamente')
    parser.add_argument('--results-dir', default='../results/experiments',
                        help='Directorio donde buscar CSVs (con --all, default: ../results/experiments)')

    args = parser.parse_args()

    if args.all:
        generate_all_reports(results_dir=args.results_dir, output_base=args.output)
    elif args.results:
        generate_full_report(args.results, args.output, args.task)
    else:
        parser.error("Especifica un CSV o usa --all")

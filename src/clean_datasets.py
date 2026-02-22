"""
clean_datasets.py — Limpieza de datasets del TFG
Problemas detectados:
  - promise_nfr_v2.csv : saltos de línea, caracteres de control Windows, textos muy cortos
  - ambiguity_v2.csv   : dashes unicode, caracteres especiales
  - completeness_v2.csv: dashes unicode, ligaduras tipográficas
  - testability_v2.csv : símbolos matemáticos unicode, dashes
  - inconsistency      : sin problemas
"""

import re
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


# ── Utilidades de limpieza ────────────────────────────────────────────────────

def fix_newlines(text: str) -> str:
    """Convierte saltos de línea y tabuladores en espacios simples."""
    if not isinstance(text, str):
        return text
    text = re.sub(r"[\r\n\t]+", " ", text)
    return re.sub(r" {2,}", " ", text).strip()


def fix_control_chars(text: str) -> str:
    """Elimina caracteres de control ASCII (0x00-0x1F, 0x7F) y los
    sustituye por espacio, excepto el espacio normal (0x20)."""
    if not isinstance(text, str):
        return text
    # Comillas inteligentes Windows-1252 → ASCII
    replacements = {
        "\x91": "'", "\x92": "'",
        "\x93": '"', "\x94": '"',
        "\x96": "-", "\x97": "-",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    # Resto de caracteres de control (STX, etc.)
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    return re.sub(r" {2,}", " ", text).strip()


def normalize_unicode(text: str) -> str:
    """Normaliza caracteres unicode habituales en corpus de requisitos."""
    if not isinstance(text, str):
        return text
    replacements = {
        "\u2013": "-",   # en-dash
        "\u2014": "-",   # em-dash
        "\u2018": "'",   # comilla izq
        "\u2019": "'",   # comilla der
        "\u201c": '"',   # comilla doble izq
        "\u201d": '"',   # comilla doble der
        "\u00b1": "+/-", # ±
        "\u00b0": "deg", # °
        "\ufb02": "fl",  # ligadura fl
        "\ufb01": "fi",  # ligadura fi
        "\u2026": "...", # …
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def clean_text(text: str, dataset: str = "") -> str:
    """Aplica todas las transformaciones en orden."""
    text = fix_control_chars(text)
    text = normalize_unicode(text)
    text = fix_newlines(text)
    return text


def flag_short(df: pd.DataFrame, col: str, min_len: int = 20) -> pd.Series:
    return df[col].str.len() < min_len


# ── Limpieza por dataset ──────────────────────────────────────────────────────

def clean_promise(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    text_col = "requirement_text" if "requirement_text" in df.columns else df.columns[1]

    before = df[text_col].copy()
    df[text_col] = df[text_col].apply(lambda t: clean_text(t, "promise"))

    short_mask = flag_short(df, text_col)
    n_short = short_mask.sum()
    n_changed = (df[text_col] != before).sum()

    log.info(f"promise_nfr_v2: {n_changed} textos modificados, {n_short} muy cortos (<20 chars)")
    if n_short:
        log.warning(f"  Textos muy cortos (revisar manualmente):")
        for _, row in df[short_mask].iterrows():
            log.warning(f"    ID={row.get('id', row.name)}: '{row[text_col]}'")

    return df


def clean_ambiguity(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    text_col = "requirement_text" if "requirement_text" in df.columns else df.columns[1]

    before = df[text_col].copy()
    df[text_col] = df[text_col].apply(lambda t: clean_text(t, "ambiguity"))
    n_changed = (df[text_col] != before).sum()
    log.info(f"ambiguity_dataset_v2: {n_changed} textos modificados")
    return df


def clean_completeness(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    text_col = "requirement_text" if "requirement_text" in df.columns else df.columns[1]

    before = df[text_col].copy()
    df[text_col] = df[text_col].apply(lambda t: clean_text(t, "completeness"))
    n_changed = (df[text_col] != before).sum()
    log.info(f"completeness_dataset_v2: {n_changed} textos modificados")
    return df


def clean_testability(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    text_col = "requirement_text" if "requirement_text" in df.columns else df.columns[1]

    before = df[text_col].copy()
    df[text_col] = df[text_col].apply(lambda t: clean_text(t, "testability"))
    n_changed = (df[text_col] != before).sum()
    log.info(f"testability_dataset_v2: {n_changed} textos modificados")
    return df


def clean_inconsistency(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Limpiar ambas columnas de texto
    for col in df.columns:
        if "text" in col.lower() or "req" in col.lower():
            df[col] = df[col].apply(lambda t: clean_text(t, "inconsistency"))
    log.info("inconsistency_dataset: sin cambios necesarios (dataset limpio)")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    files = {
        "promise_nfr_v2.csv":       clean_promise,
        "ambiguity_dataset_v2.csv": clean_ambiguity,
        "completeness_dataset_v2.csv": clean_completeness,
        "testability_dataset_v2.csv": clean_testability,
        "inconsistency_dataset.csv":  clean_inconsistency,
    }

    for filename, cleaner in files.items():
        path = DATA_DIR / filename
        if not path.exists():
            log.warning(f"No encontrado: {path}")
            continue

        df_clean = cleaner(path)
        # Sobreescribe el original (los originales están en git por si acaso)
        df_clean.to_csv(path, index=False)
        log.info(f"  -> Guardado: {path}")

    log.info("\nLimpieza completada. Verifica los cambios con: git diff data/")


if __name__ == "__main__":
    main()

#!/bin/bash
# ============================================================
# TFG - Script de gestion del proyecto
# ============================================================

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
SRC_DIR="$PROJECT_DIR/src"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ── Funciones ────────────────────────────────────────────────

setup() {
    echo -e "${CYAN}Inicializando proyecto...${NC}"

    if [ ! -d "$VENV_DIR" ]; then
        echo -e "${YELLOW}Creando entorno virtual...${NC}"
        python3 -m venv "$VENV_DIR"
    else
        echo -e "${GREEN}Entorno virtual ya existe.${NC}"
    fi

    source "$VENV_DIR/bin/activate"
    echo -e "${YELLOW}Instalando dependencias...${NC}"
    pip install -q -r "$PROJECT_DIR/requirements.txt"
    echo -e "${GREEN}Proyecto listo.${NC}"
}

activate() {
    if [ ! -d "$VENV_DIR" ]; then
        echo -e "${RED}Entorno virtual no encontrado. Ejecuta: ./run.sh setup${NC}"
        exit 1
    fi
    source "$VENV_DIR/bin/activate"
}

streamlit_app() {
    activate
    echo -e "${CYAN}Iniciando Streamlit en http://localhost:8501${NC}"
    cd "$SRC_DIR"
    streamlit run app.py --server.port 8501
}

api() {
    activate
    echo -e "${CYAN}Iniciando API REST en http://localhost:8000${NC}"
    echo -e "${CYAN}Documentacion en http://localhost:8000/docs${NC}"
    cd "$SRC_DIR"
    uvicorn api:app --reload --port 8000
}

experiment() {
    activate
    echo -e "${CYAN}Ejecutando experimento...${NC}"
    PYTHONPATH="$SRC_DIR" python "$SRC_DIR/experiment.py" "$@"
}

analysis() {
    activate
    echo -e "${CYAN}Generando analisis...${NC}"
    PYTHONPATH="$SRC_DIR" python "$SRC_DIR/analysis.py" "$@"
}

pipeline() {
    activate
    echo -e "${CYAN}Ejecutando pipeline completo sobre documento...${NC}"
    PYTHONPATH="$SRC_DIR" python "$SRC_DIR/pipeline.py" "$@"
}

check_ollama() {
    if command -v ollama &> /dev/null; then
        echo -e "${GREEN}Ollama instalado${NC}"
        if ollama list &> /dev/null; then
            echo -e "${GREEN}Ollama corriendo. Modelos disponibles:${NC}"
            ollama list
        else
            echo -e "${YELLOW}Ollama instalado pero no corriendo. Ejecuta: ollama serve${NC}"
        fi
    else
        echo -e "${RED}Ollama no instalado. Visita https://ollama.ai${NC}"
    fi

    echo ""
    if [ -f "$PROJECT_DIR/.env" ]; then
        echo -e "${GREEN}NVIDIA NIM: .env encontrado${NC}"
    else
        echo -e "${YELLOW}NVIDIA NIM: .env no encontrado (modelos API no disponibles)${NC}"
    fi
}

show_help() {
    echo ""
    echo -e "${CYAN}TFG - Optimizacion de LLMs en Ingenieria de Requisitos${NC}"
    echo ""
    echo "Uso: ./run.sh <comando> [opciones]"
    echo ""
    echo -e "${YELLOW}Comandos:${NC}"
    echo "  setup        Crear entorno virtual e instalar dependencias"
    echo "  pipeline     Analizar un documento de requisitos completo"
    echo "  streamlit    Iniciar frontend Streamlit (puerto 8501)"
    echo "  api          Iniciar API REST FastAPI (puerto 8000)"
    echo "  experiment   Ejecutar pipeline de experimentos"
    echo "  analysis     Generar graficos y analisis estadistico"
    echo "  check        Verificar estado de Ollama y NVIDIA NIM"
    echo ""
    echo -e "${YELLOW}Ejemplos:${NC}"
    echo "  ./run.sh setup"
    echo "  ./run.sh pipeline requisitos.txt"
    echo "  ./run.sh pipeline requisitos.csv --model nim_llama70b --strategy chain_of_thought"
    echo "  ./run.sh pipeline requisitos.md --skip-inconsistency"
    echo "  ./run.sh streamlit"
    echo "  ./run.sh api"
    echo "  ./run.sh experiment --dry-run --task classification --models qwen7b"
    echo "  ./run.sh analysis ../results/results_classification_XXXXX.csv"
    echo "  ./run.sh check"
    echo ""
}

# ── Main ─────────────────────────────────────────────────────

case "${1:-help}" in
    setup)      setup ;;
    pipeline)   shift; pipeline "$@" ;;
    streamlit)  streamlit_app ;;
    api)        api ;;
    experiment) shift; experiment "$@" ;;
    analysis)   shift; analysis "$@" ;;
    check)      check_ollama ;;
    help|*)     show_help ;;
esac

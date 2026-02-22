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

    echo ""
    if command -v docker &> /dev/null; then
        echo -e "${GREEN}Docker instalado: $(docker --version)${NC}"
        if docker compose version &> /dev/null; then
            echo -e "${GREEN}Docker Compose disponible${NC}"
        else
            echo -e "${YELLOW}Docker Compose no disponible${NC}"
        fi
        if docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
            echo -e "${GREEN}nvidia-container-toolkit configurado (GPU disponible en Docker)${NC}"
        else
            echo -e "${YELLOW}nvidia-container-toolkit no detectado (los modelos locales no usaran GPU en Docker)${NC}"
        fi
    else
        echo -e "${YELLOW}Docker no instalado (opcional, solo necesario para el modo contenedor)${NC}"
    fi
}

# ── Docker ────────────────────────────────────────────────────

docker_build() {
    echo -e "${CYAN}Construyendo imagen Docker...${NC}"
    docker compose build
    echo -e "${GREEN}Imagen lista.${NC}"
}

docker_up() {
    echo -e "${CYAN}Arrancando servicios Docker (modo CPU)...${NC}"
    docker compose up -d
    echo ""
    echo -e "${GREEN}Servicios activos:${NC}"
    echo -e "  Streamlit → http://localhost:8501"
    echo -e "  API REST  → http://localhost:8000/docs"
    echo -e "  Ollama    → http://localhost:11434"
    echo ""
    echo -e "${YELLOW}Si es la primera vez, descarga los modelos con:${NC}"
    echo -e "  ./run.sh docker-models"
}

install_nvidia_toolkit() {
    if command -v nvidia-ctk &> /dev/null; then
        echo -e "${GREEN}nvidia-container-toolkit ya esta instalado.${NC}"
        return 0
    fi

    echo -e "${YELLOW}nvidia-container-toolkit no encontrado. Instalando...${NC}"

    if [ ! -f /etc/os-release ]; then
        echo -e "${RED}No se pudo detectar la distribucion. Instala nvidia-container-toolkit manualmente:${NC}"
        echo -e "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        exit 1
    fi

    # shellcheck source=/dev/null
    . /etc/os-release

    case "${ID:-}" in
        fedora|rhel|centos|rocky|almalinux)
            echo -e "${CYAN}Detectado: ${PRETTY_NAME} (dnf)${NC}"
            sudo dnf install -y nvidia-container-toolkit
            ;;
        ubuntu|debian|linuxmint|pop)
            echo -e "${CYAN}Detectado: ${PRETTY_NAME} (apt)${NC}"
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
                | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
            curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
                | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
                | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
            sudo apt-get update -qq
            sudo apt-get install -y nvidia-container-toolkit
            ;;
        opensuse*|sles)
            echo -e "${CYAN}Detectado: ${PRETTY_NAME} (zypper)${NC}"
            sudo zypper install -y nvidia-container-toolkit
            ;;
        arch|manjaro|endeavouros|garuda)
            echo -e "${CYAN}Detectado: ${PRETTY_NAME} (AUR)${NC}"
            if command -v yay &> /dev/null; then
                yay -S --noconfirm nvidia-container-toolkit
            elif command -v paru &> /dev/null; then
                paru -S --noconfirm nvidia-container-toolkit
            else
                echo -e "${RED}Instala un helper de AUR (yay/paru) y vuelve a intentarlo.${NC}"
                exit 1
            fi
            ;;
        *)
            echo -e "${RED}Distribucion '${ID}' no reconocida.${NC}"
            echo -e "${YELLOW}Instala nvidia-container-toolkit manualmente:${NC}"
            echo -e "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
            exit 1
            ;;
    esac

    echo -e "${CYAN}Configurando runtime Docker para NVIDIA...${NC}"
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    echo -e "${GREEN}nvidia-container-toolkit instalado y configurado.${NC}"
}

docker_up_gpu() {
    install_nvidia_toolkit
    echo -e "${CYAN}Arrancando servicios Docker (modo GPU)...${NC}"
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
    echo ""
    echo -e "${GREEN}Servicios activos:${NC}"
    echo -e "  Streamlit → http://localhost:8501"
    echo -e "  API REST  → http://localhost:8000/docs"
    echo -e "  Ollama    → http://localhost:11434"
    echo ""
    echo -e "${YELLOW}Si es la primera vez, descarga los modelos con:${NC}"
    echo -e "  ./run.sh docker-models"
}

docker_down() {
    echo -e "${CYAN}Deteniendo servicios Docker...${NC}"
    docker compose down
    echo -e "${GREEN}Servicios detenidos.${NC}"
}

docker_models() {
    echo -e "${CYAN}Descargando modelos de Ollama en el contenedor...${NC}"
    echo -e "${YELLOW}Esto puede tardar varios minutos segun la conexion.${NC}"
    echo ""
    local models=(
        "qwen2.5:7b-instruct-q5_K_M"
        "llama3.1:8b-instruct-q4_K_M"
        "llama3.2:3b-instruct-q4_K_M"
    )
    for model in "${models[@]}"; do
        echo -e "${CYAN}Descargando $model ...${NC}"
        docker exec tfg-ollama ollama pull "$model"
    done
    echo ""
    echo -e "${GREEN}Modelos disponibles en el contenedor:${NC}"
    docker exec tfg-ollama ollama list
}

docker_logs() {
    docker compose logs -f "${2:-}"
}

show_help() {
    echo ""
    echo -e "${CYAN}TFG - Optimizacion de LLMs en Ingenieria de Requisitos${NC}"
    echo ""
    echo "Uso: ./run.sh <comando> [opciones]"
    echo ""
    echo -e "${YELLOW}Ejecucion local (requiere Python + Ollama instalados):${NC}"
    echo "  setup        Crear entorno virtual e instalar dependencias Python"
    echo "  streamlit    Iniciar frontend Streamlit (puerto 8501)"
    echo "  api          Iniciar API REST FastAPI (puerto 8000)"
    echo "  pipeline     Analizar un documento de requisitos completo"
    echo "  experiment   Ejecutar pipeline de experimentos"
    echo "  analysis     Generar graficos y analisis estadistico"
    echo "  check        Verificar estado de Ollama, NVIDIA NIM y Docker"
    echo ""
    echo -e "${YELLOW}Ejecucion en Docker:${NC}"
    echo "  docker-build     Construir la imagen Docker"
    echo "  docker-up        Arrancar todos los servicios en modo CPU"
    echo "  docker-up-gpu    Arrancar servicios con GPU (requiere nvidia-container-toolkit)"
    echo "  docker-down      Detener todos los servicios"
    echo "  docker-models    Descargar modelos de Ollama en el contenedor"
    echo "  docker-logs      Ver logs de los servicios (docker-logs [servicio])"
    echo ""
    echo -e "${YELLOW}  Nota GPU:${NC} docker-up-gpu instala nvidia-container-toolkit automaticamente"
    echo "             si no esta presente (Fedora/RHEL, Ubuntu/Debian, openSUSE, Arch)."
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
    echo "  ./run.sh docker-build && ./run.sh docker-up && ./run.sh docker-models"
    echo "  ./run.sh docker-build && ./run.sh docker-up-gpu && ./run.sh docker-models"
    echo ""
}

# ── Main ─────────────────────────────────────────────────────

case "${1:-help}" in
    setup)         setup ;;
    pipeline)      shift; pipeline "$@" ;;
    streamlit)     streamlit_app ;;
    api)           api ;;
    experiment)    shift; experiment "$@" ;;
    analysis)      shift; analysis "$@" ;;
    check)         check_ollama ;;
    docker-build)   docker_build ;;
    docker-up)      docker_up ;;
    docker-up-gpu)  docker_up_gpu ;;
    docker-down)    docker_down ;;
    docker-models)  docker_models ;;
    docker-logs)    shift; docker_logs "$@" ;;
    help|*)        show_help ;;
esac

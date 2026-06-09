#!/bin/bash
# Script para executar automaticamente o Plano de Experimentos (Testes Paralelos)
# Importante: Como usaremos `--batch`, não precisamos do XVFB visível pois não renderizaremos a UI interativa, 
# mas se precisar gerar os gifs, o ambiente rgb_array exige xvfb-run.

echo "Iniciando bateria de experimentos SUMO-RL MARL..."

cd src/

# Parâmetros gerais
EPS=500
CKPT=100
MAX_JOBS=12  # Aumentado para 12 testes em paralelo (sem o GIF, a RAM não é mais gargalo)

# Limita o PyTorch a 1 thread por processo para evitar que eles briguem pela CPU
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Arquivo temporário para guardar a fila de comandos
rm -f commands.txt

add_test() {
    local map=$1
    local algo=$2
    local lr=$3
    local freq=$4
    local eps=$5
    
    local cmd="xvfb-run -a python train.py --batch --map_name $map --algo $algo --scale 1.0 --episodes $eps --ckpt_freq $CKPT --lr $lr --train_freq $freq"
    
    if [ "$algo" != "Aleatório" ]; then
        cmd="$cmd --use_norm --use_clip"
    fi
    
    echo "$cmd" >> commands.txt
}

# ---------------------------------------------------------
# BLOCO 1: Baselines (Aleatório)
# ---------------------------------------------------------
add_test "nets/2x2grid" "Aleatório" 0.001 1 $EPS
add_test "nets/3x3grid" "Aleatório" 0.001 1 $EPS
add_test "nets/4x4-Lucas" "Aleatório" 0.001 1 $EPS

# ---------------------------------------------------------
# BLOCO 2: Padrão / Instável (LR=1e-3, Freq=1)
# ---------------------------------------------------------
add_test "nets/2x2grid" "IQL" 0.001 1 $EPS
add_test "nets/2x2grid" "VDN" 0.001 1 $EPS
add_test "nets/2x2grid" "QMIX" 0.001 1 $EPS

add_test "nets/3x3grid" "IQL" 0.001 1 $EPS
add_test "nets/3x3grid" "VDN" 0.001 1 $EPS
add_test "nets/3x3grid" "QMIX" 0.001 1 $EPS

# Removemos o 4x4 deste bloco intermediário para poupar tempo (Mapa muito pesado para treino instável)

# ---------------------------------------------------------
# BLOCO 3: Ablação de LR (LR=5e-4, Freq=1)
# ---------------------------------------------------------
add_test "nets/2x2grid" "IQL" 0.0005 1 $EPS
add_test "nets/2x2grid" "VDN" 0.0005 1 $EPS
add_test "nets/2x2grid" "QMIX" 0.0005 1 $EPS

add_test "nets/3x3grid" "IQL" 0.0005 1 $EPS
add_test "nets/3x3grid" "VDN" 0.0005 1 $EPS
add_test "nets/3x3grid" "QMIX" 0.0005 1 $EPS

# ---------------------------------------------------------
# BLOCO 4: Ablação de Freq (LR=1e-3, Freq=10)
# ---------------------------------------------------------
add_test "nets/2x2grid" "IQL" 0.001 10 $EPS
add_test "nets/2x2grid" "VDN" 0.001 10 $EPS
add_test "nets/2x2grid" "QMIX" 0.001 10 $EPS

add_test "nets/3x3grid" "IQL" 0.001 10 $EPS
add_test "nets/3x3grid" "VDN" 0.001 10 $EPS
add_test "nets/3x3grid" "QMIX" 0.001 10 $EPS

# ---------------------------------------------------------
# BLOCO 5: Estabilização Total (LR=5e-4, Freq=10)
# Aqui voltamos com o 4x4 para provar a superioridade da configuração ótima
# ---------------------------------------------------------
add_test "nets/2x2grid" "IQL" 0.0005 10 $EPS
add_test "nets/2x2grid" "VDN" 0.0005 10 $EPS
add_test "nets/2x2grid" "QMIX" 0.0005 10 $EPS

add_test "nets/3x3grid" "IQL" 0.0005 10 $EPS
add_test "nets/3x3grid" "VDN" 0.0005 10 $EPS
add_test "nets/3x3grid" "QMIX" 0.0005 10 $EPS

add_test "nets/4x4-Lucas" "IQL" 0.0005 10 $EPS
add_test "nets/4x4-Lucas" "VDN" 0.0005 10 $EPS
add_test "nets/4x4-Lucas" "QMIX" 0.0005 10 $EPS

# ---------------------------------------------------------
# EXECUÇÃO PARALELA
# ---------------------------------------------------------
total_tests=$(wc -l < commands.txt)
echo "Foram mapeados $total_tests testes."
echo "Iniciando fila de execução com $MAX_JOBS processos paralelos"

# xargs -P dispara processos em paralelo pegando linhas do arquivo
cat commands.txt | xargs -P $MAX_JOBS -I {} bash -c '{}'

echo "BATERIA DE TREINAMENTO PARALELA CONCLUÍDA!"

#!/bin/bash

# Define os melhores modelos para não processar a pasta inteira e estourar a RAM
# Como a pasta models tem muitos treinamentos (e alguns explodiram), vamos focar nos treinamentos estabilizados (Bloco 4/5) ou que chegaram ao fim.

echo "Iniciando a geração de GIFs para os modelos treinados..."

# Garante que o ambiente virtual está ativado
source /home/dvd/venv/bin/activate

for dir in models/*; do
    if [ -d "$dir" ]; then
        # Pula a geração do Aleatório, pois ele não tem arquivos .pt salvos para carregar.
        # (O aleatório pode ser gerado manualmente modificando o script se desejado)
        # if [[ "$dir" == *"Aleatório"* ]]; then
        #     echo "Pulando $dir (Baseline Aleatório não tem pesos salvos)."
        #     continue
        # fi
        
        echo "========================================="
        echo "Processando $dir"
        
        # Chama o script Python para gerar um GIF de 300 passos para não consumir muita RAM
        python src/generate_gif.py --dir "$dir" --length 300
        
        echo "Aguardando 2 segundos para liberar memória RAM..."
        sleep 2
    fi
done

echo "Geração de GIFs concluída!"

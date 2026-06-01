# SUMO-RL MARL Project

Este projeto implementa algoritmos de *Multi-Agent Reinforcement Learning* (MARL) usando o simulador de tráfego SUMO via biblioteca `sumo-rl` e API PettingZoo.

## Estrutura do Projeto

- `src/train.py`: Loop de treinamento principal usando a API PettingZoo e PyTorch. TUI interativa, suporte a múltiplos gráficos ao vivo e checkpoints.
- `src/agents.py`: Implementa os treinadores unificados (Agente Aleatório, IQL, VDN e QMIX) e o Replay Buffer global com Normalização e Gradient Clipping.
- `src/networks.py`: Contém as arquiteturas de Redes Neurais Base e Mixers (Hypernetworks).
- `src/env_setup.py`: Configura as variações de ambientes SUMO, injeta `--delay`/`--scale` e lida com renderizações virtuais/nativas.
- `src/evaluate.py`: Script para avaliar modelos previamente salvos no diretório `models/`.
- `models/`: Diretório gerado automaticamente que armazena os pesos dos treinamentos (`.pt`), metadados (`params.txt`), métricas em CSV e os vídeos avaliativos (`.gif`).
- `THEORY.md`: Fundamentação teórica (com diagramas) sobre os algoritmos MARL e o problema de semáforos.
- `HYPOTHESES.md`: Documentação listando os resultados esperados, tempos de convergência e características de cada algoritmo nos variados mapas.
- `requirements.txt`: Dependências Python para execução do projeto.

## Como Executar

### Dependências de Sistema (Linux)
Se você estiver rodando o treinamento em um servidor ou em um ambiente sem interface gráfica (para gerar os vídeos/GIFs), o `Xvfb` (X virtual framebuffer) é obrigatório para renderizar a janela virtualmente:
```bash
sudo apt-get update
sudo apt-get install xvfb
```

### Dependências Python
Certifique-se de estar no ambiente virtual configurado com as dependências instaladas (incluindo `eclipse-sumo`). As dependências Python podem ser instaladas via `requirements.txt`:

```bash
source /your/venv_path/venv/bin/activate
cd /your/path/sumo_rl_project
pip install -r requirements.txt
```

### Treinamento Silencioso (Headless)
Execute o `train.py` para abrir o menu interativo (TUI):
```bash
python train.py
```
Você poderá configurar:
- Mapa/Rede (2x2, 3x3, 4x4, cologne8)
- Algoritmo (Aleatório, IQL, VDN, QMIX)
- Renderização (SUMO-GUI, Delay, Logs)
- Estabilizadores (Normalização de Recompensa e Gradient Clipping)
- Duração dos vídeos de avaliação (GIFs) gerados ao longo do treinamento

O script exibirá gráficos interativos via Matplotlib com acompanhamento em tempo real do Throughput (carros completados), Espera Total, Recompensa e Loss. No fim, a imagem do gráfico será salva como `training_plot.png`.

**Nota sobre VDN e QMIX**: A base das redes de Mixer já está em `networks.py`. A arquitetura usa CTDE (Centralized Training with Decentralized Execution) centralizando o gradiente na fase de treino e descentralizando a execução das ações.

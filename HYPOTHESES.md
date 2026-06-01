# Hipóteses e Resultados Esperados - SUMO-RL MARL

Neste documento, listamos as características esperadas, o tempo médio de convergência e as recompensas típicas para os algoritmos implementados ao treinar no ambiente `2x2grid` (e suas variações).

## 1. Ambiente: Redes de Tráfego
- **2x2grid (Padrão)**: 4 semáforos, tráfego médio. Fácil convergência.
- **3x3grid**: 9 semáforos, tráfego moderado a pesado. Maior desafio de coordenação.
- **4x4grid**: 16 semáforos. Exige algoritmos centralizados (QMIX/VDN) para boa fluidez global.
- **cologne8**: Mapa real de Colônia (Alemanha), tráfego não-uniforme. Demanda políticas altamente generalizáveis.

---

## 2. IQL (Independent Q-Learning)
O IQL trata cada semáforo como um agente isolado. Eles não se comunicam.

- **Hipótese**: O IQL deve aprender rapidamente políticas gananciosas locais (ex: deixar o sinal verde para a rua com mais carros no momento). No entanto, como os outros semáforos também estão aprendendo, o ambiente parece "instável" (não-estacionário).
- **Tempo de Treinamento**: Rápido (em termos de computação, pois as redes são minúsculas).
- **Convergência**: Converge rapidamente (10 a 20 episódios no `2x2grid`), mas o limite máximo de recompensa é baixo.
- **Resultados Típicos (2x2grid)**:
  - Passos por episódio: 500-700
  - Recompensa acumulada: Estaciona em valores moderados negativos (pois a recompensa penaliza a fila), sofrendo variações abruptas devido à instabilidade do MARL independente.

---

## 3. VDN (Value Decomposition Networks)
VDN assume que o valor global de um estado é exatamente a soma dos valores locais.

- **Hipótese**: VDN deve superar o IQL facilmente. Ao treinar a rede usando a recompensa total do mapa (CTDE), os agentes aprendem a "se sacrificar" (ex: manter um sinal fechado mais tempo do que o ideal localmente) para não entupir o cruzamento seguinte.
- **Tempo de Treinamento**: Médio. Exige que a `Loss` calcule a propagação do gradiente através da soma.
- **Convergência**: Demora um pouco mais que o IQL (20 a 30 episódios), pois precisa entender o impacto global de ações locais.
- **Resultados Típicos (2x2grid)**:
  - Passos por episódio: Menor que o IQL (carros saem mais rápido do mapa).
  - Recompensa acumulada: Notavelmente maior e mais estável que o IQL após convergência.

---

## 4. QMIX
QMIX utiliza uma Hypernetwork para criar pesos dinâmicos, permitindo que a relação entre as Q-Values locais e a Global seja não-linear.

- **Hipótese**: QMIX deve apresentar a melhor política final. Ele é capaz de captar nuances complexas, como: "Se o cruzamento A está engarrafado e o B está vazio, a importância da ação do cruzamento A é exponencialmente maior agora".
- **Tempo de Treinamento**: Alto. A propagação do gradiente passa pela Hypernetwork, que é uma rede neural que gera pesos para outra rede. Pode consumir mais CPU/GPU.
- **Convergência**: Lenta (40 a 60 episódios). O `Epsilon` pode precisar decair mais devagar para o QMIX explorar melhor.
- **Resultados Típicos (2x2grid e 4x4grid)**:
  - Passos por episódio: Menor tempo de espera global (menor duração simulada se todos os carros chegarem ao destino).
  - Recompensa acumulada: Curva de aprendizado demorada, mas o topo de recompensa alcançado é o maior e mais estável entre os 3 algoritmos, lidando muito bem com mapas maiores (`4x4grid`).

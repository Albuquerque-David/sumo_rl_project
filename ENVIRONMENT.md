# Ambiente de Tráfego: sumo-rl

Este documento destrincha a natureza da simulação, os agentes envolvidos e o que esperamos que a IA aprenda.

## 🗺️ O Mapa Base: 2x2 Grid
O projeto roda em um mapa de malha 2x2 padrão (`nets/2x2grid`).
Isso significa que existem **4 cruzamentos principais**. Ruas chegam do Norte, Sul, Leste e Oeste conectando essas interseções. O SUMO gera rotas dinâmicas de carros (demandas) tentando fluir pela cidade.

## 🤖 Os Agentes
Nesse paradigma MARL (Multi-Agent Reinforcement Learning), **cada semáforo é um agente independente**.
Portanto, temos 4 Agentes neste cenário. Eles não "conversam" por rádio, eles apenas tomam ações de forma isolada com base no tráfego imediato que conseguem enxergar.

### Observações (O que o Agente "Vê")
A cada *step*, o agente recebe um array de números que representa:
- Densidade de carros aproximando-se das suas vias de entrada (fração da rua ocupada).
- Tamanho das filas paradas no sinal vermelho.
- Qual fase do semáforo está atualmente ativa.

### Ações (O que o Agente "Faz")
O espaço de ação é discreto (`Discrete`). O agente pode acionar botões que mudam a fase do semáforo:
- **0**: Fica verde para a rota Norte-Sul.
- **1**: Fica verde para a rota Leste-Oeste.
*(E assim sucessivamente caso existam rotas de conversão exclusivas).*

Sempre que a fase é alterada de 0 para 1, por exemplo, o SUMO embute automaticamente os segundos em amarelo de transição para evitar batidas físicas (embora no RL isso se resuma ao fluxo sendo temporariamente detido).

## 🎯 Objetivo e Recompensa
Nossa métrica/recompensa local do `sumo-rl` está definida como `diff-waiting-time` (Diferença do Tempo de Espera).
- **Recompensa**: $\text{Espera Anterior} - \text{Espera Atual}$.
- **Objetivo**: Evitar que carros fiquem muito tempo ociosos nas suas vias, reduzindo congestionamento.
- **Desejável**: O modelo perfeito percebe "ondas verdes". O cruzamento 1 fica verde a tempo do grande fluxo que acabou de ser liberado no cruzamento 2 passar sem parar de novo, alcançando cooperatividade emergente, essencial para QMIX e VDN brilhar em comparação ao IQL puro.

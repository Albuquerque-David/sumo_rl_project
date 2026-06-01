import argparse
import numpy as np
import torch
import os
from env_setup import create_env
from agents import MARLTrainer

def evaluate(model_path, map_name, algo, episodes=1, render=True):
    print(f"Iniciando avaliação do modelo {model_path} no mapa {map_name}...")
    env = create_env(net_name=map_name, render_mode='human' if render else None, show_logs=True)
    
    agents_list = env.possible_agents
    obs_dim = env.observation_space(agents_list[0]).shape[0]
    action_dim = env.action_space(agents_list[0]).n
    
    trainer = MARLTrainer(agents_list, obs_dim, action_dim, algo=algo)
    trainer.load(model_path)
    print("Pesos carregados com sucesso. Executando...")
    
    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        
        while env.agents:
            actions = trainer.get_actions(obs, epsilon=0.0) 
            next_obs, rewards, term, trunc, infos = env.step(actions)
            
            for agent in env.agents:
                if agent in rewards:
                    total_reward += rewards[agent]
            obs = next_obs
            steps += 1
            
        print(f"Avaliação - Episódio {ep+1} | Passos: {steps} | Recompensa Acumulada: {total_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Caminho para o arquivo model_ep_X.pt salvo.')
    parser.add_argument('--map_name', type=str, default='nets/2x2grid', help='Nome do mapa (ex: nets/2x2grid, nets/3x3grid)')
    parser.add_argument('--algo', type=str, required=True, choices=['Aleatório', 'IQL', 'VDN', 'QMIX'], help='Algoritmo associado ao modelo salvo.')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--no-render', action='store_true')
    args = parser.parse_args()
    
    evaluate(args.model, args.map_name, args.algo, episodes=args.episodes, render=not args.no_render)

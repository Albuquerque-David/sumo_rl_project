import argparse
import os
import imageio
import torch
from env_setup import create_env
from agents import MARLTrainer

def generate_gif(model_dir, gif_length=500):
    print(f"Gerando GIF para o modelo em {model_dir}...")
    
    # Descobrir map_name e algo pelo nome do diretório
    dir_name = os.path.basename(model_dir)
    parts = dir_name.split('_')
    if len(parts) < 3:
        print("Nome de diretório inválido.")
        return
    
    algo = parts[1]
    map_name = parts[2]
    
    # Encontrar o último modelo salvo no diretório
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    if not model_files:
        print(f"Nenhum modelo .pt encontrado em {model_dir}")
        return
        
    # Pega o modelo com o maior episódio
    model_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
    best_model_path = os.path.join(model_dir, model_files[-1])
    
    try:
        env = create_env(net_name=map_name, render_mode='rgb_array', num_seconds=gif_length, show_logs=False)
        obs, info = env.reset()
        frames = []
        
        agents_list = env.possible_agents
        obs_dim = env.observation_space(agents_list[0]).shape[0]
        action_dim = env.action_space(agents_list[0]).n
        
        trainer = MARLTrainer(agents_list, obs_dim, action_dim, algo=algo)
        trainer.load(best_model_path)
        
        print("Modelo carregado. Simulando e capturando frames...")
        steps = 0
        while env.agents and steps < gif_length:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            actions = trainer.get_actions(obs, epsilon=0.0) # Greedy
            obs, rewards, term, trunc, infos = env.step(actions)
            steps += 1
            
        env.close()
        
        if frames:
            gif_path = os.path.join(model_dir, f"resultado_{algo}_{map_name}.gif")
            imageio.mimsave(gif_path, frames, fps=10)
            print(f"GIF salvo com sucesso em: {gif_path}")
        else:
            print("Nenhum frame foi capturado.")
            
    except Exception as e:
        print(f"Erro ao gerar GIF para {model_dir}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='Caminho do diretório do modelo')
    parser.add_argument('--length', type=int, default=300, help='Tamanho da simulação em passos')
    args = parser.parse_args()
    generate_gif(args.dir, args.length)

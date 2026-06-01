import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import inquirer
from rich.console import Console
from rich.table import Table
from rich.live import Live
import datetime
import csv
import imageio

from env_setup import create_env
from agents import MARLTrainer, GlobalReplayBuffer

def show_tui():
    questions = [
        inquirer.List('map_name', message="Mapa/Rede", choices=['nets/2x2grid', 'nets/3x3grid', 'nets/4x4grid', 'nets/cologne8'], default='nets/2x2grid'),
        inquirer.List('algo', message="Algoritmo", choices=['Aleatório', 'IQL', 'VDN', 'QMIX'], default='IQL'),
        inquirer.Confirm('use_norm', message="Usar Normalização de Recompensa?", default=False),
        inquirer.Text('norm_factor', message="Fator de Normalização (divisor)", default='100.0'),
        inquirer.Confirm('use_clip', message="Usar Gradient Clipping?", default=False),
        inquirer.Confirm('render', message="Renderizar SUMO-GUI?", default=False),
        inquirer.Confirm('show_logs', message="Mostrar logs do SUMO no console?", default=False),
        inquirer.Text('delay', message="Delay na simulação (ms)", default='0'),
        inquirer.Text('scale', message="Escala de Tráfego", default='1.0'),
        inquirer.Text('episodes', message="Episódios", default='20'),
        inquirer.Text('ckpt_freq', message="Frequência de Checkpoint (salvar modelo)", default='5'),
        inquirer.Text('gif_length', message="Duração do vídeo a cada checkpoint (sim-seconds)", default='500')
    ]
    ans = inquirer.prompt(questions)
    if not ans: exit(0)
    return ans['map_name'], ans['algo'], ans['use_norm'], float(ans['norm_factor']), ans['use_clip'], ans['render'], ans['show_logs'], int(ans['delay']), float(ans['scale']), int(ans['episodes']), int(ans['ckpt_freq']), int(ans['gif_length'])

def render_checkpoint_video(trainer, ep, model_dir, scale, gif_length):
    try:
        # Cria uma nova instância oculta focada apenas em renderizar os frames para o GIF
        env = create_env(render_mode='rgb_array', num_seconds=gif_length, show_logs=False, delay=0, scale=scale)
        obs, info = env.reset()
        frames = []
        
        while env.agents:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            actions = trainer.get_actions(obs, epsilon=0.0) # Política Determinística (Greedy)
            obs, rewards, term, trunc, infos = env.step(actions)
            
        env.close()
        
        if frames:
            video_path = os.path.join(model_dir, f"checkpoint_ep_{ep}.gif")
            imageio.mimsave(video_path, frames, fps=10)
    except Exception as e:
        print(f"Erro ao salvar vídeo: {e}")

def train(map_name, algo, use_norm, norm_factor, use_clip, render, show_logs, delay, scale, episodes, ckpt_freq, gif_length, max_steps=500):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.abspath(os.path.join("..", "models", f"{timestamp}_{algo}_{map_name.replace('nets/', '')}"))
    os.makedirs(model_dir, exist_ok=True)
    
    # Salvar parametros base
    with open(os.path.join(model_dir, "params.txt"), "w") as f:
        f.write(f"Map: {map_name}\nAlgorithm: {algo}\nUse Norm: {use_norm}\nNorm Factor: {norm_factor}\nUse Clip: {use_clip}\nEpisodes: {episodes}\nDelay: {delay}\nScale: {scale}\nCheckpoint Freq: {ckpt_freq}\nGIF Length: {gif_length}\n")
    
    metrics_path = os.path.join(model_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "GlobalReward", "AccReward", "Epsilon", "AvgLoss", "Throughput", "WaitTime"])
        
    console = Console()
    console.print(f"\n[bold green]🚀 Treinando {algo} em {map_name}[/bold green] | Pasta: {model_dir}")
    
    env = create_env(net_name=map_name, render_mode='human' if render else None, show_logs=show_logs, delay=delay, scale=scale)
    obs, info = env.reset()
    
    agents_list = env.possible_agents
    obs_dim = env.observation_space(agents_list[0]).shape[0]
    action_dim = env.action_space(agents_list[0]).n
    
    trainer = MARLTrainer(agents_list, obs_dim, action_dim, algo=algo, use_norm=use_norm, norm_factor=norm_factor, use_clip=use_clip)
    buffer = GlobalReplayBuffer(capacity=10000)
    
    if render:
        plt.ion()
        fig, axs = plt.subplots(3, 2, figsize=(12, 10))
        fig.suptitle(f'Treinamento MARL: {algo}')
        
        line_rew, = axs[0, 0].plot([], [], 'b-')
        axs[0, 0].set_title('Recompensa Global do Episódio')
        
        line_acc_rew, = axs[0, 1].plot([], [], 'c-')
        axs[0, 1].set_title('Recompensa Acumulada (Total)')
        
        line_loss, = axs[1, 0].plot([], [], 'r-')
        axs[1, 0].set_title('Loss Média')
        
        line_eps, = axs[1, 1].plot([], [], 'g-')
        axs[1, 1].set_title('Epsilon')
        
        line_thr, = axs[2, 0].plot([], [], 'm-')
        axs[2, 0].set_title('Veículos Completados (Throughput)')
        
        line_wait, = axs[2, 1].plot([], [], 'y-')
        axs[2, 1].set_title('Tempo de Espera Total')
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)

    rewards_hist, acc_rew_hist, loss_hist, eps_hist, thr_hist, wait_hist = [], [], [], [], [], []
    cumulative_reward = 0

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Episódio", justify="right")
    table.add_column("Epsilon", justify="right")
    table.add_column("Rec. Ep.", justify="right")
    table.add_column("Rec. Acum.", justify="right")
    table.add_column("Loss", justify="right")
    table.add_column("Throughput", justify="right")

    with Live(table, refresh_per_second=4, console=console):
        for ep in range(1, episodes + 1):
            obs, info = env.reset()
            episode_reward = 0
            episode_losses = []
            episode_throughput = 0
            episode_waiting_time = 0
            
            epsilon = max(0.05, 1.0 - (ep / (episodes * 0.8)))
            
            step_count = 0
            for step in range(max_steps):
                if not env.agents:
                    break
                    
                actions = trainer.get_actions(obs, epsilon=epsilon)
                next_obs, rewards, terminations, truncations, infos = env.step(actions)
                
                try:
                    episode_throughput += env.unwrapped.env.sumo.simulation.getArrivedNumber()
                except Exception:
                    pass
                
                if agents_list and agents_list[0] in infos:
                    episode_waiting_time += infos[agents_list[0]].get('system_total_waiting_time', 0.0)
                
                dones = {ag: terminations.get(ag, True) or truncations.get(ag, True) for ag in agents_list}
                padded_obs = {ag: obs.get(ag, np.zeros(obs_dim)) for ag in agents_list}
                padded_next_obs = {ag: next_obs.get(ag, np.zeros(obs_dim)) for ag in agents_list}
                padded_actions = {ag: actions.get(ag, 0) for ag in agents_list}
                padded_rewards = {ag: rewards.get(ag, 0.0) for ag in agents_list}
                
                buffer.push({
                    'obs': padded_obs, 'actions': padded_actions, 'rewards': padded_rewards,
                    'next_obs': padded_next_obs, 'dones': dones
                })
                
                episode_reward += sum(rewards.values())
                
                if len(buffer) > 64:
                    loss = trainer.update(buffer.sample(64))
                    episode_losses.append(loss)
                
                obs = next_obs
                step_count += 1
                
            trainer.update_target()
            avg_loss = np.mean(episode_losses) if episode_losses else 0.0
            
            cumulative_reward += episode_reward
            rewards_hist.append(episode_reward)
            acc_rew_hist.append(cumulative_reward)
            loss_hist.append(avg_loss)
            eps_hist.append(epsilon)
            thr_hist.append(episode_throughput)
            wait_hist.append(episode_waiting_time)
            
            table.add_row(f"{ep}/{episodes}", f"{epsilon:.2f}", f"{episode_reward:.2f}", f"{cumulative_reward:.2f}", f"{avg_loss:.4f}", f"{episode_throughput}")
            
            with open(metrics_path, "a", newline="") as f:
                csv.writer(f).writerow([ep, episode_reward, cumulative_reward, epsilon, avg_loss, episode_throughput, episode_waiting_time])
            
            if render:
                line_rew.set_data(range(1, ep+1), rewards_hist)
                axs[0,0].relim(); axs[0,0].autoscale_view()
                
                line_acc_rew.set_data(range(1, ep+1), acc_rew_hist)
                axs[0,1].relim(); axs[0,1].autoscale_view()
                
                line_loss.set_data(range(1, ep+1), loss_hist)
                axs[1,0].relim(); axs[1,0].autoscale_view()
                
                line_eps.set_data(range(1, ep+1), eps_hist)
                axs[1,1].relim(); axs[1,1].autoscale_view()
                
                line_thr.set_data(range(1, ep+1), thr_hist)
                axs[2,0].relim(); axs[2,0].autoscale_view()
                
                line_wait.set_data(range(1, ep+1), wait_hist)
                axs[2,1].relim(); axs[2,1].autoscale_view()
                
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.01)

            if ep % ckpt_freq == 0:
                trainer.save(os.path.join(model_dir, f"model_ep_{ep}.pt"))
                render_checkpoint_video(trainer, ep, model_dir, scale, gif_length)

    env.close()
    
    # Salvar o plot ao final
    if render:
        fig.savefig(os.path.join(model_dir, "training_plot.png"))
        plt.close('all')
    else:
        fig, axs = plt.subplots(3, 2, figsize=(12, 10))
        fig.suptitle(f'Treinamento MARL: {algo}')
        axs[0, 0].plot(range(1, episodes+1), rewards_hist, 'b-')
        axs[0, 0].set_title('Recompensa Global do Episódio')
        axs[0, 1].plot(range(1, episodes+1), acc_rew_hist, 'c-')
        axs[0, 1].set_title('Recompensa Acumulada (Total)')
        axs[1, 0].plot(range(1, episodes+1), loss_hist, 'r-')
        axs[1, 0].set_title('Loss Média')
        axs[1, 1].plot(range(1, episodes+1), eps_hist, 'g-')
        axs[1, 1].set_title('Epsilon')
        axs[2, 0].plot(range(1, episodes+1), thr_hist, 'm-')
        axs[2, 0].set_title('Veículos Completados (Throughput)')
        axs[2, 1].plot(range(1, episodes+1), wait_hist, 'y-')
        axs[2, 1].set_title('Tempo de Espera Total')
        plt.tight_layout()
        fig.savefig(os.path.join(model_dir, "training_plot.png"))
        plt.close(fig)
        
if __name__ == "__main__":
    map_name, algo, use_norm, norm_factor, use_clip, render, show_logs, delay, scale, episodes, ckpt_freq, gif_length = show_tui()
    train(map_name, algo, use_norm, norm_factor, use_clip, render, show_logs, delay, scale, episodes, ckpt_freq, gif_length)


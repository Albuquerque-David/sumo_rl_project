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
        inquirer.List('map_name', message="Mapa/Rede", choices=['nets/2x2grid', 'nets/3x3grid', 'nets/4x4-Lucas', 'nets/cologne8'], default='nets/2x2grid'),
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
        inquirer.Text('gif_length', message="Duração do vídeo a cada checkpoint (sim-seconds)", default='500'),
        inquirer.Text('lr', message="Taxa de Aprendizado (LR)", default='0.001'),
        inquirer.Text('train_freq', message="Frequência de Treino (em passos)", default='1')
    ]
    ans = inquirer.prompt(questions)
    if not ans: exit(0)
    return ans['map_name'], ans['algo'], ans['use_norm'], float(ans['norm_factor']), ans['use_clip'], ans['render'], ans['show_logs'], int(ans['delay']), float(ans['scale']), int(ans['episodes']), int(ans['ckpt_freq']), int(ans['gif_length']), float(ans['lr']), int(ans['train_freq'])

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

def train(map_name, algo, use_norm, norm_factor, use_clip, render, show_logs, delay, scale, episodes, ckpt_freq, gif_length, lr=0.001, train_freq=1):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.abspath(os.path.join("..", "models", f"{timestamp}_{algo}_{map_name.replace('nets/', '')}"))
    os.makedirs(model_dir, exist_ok=True)
    
    # Salvar parametros base
    with open(os.path.join(model_dir, "params.txt"), "w") as f:
        f.write(f"Map: {map_name}\nAlgorithm: {algo}\nUse Norm: {use_norm}\nNorm Factor: {norm_factor}\nUse Clip: {use_clip}\nLR: {lr}\nTrain Freq: {train_freq}\nEpisodes: {episodes}\nDelay: {delay}\nScale: {scale}\nCheckpoint Freq: {ckpt_freq}\nGIF Length: {gif_length}\n")
    
    metrics_path = os.path.join(model_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Load env early to get action_dim for the CSV header
        temp_env = create_env(net_name=map_name, render_mode=None, show_logs=False, delay=0, scale=1.0)
        action_dim = temp_env.action_space(temp_env.possible_agents[0]).n
        temp_env.close()
        
        headers = ["Episode", "GlobalReward", "AccReward", "Epsilon", "AvgLoss", "Throughput", "WaitTime", "GreenMin", "GreenMax", "GreenAvg"]
        headers.extend([f"Act{a}" for a in range(action_dim)])
        writer.writerow(headers)
        
    console = Console()
    console.print(f"\n[bold green]🚀 Treinando {algo} em {map_name}[/bold green] | Pasta: {model_dir}")
    
    env = create_env(net_name=map_name, render_mode='human' if render else None, show_logs=show_logs, delay=delay, scale=scale)
    obs, info = env.reset()
    
    agents_list = env.possible_agents
    obs_dim = env.observation_space(agents_list[0]).shape[0]
    action_dim = env.action_space(agents_list[0]).n
    
    trainer = MARLTrainer(agents_list, obs_dim, action_dim, algo=algo, lr=lr, use_norm=use_norm, norm_factor=norm_factor, use_clip=use_clip)
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
    green_min_hist, green_max_hist, green_avg_hist = [], [], []
    act_freqs_hist = {a: [] for a in range(action_dim)}
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
            
            episode_green_durations = []
            action_counts = {a: 0 for a in range(action_dim)}
            last_phase = {ag: env.unwrapped.env.traffic_signals[ag].green_phase for ag in agents_list}
            
            epsilon = max(0.05, 1.0 - (ep / (episodes * 1.2)))
            
            step_count = 0
            while env.agents:
                    
                actions = trainer.get_actions(obs, epsilon=epsilon)
                
                for ag, act in actions.items():
                    action_counts[act] += 1
                    ts = env.unwrapped.env.traffic_signals[ag]
                    if act != last_phase[ag] and not ts.is_yellow:
                        episode_green_durations.append(ts.time_since_last_phase_change)
                    last_phase[ag] = act
                
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
                
                if len(buffer) > 64 and step_count % train_freq == 0:
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
            
            g_min = np.min(episode_green_durations) if episode_green_durations else 0.0
            g_max = np.max(episode_green_durations) if episode_green_durations else 0.0
            g_avg = np.mean(episode_green_durations) if episode_green_durations else 0.0
            
            green_min_hist.append(g_min)
            green_max_hist.append(g_max)
            green_avg_hist.append(g_avg)
            
            total_acts = sum(action_counts.values()) or 1
            act_freqs = {a: action_counts[a] / total_acts for a in range(action_dim)}
            for a in range(action_dim):
                act_freqs_hist[a].append(act_freqs[a])
            
            table.add_row(f"{ep}/{episodes}", f"{epsilon:.2f}", f"{episode_reward:.2f}", f"{cumulative_reward:.2f}", f"{avg_loss:.4f}", f"{episode_throughput}")
            
            with open(metrics_path, "a", newline="") as f:
                row = [ep, episode_reward, cumulative_reward, epsilon, avg_loss, episode_throughput, episode_waiting_time, g_min, g_max, g_avg]
                row.extend([act_freqs[a] for a in range(action_dim)])
                csv.writer(f).writerow(row)
            
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
                # render_checkpoint_video(trainer, ep, model_dir, scale, gif_length) # Disabled to prevent memory crashes

    env.close()
    
    # Salvar o plot ao final (4x2 para incluir as novas métricas)
    plt.close('all')
    fig, axs = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle(f'Treinamento MARL: {algo} em {map_name}')
    
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
    
    axs[3, 0].plot(range(1, episodes+1), green_avg_hist, 'k-')
    axs[3, 0].set_title('Duração Média do Verde (s)')
    
    bottom = np.zeros(episodes)
    for a in range(action_dim):
        axs[3, 1].fill_between(range(1, episodes+1), bottom, bottom + act_freqs_hist[a], label=f'Act {a}', alpha=0.7)
        bottom += act_freqs_hist[a]
    axs[3, 1].set_title('Distribuição de Ações (Frequência)')
    axs[3, 1].legend(loc='upper right')
    
    plt.tight_layout()
    fig.savefig(os.path.join(model_dir, "training_plot.png"))
    plt.close(fig)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_name', type=str, default='nets/2x2grid')
    parser.add_argument('--algo', type=str, default='IQL')
    parser.add_argument('--use_norm', action='store_true')
    parser.add_argument('--norm_factor', type=float, default=100.0)
    parser.add_argument('--use_clip', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--show_logs', action='store_true')
    parser.add_argument('--delay', type=int, default=0)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--ckpt_freq', type=int, default=50)
    parser.add_argument('--gif_length', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--train_freq', type=int, default=1)
    parser.add_argument('--batch', action='store_true', help="Roda silenciosamente pegando os args (sem menu TUI)")
    args = parser.parse_args()

    if args.batch:
        train(args.map_name, args.algo, args.use_norm, args.norm_factor, args.use_clip, 
              args.render, args.show_logs, args.delay, args.scale, args.episodes, args.ckpt_freq, args.gif_length, args.lr, args.train_freq)
    else:
        map_name, algo, use_norm, norm_factor, use_clip, render, show_logs, delay, scale, episodes, ckpt_freq, gif_length, lr, train_freq = show_tui()
        train(map_name, algo, use_norm, norm_factor, use_clip, render, show_logs, delay, scale, episodes, ckpt_freq, gif_length, lr, train_freq)


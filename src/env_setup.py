import os
import sys

if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = os.path.join(sys.prefix, "lib", "python3.12", "site-packages", "sumo")

import sumo_rl

def create_env(net_name='nets/2x2grid', render_mode=None, num_seconds=3600, show_logs=False, delay=0, scale=1.0):
    """
    Cria o ambiente de MARL usando sumo-rl e PettingZoo (parallel_env).
    
    Args:
        net_name (str): Diretório/nome da rede embutida no sumo-rl. Padrão: 'nets/2x2grid'.
        render_mode (str): 'human' (usa sumo-gui), 'rgb_array' ou None.
        num_seconds (int): Duração da simulação.
        show_logs (bool): Se exibe ou não os logs 'Step #...' do SUMO.
        delay (int): Atraso em ms entre os passos da simulação (para visualizar melhor na GUI).
        scale (float): Escala da demanda de tráfego. Padrão 1.0.
    """
    sumo_rl_path = os.path.dirname(sumo_rl.__file__)
    
    if net_name == 'nets/2x2grid':
        net_file = os.path.join(sumo_rl_path, net_name, "2x2.net.xml")
        route_file = os.path.join(sumo_rl_path, net_name, "2x2.rou.xml")
    elif net_name == 'nets/3x3grid':
        net_file = os.path.join(sumo_rl_path, net_name, "3x3Grid2lanes.net.xml")
        route_file = os.path.join(sumo_rl_path, net_name, "routes14000.rou.xml")
    elif net_name == 'nets/4x4-Lucas':
        net_file = os.path.join(sumo_rl_path, "nets/4x4-Lucas", "4x4.net.xml")
        route_file = os.path.join(sumo_rl_path, "nets/4x4-Lucas", "4x4c1c2c1c2.rou.xml")
    elif net_name == 'nets/cologne8':
        net_file = os.path.join(sumo_rl_path, "nets", "RESCO", "cologne8", "cologne8.net.xml")
        route_file = os.path.join(sumo_rl_path, "nets", "RESCO", "cologne8", "cologne8.rou.xml")
    else:
        net_file = f"{net_name}.net.xml"
        route_file = f"{net_name}.rou.xml"
        
    sumo_cmd = []
    if not show_logs:
        sumo_cmd.append("--no-step-log")
    if delay > 0:
        sumo_cmd.extend(["--delay", str(delay)])
    if scale != 1.0:
        sumo_cmd.extend(["--scale", str(scale)])
        
    additional_cmd = " ".join(sumo_cmd)
        
    env = sumo_rl.parallel_env(
        net_file=net_file,
        route_file=route_file,
        use_gui=(render_mode == 'human'),
        render_mode=render_mode,
        num_seconds=num_seconds,
        reward_fn='diff-waiting-time',
        additional_sumo_cmd=additional_cmd if additional_cmd else None
    )
    return env

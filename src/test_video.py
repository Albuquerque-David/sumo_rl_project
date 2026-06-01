import os
import sys

if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = os.path.join(sys.prefix, "lib", "python3.12", "site-packages", "sumo")

import sumo_rl

sumo_rl_path = os.path.dirname(sumo_rl.__file__)
net_file = os.path.join(sumo_rl_path, "nets/2x2grid", "2x2.net.xml")
route_file = os.path.join(sumo_rl_path, "nets/2x2grid", "2x2.rou.xml")

env = sumo_rl.parallel_env(
    net_file=net_file,
    route_file=route_file,
    use_gui=False,
    render_mode='rgb_array',
    num_seconds=100
)

obs, info = env.reset()
frames = []

for _ in range(50):
    if not env.agents:
        break
    frame = env.render()
    if frame is not None:
        frames.append(frame)
    
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, term, trunc, infos = env.step(actions)

env.close()

if frames:
    print(f"Captured {len(frames)} frames. Sample shape: {frames[0].shape}")
else:
    print("No frames captured! env.render() returned None.")

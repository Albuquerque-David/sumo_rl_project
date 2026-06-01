import pandas as pd
import glob
import os

def analyze(file_path, window=50):
    print(f"--- Analysis for {os.path.basename(os.path.dirname(file_path))} ---")
    df = pd.read_csv(file_path)
    
    # Split into Early (Random-ish), Mid, and Late stages
    n = len(df)
    
    early = df.iloc[:100]['GlobalReward']
    late = df.iloc[-100:]['GlobalReward']
    
    print(f"Total Episodes: {n}")
    print(f"Early (eps 1-100)  -> Mean: {early.mean():.2f} | Std: {early.std():.2f} | Min: {early.min():.2f} | Max: {early.max():.2f}")
    if n >= 200:
        print(f"Late (eps {n-100}-{n}) -> Mean: {late.mean():.2f} | Std: {late.std():.2f} | Min: {late.min():.2f} | Max: {late.max():.2f}")
    
    # Check what % of episodes are better than -100
    early_good = (early > -100).mean() * 100
    late_good = (late > -100).mean() * 100
    print(f"% Episodes > -100: Early = {early_good:.1f}% | Late = {late_good:.1f}%")
    print("\n")

models_dir = './models'
for d in sorted(glob.glob(os.path.join(models_dir, '20260531-*'))):
    metrics_file = os.path.join(d, 'metrics.csv')
    if os.path.exists(metrics_file):
        analyze(metrics_file)

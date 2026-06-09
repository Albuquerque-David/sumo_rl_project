import pandas as pd
import glob
import os

directories = glob.glob('models/20260604-*') + glob.glob('models/20260605-*') + glob.glob('models/20260606-*') + glob.glob('models/20260607-*') + glob.glob('models/20260608-*')
results = []

for d in directories:
    csv_file = os.path.join(d, 'metrics.csv')
    if not os.path.exists(csv_file):
        continue
    
    df = pd.read_csv(csv_file)
    if len(df) < 450:
        continue
        
    algo = os.path.basename(d).split('_')[1]
    map_name = os.path.basename(d).split('_')[2]
    
    # Média do tempo verde
    green_avg = df['GreenAvg'].mean() if 'GreenAvg' in df.columns else 0
    
    # WaitTime: AVG, MAX (ep), MIN (ep)
    wt_avg = df['WaitTime'].mean()
    wt_max_idx = df['WaitTime'].idxmax()
    wt_max = df.loc[wt_max_idx, 'WaitTime']
    wt_max_ep = df.loc[wt_max_idx, 'Episode']
    
    wt_min_idx = df['WaitTime'].idxmin()
    wt_min = df.loc[wt_min_idx, 'WaitTime']
    wt_min_ep = df.loc[wt_min_idx, 'Episode']
    
    # Throughput: AVG, MAX (ep), MIN (ep)
    thr_avg = df['Throughput'].mean()
    thr_max_idx = df['Throughput'].idxmax()
    thr_max = df.loc[thr_max_idx, 'Throughput']
    thr_max_ep = df.loc[thr_max_idx, 'Episode']
    
    thr_min_idx = df['Throughput'].idxmin()
    thr_min = df.loc[thr_min_idx, 'Throughput']
    thr_min_ep = df.loc[thr_min_idx, 'Episode']
    
    results.append({
        'Map': map_name,
        'Algo': algo,
        'GreenAvg': round(green_avg, 2),
        'WT_Avg': round(wt_avg, 2),
        'WT_Min': f"{wt_min} (Ep {wt_min_ep})",
        'WT_Max': f"{wt_max} (Ep {wt_max_ep})",
        'Thr_Avg': round(thr_avg, 2),
        'Thr_Min': f"{thr_min} (Ep {thr_min_ep})",
        'Thr_Max': f"{thr_max} (Ep {thr_max_ep})"
    })

results_df = pd.DataFrame(results)
if not results_df.empty:
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    for m in results_df['Map'].unique():
        print(f"\n{'='*50}\n MAP: {m} \n{'='*50}")
        map_df = results_df[results_df['Map'] == m].sort_values('WT_Avg')
        print(map_df.to_string(index=False))

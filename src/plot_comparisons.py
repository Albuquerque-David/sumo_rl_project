import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

def plot_comparisons(models_dir='../models'):
    directories = sorted(glob.glob(os.path.join(models_dir, '*')))
    
    if not directories:
        print("Nenhum modelo encontrado.")
        return
        
    plt.figure(figsize=(16, 12))
    
    metrics = [
        ('GlobalReward', 'Recompensa Global do Episódio', 'Recompensa'),
        ('AccReward', 'Recompensa Acumulada (Total)', 'Soma de Recompensa'),
        ('AvgLoss', 'Loss Média', 'Loss'),
        ('Epsilon', 'Epsilon (Taxa de Exploração)', 'Epsilon'),
        ('Throughput', 'Veículos Completados (Throughput)', 'Carros Completados'),
        ('WaitTime', 'Tempo de Espera Total', 'Tempo de Espera Acumulado')
    ]
    
    for i, (col, title, ylabel) in enumerate(metrics, 1):
        plt.subplot(3, 2, i)
        for d in directories:
            metrics_file = os.path.join(d, 'metrics.csv')
            if os.path.exists(metrics_file):
                df = pd.read_csv(metrics_file)
                parts = os.path.basename(d).split('_')
                algo = parts[1] if len(parts) > 1 else 'N/A'
                map_name = parts[2] if len(parts) > 2 else 'N/A'
                label = f"{algo} ({map_name})"
                
                if col in df.columns:
                    plt.plot(df['Episode'], df[col], label=label)
                elif col == 'AccReward' and 'GlobalReward' in df.columns:
                    # Falback para csvs antigos que não tinham AccReward
                    plt.plot(df['Episode'], df['GlobalReward'].cumsum(), label=label)
        
        plt.title(title)
        plt.xlabel('Episódios')
        plt.ylabel(ylabel)
        plt.legend(loc='best', fontsize='small')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, 'comparison_plot.png'))
    print(f"Gráfico completo de comparação salvo em {os.path.join(models_dir, 'comparison_plot.png')}")
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='../models', help='Diretório contendo os modelos')
    args = parser.parse_args()
    
    plot_comparisons(args.dir)

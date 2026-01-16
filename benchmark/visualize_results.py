import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    json_path = Path('benchmark/results.json')
    graph_path = Path('benchmark/benchmark_graph.png')
    table_path = Path('benchmark/benchmark_table.png')
    
    if not json_path.exists():
        print("benchmark/results.json not found.")
        return
        
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
        
    fonts = list(results.keys())
    if not fonts: return
        
    accuracies = [max(0, 1.0 - results[f]['cer']) * 100 for f in fonts] # Char Accuracy
    cers = [results[f]['cer'] for f in fonts]
    fps_list = [results[f]['fps'] for f in fonts]
    
    font_labels = [f.replace('.ttf', '').replace('_', ' ') for f in fonts]
    
    # Sort
    sorted_indices = np.argsort(accuracies)
    font_labels = [font_labels[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    cers = [cers[i] for i in sorted_indices]
    fps_list = [fps_list[i] for i in sorted_indices]
    fonts_sorted = [fonts[i] for i in sorted_indices]
    
    # Style
    try: plt.style.use('ggplot')
    except: pass
    
    # 1. Generate Graph
    fig1, ax1 = plt.subplots(figsize=(10, 6), facecolor='#f8f9fa')
    
    try: colors = plt.cm.viridis(np.linspace(0.4, 0.9, len(accuracies)))
    except: colors = 'skyblue'
    
    bars = ax1.barh(font_labels, accuracies, color=colors, edgecolor='none', height=0.7)
    
    ax1.set_xlabel('Character Accuracy (%)', fontweight='bold', color='#34495e')
    ax1.set_title('OCR Character Accuracy by Font', fontsize=16, fontweight='bold', color='#2c3e50')
    ax1.set_xlim(0, 115)
    ax1.grid(axis='x', linestyle='--', alpha=0.6)
    
    for spine in ax1.spines.values(): spine.set_visible(False)
    
    for bar in bars:
        width = bar.get_width()
        ax1.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                 ha='left', va='center', fontweight='bold', color='#2c3e50', fontsize=10)
                 
    plt.tight_layout()
    plt.savefig(graph_path, dpi=150, facecolor='#f8f9fa')
    print(f"Saved {graph_path}")
    plt.close()
    
    # 2. Generate Table
    # Height calculation: header + rows + padding
    num_rows = len(fonts) + 2 # Header + Avg
    fig_height = max(4, num_rows * 0.5)
    
    fig2, ax2 = plt.subplots(figsize=(10, fig_height), facecolor='#f8f9fa')
    ax2.axis('off')
    
    table_data = [['Font Family', 'Char Acc', 'CER', 'Speed (ms)']]
    for i in range(len(fonts_sorted) - 1, -1, -1):
        f = fonts_sorted[i]
        label = font_labels[i]
        char_acc = max(0, 1.0 - results[f]['cer']) * 100
        acc = f"{char_acc:.1f}%"
        cer = f"{results[f]['cer']:.4f}"
        fps = results[f]['fps']
        ms = 1000.0 / fps if fps > 0 else 0
        table_data.append([label, acc, cer, f"{ms:.1f}"])
        
    avg_cer = np.mean([results[f]['cer'] for f in fonts])
    avg_acc = max(0, 1.0 - avg_cer) * 100
    avg_fps = np.mean([results[f]['fps'] for f in fonts])
    avg_ms = 1000.0 / avg_fps if avg_fps > 0 else 0
    table_data.append(['AVERAGE', f"{avg_acc:.1f}%", f"{avg_cer:.4f}", f"{avg_ms:.1f}"])
    
    # Colors
    header_color = '#d0d0d0' # Light grey header
    row_colors = ['#f8f9fa', '#ffffff']
    
    table = ax2.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.4, 0.2, 0.2, 0.2],
                      edges='horizontal', bbox=[0, 0, 1, 1])
                      
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#cccccc')
        if row == 0:
            cell.set_text_props(weight='bold', color='black') # Black text
            cell.set_facecolor(header_color)
        elif row == len(table_data) - 1:
            cell.set_text_props(weight='bold', color='black')
            cell.set_facecolor('#e0e0e0')
        else:
            cell.set_facecolor(row_colors[row % 2])
            cell.set_text_props(color='black')
            
    plt.tight_layout()
    plt.savefig(table_path, dpi=150, facecolor='#f8f9fa', bbox_inches='tight', pad_inches=0.1)
    print(f"Saved {table_path}")
    plt.close()

if __name__ == "__main__":
    main()

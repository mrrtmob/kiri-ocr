import time
import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import unicodedata

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kiri_ocr import OCR

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def calculate_cer(reference, hypothesis):
    """Calculate Character Error Rate"""
    if not reference:
        return 1.0 if hypothesis else 0.0
    dist = levenshtein_distance(reference, hypothesis)
    return dist / len(reference)

def main():
    print("Initializing OCR model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        model_path = 'models/model.kiri'
        if not os.path.exists(model_path):
             print(f"Warning: {model_path} not found.")
        
        ocr = OCR(model_path=model_path, device=device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    base_data_dir = Path('benchmark/data')
    
    # Find font directories
    font_dirs = [d for d in base_data_dir.iterdir() if d.is_dir()]
    if not font_dirs:
        print("No data found in benchmark/data")
        return
        
    print(f"Found {len(font_dirs)} fonts to benchmark")
    
    font_results = {}
    all_results_detailed = []
    
    global_total_images = 0
    global_total_time = 0
    
    for font_dir in font_dirs:
        font_name = font_dir.name
        print(f"\nBenchmarking font: {font_name}")
        
        splits = ['train', 'val']
        
        total_images = 0
        total_time = 0
        total_cer = 0
        correct_lines = 0
        
        for split in splits:
            labels_file = font_dir / split / 'labels.txt'
            images_dir = font_dir / split / 'images'
            
            if not labels_file.exists():
                continue
                
            with open(labels_file, 'r', encoding='utf-8') as f:
                lines = [line.strip().split('\t', 1) for line in f if line.strip()]
                
            for img_name, gt_text in tqdm(lines, desc=f"  {split}"):
                img_path = images_dir / img_name
                if not img_path.exists():
                    continue
                    
                # Run inference
                start_time = time.time()
                try:
                    # Use recognize_single_line_image to skip detection
                    pred_text, _ = ocr.recognize_single_line_image(str(img_path))
                except Exception as e:
                    # print(f"Error processing {img_name}: {e}")
                    continue
                end_time = time.time()
                
                duration = end_time - start_time
                
                # Normalize
                gt_text = unicodedata.normalize('NFC', gt_text.strip())
                pred_text = unicodedata.normalize('NFC', pred_text.strip())
                
                # Metrics
                cer = calculate_cer(gt_text, pred_text)
                
                total_images += 1
                total_time += duration
                total_cer += cer
                
                if gt_text == pred_text:
                    correct_lines += 1
                elif total_images <= 3: # Debug: Print first few errors per font
                    print(f"\n    [Mismatch] GT: '{gt_text}' | Pred: '{pred_text}' | CER: {cer:.2f}")
                    
                all_results_detailed.append({
                    'font': font_name,
                    'image': img_name,
                    'gt': gt_text,
                    'pred': pred_text,
                    'cer': cer,
                    'time': duration
                })
        
        if total_images > 0:
            avg_cer = total_cer / total_images
            accuracy = correct_lines / total_images
            char_accuracy = max(0.0, 1.0 - avg_cer)
            avg_time = total_time / total_images
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            font_results[font_name] = {
                'images': total_images,
                'cer': avg_cer,
                'accuracy': accuracy,
                'char_accuracy': char_accuracy,
                'fps': fps
            }
            
            global_total_images += total_images
            global_total_time += total_time
            
            print(f"  Result for {font_name}:")
            print(f"    CER: {avg_cer:.4f}")
            print(f"    Line Acc: {accuracy*100:.2f}%")
            print(f"    Char Acc: {char_accuracy*100:.2f}%")
            print(f"    FPS: {fps:.2f}")

    if not font_results:
        print("No results collected.")
        return

    # Save results to JSON
    import json
    with open('benchmark/results.json', 'w', encoding='utf-8') as f:
        json.dump(font_results, f, indent=4)

    # Save detailed results
    with open('benchmark/results.txt', 'w', encoding='utf-8') as f:
        f.write("Benchmark Results per Font\n")
        f.write("==========================\n")
        for font, res in font_results.items():
            f.write(f"Font: {font}\n")
            f.write(f"  Images: {res['images']}\n")
            f.write(f"  CER: {res['cer']:.4f}\n")
            f.write(f"  Accuracy: {res['accuracy']*100:.2f}%\n")
            f.write(f"  FPS: {res['fps']:.2f}\n\n")
            
        f.write("Detailed Results:\n")
        f.write("Font\tImage\tCER\tTime\tGT\tPred\n")
        for res in all_results_detailed:
            f.write(f"{res['font']}\t{res['image']}\t{res['cer']:.4f}\t{res['time']:.4f}\t{res['gt']}\t{res['pred']}\n")
            
    print("\nDetailed results saved to benchmark/results.txt")

if __name__ == "__main__":
    main()

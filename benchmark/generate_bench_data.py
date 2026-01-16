import os
import shutil
import random
from pathlib import Path
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kiri_ocr.generator import DatasetGenerator

def setup_benchmark_fonts():
    """Setup fonts for benchmark"""
    source_fonts_dir = Path('fonts')
    bench_fonts_dir = Path('benchmark/fonts')
    
    if not source_fonts_dir.exists():
        print("Error: fonts directory not found")
        sys.exit(1)
        
    if bench_fonts_dir.exists():
        shutil.rmtree(bench_fonts_dir)
    bench_fonts_dir.mkdir(parents=True)
    
    # List of 10 popular fonts to use
    popular_fonts = [
        'KhmerOSbattambang.ttf',
        'KhmerOScontent.ttf',
        'KhmerOSsiemreap.ttf',
        'KhmerOSmuollight.ttf',
        'Kantumruy Regular.ttf',
        'Battambang-Regular.ttf',
        'Moul.ttf',
        'NokoraRegular.ttf',
        'Suwannaphum.ttf',
        'Khmer-Regular.ttf'
    ]
    
    print("Setting up benchmark fonts...")
    found_fonts = []
    for font_name in popular_fonts:
        src = source_fonts_dir / font_name
        if src.exists():
            shutil.copy(src, bench_fonts_dir / font_name)
            print(f"  Copied {font_name}")
            found_fonts.append(font_name)
        else:
            print(f"  Warning: {font_name} not found in fonts/")
            
    return found_fonts

def create_benchmark_text(input_file, output_file, count=100):
    """Create a text file with random lines"""
    print(f"Selecting {count} random lines from {input_file}...")
    
    selected_lines = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            # Reservoir sampling
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue
                
                if i < count:
                    selected_lines.append(line)
                else:
                    j = random.randint(0, i)
                    if j < count:
                        selected_lines[j] = line
                        
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in selected_lines:
                f.write(line + '\n')
                
        print(f"Created {output_file} with {len(selected_lines)} lines")
        
    except Exception as e:
        print(f"Error reading text file: {e}")
        sys.exit(1)

def main():
    # 1. Setup fonts
    fonts = setup_benchmark_fonts()
    
    # 2. Create text file (100 lines for each font)
    create_benchmark_text('textlines.txt', 'benchmark/bench_text.txt', 1000)
    
    # 3. Generate images per font
    print("\nGenerating benchmark images per font...")
    
    base_output_dir = Path('benchmark/data')
    if base_output_dir.exists():
        shutil.rmtree(base_output_dir)
    base_output_dir.mkdir(parents=True)
    
    for font_name in fonts:
        print(f"\n==========================================")
        print(f"Generating for font: {font_name}")
        print(f"==========================================")
        
        font_clean = Path(font_name).stem.replace(' ', '_').replace('.', '_')
        output_dir = base_output_dir / font_clean
        
        # Create temp dir for this font to force generator to use ONLY this font
        temp_font_dir = Path(f'benchmark/temp_fonts/{font_clean}')
        if temp_font_dir.exists(): shutil.rmtree(temp_font_dir)
        temp_font_dir.mkdir(parents=True)
        
        # Copy font
        shutil.copy(Path('benchmark/fonts') / font_name, temp_font_dir / font_name)
            
        # Initialize generator with this specific font dir
        try:
            # Suppress stdout of generator to reduce clutter
            # sys.stdout = open(os.devnull, 'w') 
            specific_generator = DatasetGenerator(
                language='mixed',
                image_height=32,
                image_width=512,
                fonts_dir=str(temp_font_dir)
            )
            
            specific_generator.generate_dataset(
                train_file='benchmark/bench_text.txt',
                output_dir=str(output_dir),
                train_augment=1,
                val_file=None, # It will auto split, that's fine
                random_augment=True
            )
        except Exception as e:
            print(f"Error generating for {font_name}: {e}")
        finally:
            # sys.stdout = sys.__stdout__
            pass
        
        # Cleanup temp
        shutil.rmtree(temp_font_dir.parent)
    
    print("\nBenchmark data generation complete.")

if __name__ == "__main__":
    main()

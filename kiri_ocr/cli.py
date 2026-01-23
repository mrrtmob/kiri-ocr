import json
import argparse
import sys
import time
from pathlib import Path
import yaml

# Updated default config to include architecture
DEFAULT_TRAIN_CONFIG = {
    "arch": "crnn",       # Options: 'crnn', 'transformer'
    "height": 32,         # 32 for CRNN, 48 for Transformer
    "batch_size": 32,
    "epochs": 100,
    "hidden_size": 256,
    "device": "cuda",
    "output_dir": "models",
    "train_labels": "data/train/labels.txt",
    "val_labels": "data/val/labels.txt",
    "vocab": "vocab_char.json", # Required for Transformer
    "lr": 0.001,
    "weight_decay": 0.0001
}

def init_config(args):
    path = args.output
    if path == 'config.yaml' and not path.endswith('.yaml') and not path.endswith('.yml'):
         pass 
    
    with open(path, 'w') as f:
        yaml.dump(DEFAULT_TRAIN_CONFIG, f, default_flow_style=False)
    print(f"✓ Created default config at {path}")

def run_inference(args):
    import numpy as np
    from .core import OCR
    from .renderer import DocumentRenderer
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    if args.verbose:
        print("\n" + "="*70)
        print("  📄 Kiri OCR System")
        print("="*70)
    
    try:
        ocr = OCR(
            model_path=args.model,
            charset_path=args.charset,
            language=args.language,
            padding=args.padding,
            device=args.device,
            verbose=args.verbose
        )
        
        if not args.verbose:
            print(f"Processing {args.image}...")
            
        full_text, results = ocr.extract_text(args.image, mode=args.mode, verbose=args.verbose)
        
        text_output = output_dir / 'extracted_text.txt'
        with open(text_output, 'w', encoding='utf-8') as f:
            f.write(full_text)
        if args.verbose:
            print(f"\n✓ Text saved to {text_output}")
        
        json_output = output_dir / 'ocr_results.json'
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        if args.verbose:
            print(f"✓ JSON saved to {json_output}")
        
        if not args.no_render:
            renderer = DocumentRenderer()
            renderer.draw_boxes(args.image, results, output_path=str(output_dir / 'boxes.png'))
            renderer.draw_results(args.image, results, output_path=str(output_dir / 'ocr_result.png'))
            renderer.create_report(args.image, results, output_path=str(output_dir / 'report.html'))
        
        if args.verbose:
            print("\n" + "="*70)
            print("  ✅ Processing Complete!")
            print("="*70)
            print(f"  Regions detected: {len(results)}")
            if results:
                print(f"  Average confidence: {np.mean([r['confidence'] for r in results])*100:.2f}%")
            print(f"  Output directory: {output_dir}")
            print("="*70 + "\n")
        else:
            if results:
                for res in results:
                    print(res['text'])
            print(f"\n✓ Saved results to {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if "No such file" in str(e) and "model" in str(e):
             print("\nTip: Make sure you have trained a model or specified the correct path with --model")

def merge_config(args, defaults):
    """Merge defaults < config file < args"""
    config = defaults.copy()
    
    if hasattr(args, 'config') and args.config:
        try:
            with open(args.config, 'r') as f:
                if args.config.endswith('.json'):
                    file_config = json.load(f)
                else:
                    file_config = yaml.safe_load(f)
                
                if file_config:
                    config.update(file_config)
            print(f"Loaded config from {args.config}")
        except Exception as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)
            
    for key, value in vars(args).items():
        if value is not None and key in config:
            config[key] = value
            
    for key, value in config.items():
        setattr(args, key, value)
        
    return args

def print_banner(version="0.0.0"):
    banner = f"""
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║     ██╗  ██╗██╗██████╗ ██╗      ██████╗  ██████╗██████╗        ║
║     ██║ ██╔╝██║██╔══██╗██║     ██╔═══██╗██╔════╝██╔══██╗       ║
║     █████╔╝ ██║██████╔╝██║     ██║   ██║██║     ██████╔╝       ║
║     ██╔═██╗ ██║██╔══██╗██║     ██║   ██║██║     ██╔══██╗       ║
║     ██║  ██╗██║██║  ██║██║     ╚██████╔╝╚██████╗██║  ██║       ║
║     ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝      ╚═════╝  ╚═════╝╚═╝  ╚═╝       ║
║                                                                ║
║                                                                ║
║                   Version: {version:^10}                          ║
╚════════════════════════════════════════════════════════════════╝
"""
    print(banner)
    
def main():
    start_setup = time.time()
    try:
        from . import __version__
    except ImportError:
        __version__ = "0.0.0"
    
    if '--version' in sys.argv:
        print_banner(__version__)
        sys.exit(0)
        
    show_banner = len(sys.argv) == 1 or '-h' in sys.argv or '--help' in sys.argv
    if show_banner:
        print_banner(__version__)
    
    parser = argparse.ArgumentParser(
        description='Kiri OCR - Khmer & English OCR System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False
    )
    
    parser.add_argument('-h', '--help', action='store_true', help='Show this help message and exit')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # === PREDICT ===
    base_path = Path(__file__).parent
    default_model = 'mrrtmob/kiri-ocr'
    default_charset = base_path / 'models' / 'charset_lite.txt'
    
    predict_parser = subparsers.add_parser('predict', help='🔍 Run OCR on an image')
    predict_parser.add_argument('image', help='Path to document image')
    predict_parser.add_argument('--mode', choices=['lines', 'words'], default='lines', help='Detection mode')
    predict_parser.add_argument('--model', default=str(default_model), help='Path to model file')
    predict_parser.add_argument('--charset', default=str(default_charset), help='Path to charset file')
    predict_parser.add_argument('--language', choices=['english', 'khmer', 'mixed'], default='mixed')
    predict_parser.add_argument('--padding', type=int, default=10, help='Padding pixels')
    predict_parser.add_argument('--output', '-o', default='output', help='Output directory')
    predict_parser.add_argument('--no-render', action='store_true', help='Skip rendering')
    predict_parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Device')
    predict_parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    # === TRAIN ===
    train_parser = subparsers.add_parser('train', help='🎓 Train the OCR model')
    
    # NEW ARGUMENT: Architecture Selection
    train_parser.add_argument('--arch', choices=['crnn', 'transformer'], default='crnn', 
                             help='Model architecture: "crnn" (Lightweight) or "transformer" (Infinity)')
    
    train_parser.add_argument('--config', help='Path to config file')
    train_parser.add_argument('--train-labels', help='Path to training labels file')
    train_parser.add_argument('--val-labels', help='Path to validation labels file')
    
    # HuggingFace Args
    train_parser.add_argument('--hf-dataset', help='HuggingFace dataset ID')
    train_parser.add_argument('--hf-subset', help='HuggingFace subset')
    train_parser.add_argument('--hf-train-split', default='train', help='Train split')
    train_parser.add_argument('--hf-val-split', help='Validation split')
    train_parser.add_argument('--hf-streaming', action='store_true', help='Use streaming')
    train_parser.add_argument('--hf-image-col', default='image', help='Image column')
    train_parser.add_argument('--hf-text-col', default='text', help='Text column')
    train_parser.add_argument('--hf-val-percent', type=float, default=0.1, help='Val % if no split found')
    
    # Common Training Args
    train_parser.add_argument('--output-dir', help='Directory to save model')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    train_parser.add_argument('--height', type=int, help='Image height (CRNN=32, Transformer=48)')
    train_parser.add_argument('--hidden-size', type=int, help='Hidden size')
    train_parser.add_argument('--device', help='Device (cuda/cpu)')
    train_parser.add_argument('--from-model', help='Path to pretrained model')
    train_parser.add_argument('--lr', type=float, help='Learning rate')
    train_parser.add_argument('--weight-decay', type=float, help='Weight decay')
    
    # NEW ARGUMENT: Transformer Specific
    train_parser.add_argument('--vocab', help='Path to vocab_char.json (Required for Transformer)')

    # === GENERATE ===
    gen_parser = subparsers.add_parser('generate', help='🎨 Generate synthetic training data')
    gen_parser.add_argument('--train-file', '-t', required=True, help='Training text file')
    gen_parser.add_argument('--val-file', '-v', default=None, help='Validation text file')
    gen_parser.add_argument('--output', '-o', default='data', help='Output directory')
    gen_parser.add_argument('--language', '-l', choices=['english', 'khmer', 'mixed'], default='mixed')
    gen_parser.add_argument('--augment', '-a', type=int, default=1, help='Augmentation factor')
    gen_parser.add_argument('--val-augment', type=int, default=1, help='Validation augmentation')
    gen_parser.add_argument('--height', type=int, default=32, help='Image height')
    gen_parser.add_argument('--width', type=int, default=512, help='Image width')
    gen_parser.add_argument('--fonts-dir', default='fonts', help='Fonts directory')
    gen_parser.add_argument('--font-mode', choices=['random', 'all'], default='random')
    gen_parser.add_argument('--random-augment', action='store_true', help='Apply random augmentations')
    
    # === DETECTOR TOOLS ===
    gen_det_parser = subparsers.add_parser('generate-detector', help='🖼️ Generate detector data')
    gen_det_parser.add_argument('--text-file', required=True)
    gen_det_parser.add_argument('--fonts-dir', default='fonts')
    gen_det_parser.add_argument('--font', help='Specific font')
    gen_det_parser.add_argument('--output', default='detector_dataset')
    gen_det_parser.add_argument('--num-train', type=int, default=800)
    gen_det_parser.add_argument('--num-val', type=int, default=200)
    gen_det_parser.add_argument('--min-lines', type=int, default=15)
    gen_det_parser.add_argument('--max-lines', type=int, default=50)
    gen_det_parser.add_argument('--image-height', type=int, default=512)
    gen_det_parser.add_argument('--no-augment', action='store_true')
    gen_det_parser.add_argument('--workers', type=int, default=1)

    train_det_parser = subparsers.add_parser('train-detector', help='🎯 Train detector')
    train_det_parser.add_argument('--data-yaml', default='detector_dataset/data.yaml')
    train_det_parser.add_argument('--model-size', choices=['n', 's', 'm', 'l', 'x'], default='n')
    train_det_parser.add_argument('--epochs', type=int, default=100)
    train_det_parser.add_argument('--batch-size', type=int, default=16)
    train_det_parser.add_argument('--image-size', type=int, default=640)
    train_det_parser.add_argument('--name', default='khmer_text_detector')

    # === CONFIG ===
    init_parser = subparsers.add_parser('init-config', help='⚙️ Create config file')
    init_parser.add_argument('--output', '-o', default='config.yaml')
    
    # Default to predict
    if len(sys.argv) > 1 and sys.argv[1] not in ['predict', 'train', 'generate', 'init-config', 'generate-detector', 'train-detector', '-h', '--help', '--version']:
        sys.argv.insert(1, 'predict')
    
    if len(sys.argv) > 1 and ('-v' in sys.argv or '--verbose' in sys.argv):
        print(f"CLI Setup: {time.time()-start_setup:.3f}s", file=sys.stderr)
    
    args = parser.parse_args()
    
    if hasattr(args, 'help') and args.help:
        parser.print_help()
        sys.exit(0)
    
    # === COMMAND ROUTING ===
    if args.command == 'predict':
        run_inference(args)
        
    elif args.command == 'train':
        # Merge config first
        args = merge_config(args, DEFAULT_TRAIN_CONFIG)
        
        # Check constraints
        if not args.train_labels and not args.hf_dataset:
            print("❌ Error: --train-labels or --hf-dataset is required")
            sys.exit(1)

        # === NEW LOGIC: Dispatch based on Architecture ===
        if args.arch == 'transformer':
            print("\n🔄 Switching to Transformer (Infinity) Training Pipeline...")
            
            # Ensure proper height for Transformer
            if args.height == 32: 
                print("⚠️ Warning: Transformer model usually requires height=48. Overriding default.")
                args.height = 48
                
            # Auto-vocab generation is supported, so we relax the check
            # if not args.vocab and not Path('vocab_char.json').exists():
            #      print("❌ Error: --vocab (path to vocab_char.json) is required for Transformer training.")
            #      sys.exit(1)

            # Import the new training module (Assuming you saved the previous code as training_transformer.py)
            try:
                from .training_transformer import train_command as train_transformer_cmd
                train_transformer_cmd(args)
            except ImportError:
                print("❌ Error: Could not find '.training_transformer'. Make sure you saved the training code.")
                sys.exit(1)
        else:
            # Default CRNN Training
            from .training import train_command
            train_command(args)
            
    elif args.command == 'generate':
        from .generator import generate_command
        generate_command(args)
    elif args.command == 'generate-detector':
        from .detector.dataset import generate_detector_dataset_command
        generate_detector_dataset_command(args)
    elif args.command == 'train-detector':
        from .detector.training import train_detector_command
        train_detector_command(args)
    elif args.command == 'init-config':
        init_config(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
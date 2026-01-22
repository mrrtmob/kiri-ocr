from ultralytics import YOLO
import torch
import os
import argparse

class DetectorTrainer:
    def __init__(self, data_yaml='detector_dataset/data.yaml'):
        self.data_yaml = data_yaml
        self.check_setup()
    
    def check_setup(self):
        """Check if dataset and dependencies are ready"""
        print("=== Checking Setup ===")
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            self.device = 0
        else:
            print("⚠ No GPU found, training will use CPU (slower)")
            self.device = 'cpu'
        
        # Check dataset
        if os.path.exists(self.data_yaml):
            print(f"✓ Dataset config found: {self.data_yaml}")
        else:
            print(f"✗ Dataset config not found: {self.data_yaml}")
            print("Please run the dataset generator first!")
            # We don't exit here to allow library usage to handle error, but for CLI it will fail later or here
            
        
        # Check if images exist
        dataset_dir = os.path.dirname(self.data_yaml)
        train_imgs = f"{dataset_dir}/images/train"
        val_imgs = f"{dataset_dir}/images/val"
        
        if os.path.exists(train_imgs):
            num_train = len([f for f in os.listdir(train_imgs) if f.endswith(('.jpg', '.png'))])
            print(f"✓ Training images: {num_train}")
        else:
            print(f"✗ Training images not found in {train_imgs}")
        
        if os.path.exists(val_imgs):
            num_val = len([f for f in os.listdir(val_imgs) if f.endswith(('.jpg', '.png'))])
            print(f"✓ Validation images: {num_val}")
        else:
            print(f"✗ Validation images not found in {val_imgs}")
        
        print("✓ Setup check complete!\n")
    
    def train_model(self, 
                   model_size='n',      # 'n', 's', 'm', 'l', 'x'
                   epochs=100,
                   batch_size=16,
                   image_size=640,
                   name='khmer_text_detector'):
        """Train YOLO model on Khmer text dataset"""
        
        print("=== Starting Training ===")
        print(f"Model: YOLOv8{model_size}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Image size: {image_size}")
        print(f"Device: {self.device}")
        print()
        
        # Load pretrained model
        model_path = f'yolov8{model_size}.pt'
        print(f"Loading pretrained model: {model_path}")
        model = YOLO(model_path)
        
        # Train the model
        results = model.train(
            data=self.data_yaml,
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            name=name,
            device=self.device,
            
            # Training parameters
            patience=20,              # Early stopping patience
            save=True,                # Save checkpoints
            save_period=10,           # Save every N epochs
            cache=False,              # Cache images for faster training (use True if you have RAM)
            
            # Optimization
            optimizer='Adam',         # Adam optimizer
            lr0=0.01,                 # Initial learning rate
            lrf=0.01,                 # Final learning rate
            momentum=0.937,           # SGD momentum
            weight_decay=0.0005,      # Optimizer weight decay
            
            # Augmentation
            augment=True,             # Use data augmentation
            mosaic=1.0,               # Mosaic augmentation probability
            mixup=0.0,                # Mixup augmentation probability
            copy_paste=0.0,           # Copy-paste augmentation
            degrees=10.0,             # Rotation degrees
            translate=0.1,            # Translation
            scale=0.5,                # Scale
            shear=0.0,                # Shear
            perspective=0.0,          # Perspective
            flipud=0.0,               # Flip up-down
            fliplr=0.5,               # Flip left-right
            hsv_h=0.015,              # HSV-Hue augmentation
            hsv_s=0.7,                # HSV-Saturation
            hsv_v=0.4,                # HSV-Value
            
            # Other settings
            verbose=True,             # Verbose output
            seed=42,                  # Random seed
            deterministic=True,       # Deterministic training
            single_cls=True,          # Single class dataset
            rect=False,               # Rectangular training
            cos_lr=False,             # Cosine learning rate scheduler
            close_mosaic=10,          # Disable mosaic last N epochs
            amp=True,                 # Automatic Mixed Precision
            fraction=1.0,             # Dataset fraction to use
            profile=False,            # Profile ONNX and TensorRT speeds
            
            # Validation
            val=True,                 # Validate during training
            plots=True,               # Save plots
            
            # Project organization
            project='runs/detect',
            exist_ok=True,
        )
        
        print("\n=== Training Complete! ===")
        print(f"Best model saved to: runs/detect/{name}/weights/best.pt")
        print(f"Last model saved to: runs/detect/{name}/weights/last.pt")
        print(f"Results saved to: runs/detect/{name}/")
        
        return results
    
    def resume_training(self, checkpoint_path, epochs=50):
        """Resume training from a checkpoint"""
        print(f"Resuming training from: {checkpoint_path}")
        model = YOLO(checkpoint_path)
        
        results = model.train(
            resume=True,
            epochs=epochs
        )
        
        return results
    
    def evaluate_model(self, model_path):
        """Evaluate trained model"""
        print(f"\n=== Evaluating Model ===")
        model = YOLO(model_path)
        
        # Validate on validation set
        metrics = model.val(data=self.data_yaml)
        
        print(f"\nMetrics:")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")
        
        return metrics

def train_detector_command(args):
    """
    CLI Command handler for detector training
    """
    # Initialize trainer
    if not os.path.exists(args.data_yaml):
         print(f"Error: Data config '{args.data_yaml}' not found. Please run generate-detector-dataset first.")
         exit(1)

    trainer = DetectorTrainer(data_yaml=args.data_yaml)
    
    # Train model
    results = trainer.train_model(
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        name=args.name
    )
    
    # Evaluate the best model
    best_model_path = f'runs/detect/{args.name}/weights/best.pt'
    if os.path.exists(best_model_path):
        trainer.evaluate_model(best_model_path)
    
    print("\n=== Training Summary ===")
    print("To use your model for detection:")
    print(f"  model = YOLO('{best_model_path}')")
    print("\nTo view training results:")
    print(f"  Check: runs/detect/{args.name}/")

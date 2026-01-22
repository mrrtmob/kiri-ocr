from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path

class YOLOTextDetector:
    def __init__(self, model_path='runs/detect/khmer_text_detector/weights/best.pt'):
        """
        Initialize YOLO Text Detector
        
        Args:
            model_path: Path to your trained YOLO model
        """
        # Allow passing None or empty string to defer loading or if handled elsewhere
        # But typically we want to load it.
        # If model_path doesn't exist, we might want to use a default or download it?
        # For now, we'll stick to the user's logic but maybe softer error or check.
        
        self.model = None
        if model_path and os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            self.model = YOLO(model_path)
            print("Model loaded successfully!")
        else:
            # Fallback or just print warning if not found, 
            # but user logic raised FileNotFoundError.
            # I will keep it raising error if path is provided but missing.
            if model_path:
                 raise FileNotFoundError(f"Model not found: {model_path}")
    
    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = YOLO(model_path)

    def detect_image(self, image_path, conf_threshold=0.25, save_path=None, show_labels=True):
        """
        Detect text in a single image
        
        Args:
            image_path: Path to input image (or numpy array)
            conf_threshold: Confidence threshold (0.0 - 1.0)
            save_path: Path to save output image (None = auto-generate)
            show_labels: Whether to show confidence labels
            
        Returns:
            List of detections with bounding boxes and confidence scores
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Please initialize with a valid model path.")

        # Handle numpy array input
        if isinstance(image_path, np.ndarray):
            image = image_path.copy()
            # YOLO can take numpy array directly
            source = image
        else:
            if not os.path.exists(image_path):
                print(f"Error: Image not found: {image_path}")
                return []
            print(f"\nProcessing: {image_path}")
            source = image_path
            image = cv2.imread(image_path)
        
        # Run detection
        results = self.model(source, conf=conf_threshold, verbose=False)
        
        # Get detections
        boxes = results[0].boxes
        detections = []
        
        # print(f"Found {len(boxes)} text regions")
        
        for i, box in enumerate(boxes):
            # Get coordinates and confidence
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            
            # Store detection
            detection = {
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': float(conf),
                'area': (int(x2) - int(x1)) * (int(y2) - int(y1))
            }
            detections.append(detection)
            
            # Draw rectangle (green)
            cv2.rectangle(image, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         (0, 255, 0), 2)
            
            # Add label with confidence
            if show_labels:
                label = f'Text: {conf:.2f}'
                # Background for text
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(image, 
                            (int(x1), int(y1) - 25), 
                            (int(x1) + w, int(y1)), 
                            (0, 255, 0), -1)
                # Text
                cv2.putText(image, label, 
                           (int(x1), int(y1) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 0, 0), 2)
            
            # print(f"  Detection {i+1}: Box({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}) | Conf: {conf:.3f}")
        
        # Save output only if path provided
        if save_path:
            cv2.imwrite(save_path, image)
            print(f"Saved result to: {save_path}")
        
        return detections
    
    def detect_batch(self, image_folder, output_folder='output_detections', 
                    conf_threshold=0.25, extensions=('.jpg', '.png', '.jpeg')):
        """
        Detect text in multiple images
        
        Args:
            image_folder: Folder containing images
            output_folder: Folder to save results
            conf_threshold: Confidence threshold
            extensions: Tuple of valid image extensions
            
        Returns:
            Dictionary with results for each image
        """
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all images
        image_files = []
        for ext in extensions:
            image_files.extend(Path(image_folder).glob(f'*{ext}'))
        
        print(f"\n=== Batch Processing ===")
        print(f"Found {len(image_files)} images in {image_folder}")
        
        all_results = {}
        total_detections = 0
        
        for i, img_path in enumerate(image_files, 1):
            output_path = os.path.join(output_folder, f'detected_{img_path.name}')
            
            detections = self.detect_image(
                str(img_path), 
                conf_threshold=conf_threshold,
                save_path=output_path,
                show_labels=True
            )
            
            all_results[str(img_path)] = detections
            total_detections += len(detections)
            
            print(f"Progress: {i}/{len(image_files)}")
        
        print(f"\n=== Batch Complete ===")
        print(f"Processed: {len(image_files)} images")
        print(f"Total detections: {total_detections}")
        print(f"Average per image: {total_detections/len(image_files):.1f}")
        print(f"Results saved to: {output_folder}")
        
        return all_results
    
    def extract_text_regions(self, image_path, output_folder='text_regions', 
                            conf_threshold=0.25, padding=5):
        """
        Extract detected text regions as separate images
        
        Args:
            image_path: Path to input image
            output_folder: Folder to save cropped regions
            conf_threshold: Confidence threshold
            padding: Extra padding around text (pixels)
            
        Returns:
            List of cropped region file paths
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        os.makedirs(output_folder, exist_ok=True)
        
        # Run detection
        results = self.model(image_path, conf=conf_threshold, verbose=False)
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        
        boxes = results[0].boxes
        region_paths = []
        
        print(f"\nExtracting {len(boxes)} text regions from {image_path}")
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = box.conf[0].cpu().numpy()
            
            # Add padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # Crop region
            region = image[y1:y2, x1:x2]
            
            # Save
            base_name = Path(image_path).stem
            region_path = os.path.join(output_folder, f'{base_name}_region_{i+1}_conf{conf:.2f}.png')
            cv2.imwrite(region_path, region)
            region_paths.append(region_path)
            
            print(f"  Saved region {i+1}: {region_path}")
        
        return region_paths
    
    def detect_video(self, video_path, output_path='output_video.mp4', 
                    conf_threshold=0.25, show_fps=True):
        """
        Detect text in video
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            conf_threshold: Confidence threshold
            show_fps: Show FPS counter
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        print(f"\n=== Processing Video ===")
        print(f"Input: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        # Get properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Total frames: {total_frames}")
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_detections = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = self.model(frame, conf=conf_threshold, verbose=False)
            
            # Draw boxes
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = box.conf[0].cpu().numpy()
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{conf:.2f}', 
                           (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
                
                total_detections += 1
            
            # Show FPS
            if show_fps:
                cv2.putText(frame, f'Frame: {frame_count}/{total_frames}', 
                           (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Progress: {frame_count}/{total_frames} frames")
        
        cap.release()
        out.release()
        
        print(f"\n=== Video Complete ===")
        print(f"Processed: {frame_count} frames")
        print(f"Total detections: {total_detections}")
        print(f"Output saved to: {output_path}")
    
    def detect_webcam(self, conf_threshold=0.25, save_video=False):
        """
        Real-time detection from webcam
        
        Args:
            conf_threshold: Confidence threshold
            save_video: Whether to save the webcam feed
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        print("\n=== Webcam Detection ===")
        print("Press 'q' to quit")
        print("Press 's' to save current frame")
        
        cap = cv2.VideoCapture(0)
        
        # Video writer (optional)
        if save_video:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('webcam_output.mp4', fourcc, 20.0, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = self.model(frame, conf=conf_threshold, verbose=False)
            
            # Draw boxes
            num_detections = 0
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = box.conf[0].cpu().numpy()
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Text: {conf:.2f}', 
                           (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 255, 0), 2)
                num_detections += 1
            
            # Show info
            cv2.putText(frame, f'Detections: {num_detections}', 
                       (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            
            # Display
            cv2.imshow('Khmer Text Detection - Press Q to quit', frame)
            
            if save_video:
                out.write(frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_path = f'screenshot_{frame_count}.png'
                cv2.imwrite(screenshot_path, frame)
                print(f"Saved screenshot: {screenshot_path}")
            
            frame_count += 1
        
        cap.release()
        if save_video:
            out.release()
            print("Webcam video saved to: webcam_output.mp4")
        cv2.destroyAllWindows()

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

class DocumentRenderer:
    """Render OCR results on image"""
    
    def __init__(self, font_path=None, font_size=12):
        self.font_size = font_size
        
        # Try to load font for text rendering
        self.font = None
        if font_path and Path(font_path).exists():
            try:
                self.font = ImageFont.truetype(font_path, font_size)
            except:
                pass
        
        # Fallback fonts
        if self.font is None:
            # Priority list including Khmer fonts
            candidate_fonts = [
                'fonts/KhmerOSbattambang.ttf',
                'fonts/Battambang-Regular.ttf',
                'fonts/NotoSansKhmer-Regular.ttf',
                'Arial.ttf',
                'DejaVuSans.ttf',
                'NotoSans-Regular.ttf'
            ]
            
            # Check for any TTF in fonts/ directory
            if Path('fonts').exists():
                candidate_fonts = [str(f) for f in Path('fonts').glob('*.ttf')] + candidate_fonts
            
            for font_name in candidate_fonts:
                try:
                    self.font = ImageFont.truetype(font_name, font_size)
                    break
                except:
                    continue
    
    def draw_boxes(self, image_path, results, output_path='output_boxes.png'):
        """Draw bounding boxes only"""
        img = cv2.imread(str(image_path))
        
        for result in results:
            x, y, w, h = result['box']
            conf = result['confidence']
            
            # Color based on confidence
            if conf > 0.9:
                color = (0, 255, 0)  # Green
            elif conf > 0.7:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 0, 255)  # Red
            
            # Draw box
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            
            # Draw line number
            if 'line_number' in result:
                cv2.putText(img, str(result['line_number']), (x-5, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imwrite(output_path, img)
        print(f"\n✓ Boxes saved to {output_path}")
        
        return img
    
    def draw_results(self, image_path, results, output_path='output_ocr.png',
                    show_text=True, show_confidence=True):
        """Draw boxes with recognized text"""
        img = cv2.imread(str(image_path))
        
        # Convert to PIL for better text rendering
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        for result in results:
            x, y, w, h = result['box']
            text = result['text']
            conf = result['confidence']
            
            # Color based on confidence
            if conf > 0.9:
                color = (0, 255, 0)
            elif conf > 0.7:
                color = (255, 165, 0)
            else:
                color = (255, 0, 0)
            
            # Draw box
            draw.rectangle([x, y, x+w, y+h], outline=color, width=2)
            
            if show_text:
                # Prepare label
                label = text[:50]  # Limit length
                if show_confidence:
                    label += f" ({conf*100:.0f}%)"
                
                # Background for text
                if self.font:
                    try:
                        # Calculate text position
                        left, top, right, bottom = draw.textbbox((0, 0), label, font=self.font)
                        text_height = bottom - top
                        text_y = y - text_height - 5
                        
                        bbox = draw.textbbox((x, text_y), label, font=self.font)
                        # Add padding
                        bbox = (bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2)
                        
                        draw.rectangle(bbox, fill=color)
                        draw.text((x, text_y), label, fill=(255, 255, 255), font=self.font)
                    except:
                        # Fallback
                        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                        cv2.putText(img_cv, label, (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(img_pil)
        
        # Save
        img_pil.save(output_path)
        print(f"✓ Results saved to {output_path}")
        
        return img_pil
    
    def create_report(self, image_path, results, output_path='ocr_report.html'):
        """Create HTML report"""
        # Calculate stats safely
        avg_conf = np.mean([r['confidence'] for r in results]) * 100 if results else 0.0

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>OCR Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #4CAF50; color: white; padding: 20px; border-radius: 5px; }}
        .result {{ padding: 10px; margin: 10px 0; border-left: 4px solid #4CAF50; }}
        .conf-high {{ border-color: #4CAF50; background: #f1f8f4; }}
        .conf-medium {{ border-color: #FF9800; background: #fff8f1; }}
        .conf-low {{ border-color: #F44336; background: #fef1f1; }}
        .text {{ font-size: 18px; font-weight: bold; margin-bottom: 5px; }}
        .confidence {{ color: #666; font-size: 14px; }}
        .stats {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .full-text {{ background: #f5f5f5; padding: 20px; white-space: pre-wrap; 
                     font-family: monospace; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📄 OCR Report</h1>
        <p>Document: {Path(image_path).name}</p>
    </div>
    
    <div class="stats">
        <strong>Statistics:</strong><br>
        Total Regions: {len(results)}<br>
        Average Confidence: {avg_conf:.2f}%<br>
        High Confidence (>90%): {sum(1 for r in results if r['confidence'] > 0.9)}<br>
        Medium Confidence (70-90%): {sum(1 for r in results if 0.7 < r['confidence'] <= 0.9)}<br>
        Low Confidence (<70%): {sum(1 for r in results if r['confidence'] <= 0.7)}
    </div>
    
    <h2>📝 Full Text</h2>
    <div class="full-text">{"<br>".join([r['text'] for r in results])}</div>
    
    <h2>📋 Detailed Results</h2>
"""
        
        for i, result in enumerate(results, 1):
            conf = result['confidence']
            conf_class = 'conf-high' if conf > 0.9 else ('conf-medium' if conf > 0.7 else 'conf-low')
            
            html += f"""
    <div class="result {conf_class}">
        <div class="text">{i}. {result['text']}</div>
        <div class="confidence">Confidence: {conf*100:.2f}% | Box: {result['box']}</div>
    </div>
"""
        
        html += """
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✓ Report saved to {output_path}")

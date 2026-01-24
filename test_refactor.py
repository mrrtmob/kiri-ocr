from kiri_ocr import OCR

# Initialize (auto-downloads from Hugging Face)
ocr = OCR()

# Extract text from document
text, results = ocr.extract_text('image.png')
print(text)

# Get detailed results with confidence scores
for line in results:
    print(f"{line['text']} (confidence: {line['confidence']:.1%})")

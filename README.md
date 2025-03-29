# PDFusion - Advanced PDF Comparison Tool

PDFusion is a powerful PDF comparison tool that supports both text and image-based PDFs, with AI-powered analysis of differences.

## Features

- 📄 Hybrid Support: Compare both text-based and image-based PDFs
- 🤖 AI Analysis: Optional LLM-powered explanation of differences
- 🔍 OCR Support: Automatic text extraction from images
- 🎨 Visual Highlighting: Clear color-coded differences
- 🚀 Easy to Use: Simple web interface

## Setup

1. Install Python 3.9 or higher
2. Install Tesseract OCR:
   - Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki
   - Linux: `sudo apt-get install tesseract-ocr`
   - Mac: `brew install tesseract`

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python pdf_fusion.py
```

2. Open the provided URL in your web browser
3. Upload two PDFs for comparison
4. (Optional) Add your OpenAI API key for AI-powered analysis
5. View the differences highlighted in red (removed) and green (added)

## Notes

- For image-based PDFs, the tool uses OCR to extract text
- The OpenAI API key is optional but enables AI analysis of differences
- Temporary files are automatically cleaned up after comparison 
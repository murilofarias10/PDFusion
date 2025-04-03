# PDFusion

PDFusion is an AI-powered PDF comparison tool that helps you identify differences between PDF documents quickly and efficiently.

## Demo Website

Visit our [Demo Website](https://murilofarias10.github.io/PDFusion/) to see how the interface looks.

**Note:** The GitHub Pages demo is a static version showing the UI only. For full functionality, you need to run the Flask application locally.

## Features

- Visual comparison of PDF documents
- Text extraction and difference highlighting
- Image extraction and comparison
- AI-powered analysis of differences
- Works with both text-heavy documents and diagrams/drawings

## Running Locally

To run the full application locally with all features:

1. Clone the repository:
   ```
   git clone https://github.com/murilofarias10/PDFusion.git
   cd PDFusion
   ```

2. Install dependencies:
   ```
   pip install flask fitz Pillow difflib openai pytesseract
   ```

3. Set your OpenAI API key (optional, but recommended for AI-powered analysis):
   ```
   export OPENAI_API_KEY=your_api_key_here
   ```

4. Run the Flask application:
   ```
   python app.py
   ```

5. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## How It Works

1. **Upload** - Upload your original and modified PDF documents
2. **Extract** - The system extracts text and images from both documents
3. **Process** - Text and visual differences are identified
4. **Analyze** - AI analyzes the significance of the changes
5. **Compare** - View highlighted differences in an easy-to-understand format

## Requirements

- Python 3.7+
- Flask
- PyMuPDF (fitz)
- Pillow
- OpenAI Python library (for AI analysis)
- Tesseract OCR (for image text extraction)

## Contact

For questions or issues, please open an issue on GitHub.

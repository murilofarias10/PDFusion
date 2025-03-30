# PDFusion - PDF Comparison Tool

PDFusion is a web application that allows users to compare two PDF documents, highlighting differences in both text content and images. The tool leverages AI to provide a concise summary of the key differences between the documents, with special handling for technical drawings.

## Features

- **Text Comparison**: 
  - Line-by-line comparison using difflib with traditional HTML diff view
  - Word-by-word comparison with color-highlighted additions and deletions
  - Easily identify added, removed, and modified text

- **Image & Drawing Comparison**: 
  - Automatic detection of technical drawings vs. regular images
  - Enhanced drawing comparison with intelligent contour detection
  - Color-coded highlighting of additions (green) and deletions (red)
  - Side-by-side view of original and modified pages

- **AI-Powered Analysis**: Uses OpenAI's GPT models to generate human-readable summaries of detected differences, including both text and drawing modifications.

- **Modern UI**: Clean, responsive interface built with Bootstrap for a seamless user experience.

## Technology Stack

- **Backend**: Flask (Python)
- **PDF Processing**: PyPDF2 for text extraction and PyMuPDF for page rendering
- **Text Comparison**: difflib for comparing text content
- **Image Processing**: OpenCV and NumPy for image and drawing comparison
- **AI Integration**: OpenAI API for generating difference summaries
- **Frontend**: HTML, CSS, Bootstrap 5

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/PDFusion.git
   cd PDFusion
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up your OpenAI API key:
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

5. Run the application:
   ```
   python app.py
   ```

6. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

1. Upload two PDF files you want to compare.
2. Click the "Compare PDFs" button.
3. View the AI-generated summary of differences.
4. Switch between tabs to explore:
   - Word-by-word text differences
   - Traditional line-by-line text comparison
   - Visual differences with highlighted changes
   - Side-by-side page comparison

## How It Works

PDFusion performs these types of comparisons:

1. **Text Comparison**: 
   - Extracts text from both PDFs using PyPDF2
   - Provides both line-by-line comparison (traditional diff) and word-by-word comparison
   - Uses different highlighting colors for additions and deletions

2. **Image & Drawing Comparison**:
   - Renders each PDF page as an image using PyMuPDF
   - Uses intelligent detection to identify technical drawings
   - For technical drawings: applies contour detection to identify specific additions and deletions
   - For regular images: uses pixel-level comparison to highlight differences

3. **AI Analysis**:
   - Sends the text differences, image comparison results, and drawing analysis to OpenAI's API
   - Generates a human-readable summary of all detected differences
   - Specially handles technical drawings to provide meaningful insights

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [PyPDF2](https://pypdf2.readthedocs.io/) for PDF text extraction
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF image extraction and rendering
- [OpenAI](https://openai.com/) for providing the API for AI-powered summaries
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [OpenCV](https://opencv.org/) for image and drawing processing capabilities
- [Bootstrap](https://getbootstrap.com/) for the frontend components 
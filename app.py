from flask import Flask, render_template, request, jsonify
import fitz  # PyMuPDF
import difflib
import pytesseract
from PIL import Image, ImageDraw, ImageChops
import io
import numpy as np
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
import tempfile
from werkzeug.utils import secure_filename
import base64
import cv2

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'temp_uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class PDFComparator:
    def __init__(self):
        """Initialize the PDF comparator with OpenAI API key from environment."""
        self.api_key = os.getenv('OPENAI_API_KEY')
        if self.api_key:
            try:
                self.llm = ChatOpenAI(api_key=self.api_key)
                self.prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an expert at analyzing differences between documents, including both text and technical drawings. "
                            "Explain the key differences in a clear, concise way."),
                    ("user", "Here are the differences between two PDFs:\n\n"
                            "Text Differences:\n{differences}\n\n"
                            "Drawing Changes:\n{drawing_changes}\n\n"
                            "Please explain these differences in a simple way, focusing on both text changes and visual/drawing modifications.")
                ])
            except Exception as e:
                print(f"Error initializing LLM: {str(e)}")
                self.llm = None
        else:
            self.llm = None

    def analyze_drawing_changes(self, changes):
        """Analyze the drawing changes and return a description."""
        if not changes:
            return "No changes detected in drawings."
        
        total_changes = len(changes)
        additions = sum(1 for c in changes if c['type'] == 'addition')
        deletions = sum(1 for c in changes if c['type'] == 'deletion')
        
        # Calculate positions of changes
        positions = []
        for change in changes:
            if change.get('position'):
                x, y, w, h = change['position']
                area = change['area']
                if area > 100:  # Only mention significant changes
                    pos = "top" if y < 400 else "bottom"
                    pos += " " + ("left" if x < 400 else "right")
                    positions.append(f"{change['type']} in {pos} area")
        
        position_info = ""
        if positions:
            position_info = "\nLocations: " + "; ".join(positions[:3])
            if len(positions) > 3:
                position_info += f" and {len(positions) - 3} more changes"
        
        return f"Found {total_changes} changes in the drawings: {additions} additions (shown in green) and {deletions} deletions (shown in red).{position_info}"

    def compare_images(self, img1, img2):
        """Compare two images and detect additions (green) and deletions (red)."""
        # Convert images to numpy arrays
        np_img1 = np.array(img1)
        np_img2 = np.array(img2)

        # Ensure images are in RGB format
        if len(np_img1.shape) == 2:
            np_img1 = cv2.cvtColor(np_img1, cv2.COLOR_GRAY2RGB)
        if len(np_img2.shape) == 2:
            np_img2 = cv2.cvtColor(np_img2, cv2.COLOR_GRAY2RGB)

        # Convert to grayscale for comparison
        gray1 = cv2.cvtColor(np_img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(np_img2, cv2.COLOR_RGB2GRAY)

        # Find differences with improved thresholding
        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to better detect line changes
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        thresh = cv2.erode(thresh, kernel, iterations=1)

        # Find contours of differences
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create result images
        result_original = np_img1.copy()
        result_modified = np_img2.copy()
        changes = []

        # Create semi-transparent overlays
        overlay_original = np.zeros_like(result_original)
        overlay_modified = np.zeros_like(result_modified)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 15:  # Lower threshold for detecting smaller changes
                # Compare the regions to determine if it's an addition or deletion
                mask = np.zeros_like(gray1)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                
                # Get average intensities in both images
                region1 = cv2.mean(gray1, mask=mask)[0]
                region2 = cv2.mean(gray2, mask=mask)[0]
                
                # Compare intensities to determine addition or deletion
                is_addition = region1 > region2
                
                if is_addition:
                    # Draw semi-transparent green on modified image for additions
                    cv2.drawContours(overlay_modified, [contour], -1, (50, 255, 50), -1)  # Light green fill
                    cv2.drawContours(result_modified, [contour], -1, (0, 200, 0), 2)  # Darker green outline
                else:
                    # Draw semi-transparent red on original image for deletions
                    cv2.drawContours(overlay_original, [contour], -1, (50, 50, 255), -1)  # Light red fill
                    cv2.drawContours(result_original, [contour], -1, (0, 0, 200), 2)  # Darker red outline
                
                changes.append({
                    'type': 'addition' if is_addition else 'deletion',
                    'area': area,
                    'position': cv2.boundingRect(contour)
                })

        # Create a white background for better text visibility
        white_bg_original = np.ones_like(result_original) * 255
        white_bg_modified = np.ones_like(result_modified) * 255

        # Blend the overlays with increased opacity but maintain text visibility
        alpha_overlay = 0.3  # Reduced opacity for better text visibility
        alpha_original = 0.7  # Keep more of the original content visible
        
        # Blend original content with white background
        result_original = cv2.addWeighted(result_original, alpha_original, white_bg_original, 1 - alpha_original, 0)
        result_modified = cv2.addWeighted(result_modified, alpha_original, white_bg_modified, 1 - alpha_original, 0)
        
        # Add the colored overlays
        result_original = cv2.addWeighted(result_original, 1, overlay_original, alpha_overlay, 0)
        result_modified = cv2.addWeighted(result_modified, 1, overlay_modified, alpha_overlay, 0)

        # Convert back to PIL Images
        result_original_pil = Image.fromarray(cv2.cvtColor(result_original, cv2.COLOR_BGR2RGB))
        result_modified_pil = Image.fromarray(cv2.cvtColor(result_modified, cv2.COLOR_BGR2RGB))

        return result_original_pil, result_modified_pil, changes

    def extract_words(self, text):
        """Extract words from text, handling various separators and special characters."""
        # Remove special characters and normalize spaces
        text = text.lower()
        text = ''.join(c for c in text if c.isalnum() or c.isspace())
        # Split into words and filter out empty strings
        words = [word for word in text.split() if word]
        return words

    def is_text_based_pdf(self, pdf_path):
        """Check if PDF is text-based by analyzing its content."""
        doc = fitz.open(pdf_path)
        total_text = ""
        total_images = 0
        
        for page in doc:
            # Get text content
            text = page.get_text()
            total_text += text
            
            # Count images
            image_list = page.get_images()
            total_images += len(image_list)
        
        # Calculate text density
        words = self.extract_words(total_text)
        text_density = len(words) / (len(doc) + 1)  # Avoid division by zero
        
        # PDF is considered text-based if:
        # 1. Has significant text density (> 50 words per page)
        # 2. Has few images (< 3 per page)
        return text_density > 50 and (total_images / len(doc)) < 3

    def compare_words(self, words1, words2):
        """Compare two lists of words and return differences."""
        # Use difflib to find word differences
        matcher = difflib.SequenceMatcher(None, words1, words2)
        
        # Prepare word-level differences
        word_diffs = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                # Show removed words in red
                for word in words1[i1:i2]:
                    word_diffs.append(f'<span class="removed-word">{word}</span>')
                # Show added words in green
                for word in words2[j1:j2]:
                    word_diffs.append(f'<span class="added-word">{word}</span>')
            elif tag == 'delete':
                # Show removed words in red
                for word in words1[i1:i2]:
                    word_diffs.append(f'<span class="removed-word">{word}</span>')
            elif tag == 'insert':
                # Show added words in green
                for word in words2[j1:j2]:
                    word_diffs.append(f'<span class="added-word">{word}</span>')
            elif tag == 'equal':
                # Show unchanged words normally
                for word in words1[i1:i2]:
                    word_diffs.append(f'<span class="unchanged-word">{word}</span>')
        
        return ' '.join(word_diffs)

    def extract_text_and_images(self, pdf_path):
        """Extract text and images from PDF."""
        doc = fitz.open(pdf_path)
        result = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_dict = {
                'text': page.get_text(),
                'images': []
            }
            
            # Get visual representation
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Store the image for comparison
            page_dict['image_obj'] = img
            
            # Convert to base64 for web display
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            page_dict['images'].append(base64.b64encode(img_byte_arr).decode())
            
            result.append(page_dict)
        
        return result

    def compare_pdfs(self, pdf1_path, pdf2_path):
        """Compare PDFs and return text differences, visual differences, and AI analysis."""
        # Extract content from both PDFs
        pdf1_content = self.extract_text_and_images(pdf1_path)
        pdf2_content = self.extract_text_and_images(pdf2_path)
        
        # Compare texts for all PDFs (removed text-based check)
        # Extract words from both PDFs
        words1 = self.extract_words("\n".join(page['text'] for page in pdf1_content))
        words2 = self.extract_words("\n".join(page['text'] for page in pdf2_content))
        
        # Compare words and generate HTML diff
        text_diff = self.compare_words(words1, words2)
        
        # Compare images and highlight differences
        original_images = []
        modified_images = []
        all_changes = []
        max_pages = max(len(pdf1_content), len(pdf2_content))
        
        for i in range(max_pages):
            if i < len(pdf1_content) and i < len(pdf2_content):
                # Compare corresponding pages
                original_img, modified_img, changes = self.compare_images(
                    pdf1_content[i]['image_obj'],
                    pdf2_content[i]['image_obj']
                )
                all_changes.extend(changes)
                
                # Convert original to base64
                img_byte_arr = io.BytesIO()
                original_img.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                original_images.append(base64.b64encode(img_byte_arr).decode())
                
                # Convert modified to base64
                img_byte_arr = io.BytesIO()
                modified_img.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                modified_images.append(base64.b64encode(img_byte_arr).decode())
            elif i < len(pdf1_content):
                # Page was removed
                img_byte_arr = io.BytesIO()
                pdf1_content[i]['image_obj'].save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                original_images.append(base64.b64encode(img_byte_arr).decode())
                all_changes.append({'type': 'deletion', 'area': 'full_page'})
            else:
                # New page added
                img_byte_arr = io.BytesIO()
                pdf2_content[i]['image_obj'].save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                modified_images.append(base64.b64encode(img_byte_arr).decode())
                all_changes.append({'type': 'addition', 'area': 'full_page'})
        
        # Get LLM analysis if available
        llm_analysis = ""
        if self.llm and (text_diff.strip() or all_changes):
            try:
                drawing_changes = self.analyze_drawing_changes(all_changes)
                chain = self.prompt | self.llm
                response = chain.invoke({
                    "differences": text_diff or "No text changes detected.",
                    "drawing_changes": drawing_changes
                })
                llm_analysis = response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                llm_analysis = f"Error generating AI analysis: {str(e)}"
        
        return {
            'text_diff': text_diff,
            'pdf1_images': original_images,
            'pdf2_images': modified_images,
            'llm_analysis': llm_analysis
        }

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    if 'pdf1' not in request.files or 'pdf2' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    pdf1 = request.files['pdf1']
    pdf2 = request.files['pdf2']
    api_key = request.form.get('api_key', '')
    
    if not pdf1.filename or not pdf2.filename:
        return jsonify({'error': 'No files selected'}), 400
    
    if not allowed_file(pdf1.filename) or not allowed_file(pdf2.filename):
        return jsonify({'error': 'Invalid file type. Please upload PDF files only'}), 400
    
    try:
        # Save files temporarily
        pdf1_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(pdf1.filename))
        pdf2_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(pdf2.filename))
        
        pdf1.save(pdf1_path)
        pdf2.save(pdf2_path)
        
        # Compare PDFs
        comparator = PDFComparator()
        result = comparator.compare_pdfs(pdf1_path, pdf2_path)
        
        # Clean up
        os.remove(pdf1_path)
        os.remove(pdf2_path)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 
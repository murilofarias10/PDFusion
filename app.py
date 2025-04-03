from flask import Flask, request, render_template, redirect, url_for, send_from_directory, session, jsonify
import os
import fitz  # PyMuPDF
from PIL import Image
import uuid
import difflib
import openai
import re
import json
import io
import pytesseract
from flask import g
import time
import threading
from datetime import datetime, timedelta
diff_store = {}

app = Flask(__name__)
app.secret_key = 'pdfusion-secret'
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
IMAGES_FOLDER = os.path.join(OUTPUT_FOLDER, 'images')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(IMAGES_FOLDER, exist_ok=True)

# Maximum age of files to keep (in hours)
MAX_FILE_AGE_HOURS = 2

# Configure pytesseract path if needed (example for Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

openai.api_key = os.getenv("OPENAI_API_KEY")

def cleanup_old_files():
    """
    Clean up old files in the uploads and output directories.
    This runs in a background thread every 30 minutes.
    """
    while True:
        try:
            # Calculate cutoff time (files older than this will be deleted)
            cutoff_time = datetime.now() - timedelta(hours=MAX_FILE_AGE_HOURS)
            
            # Clean uploads folder
            uploads_cleaned = clean_directory(UPLOAD_FOLDER, cutoff_time)
            
            # Clean output folder
            output_cleaned = clean_directory(OUTPUT_FOLDER, cutoff_time)
            
            # Clean images subfolder
            images_cleaned = clean_directory(IMAGES_FOLDER, cutoff_time)
            
            # Log summary
            total_cleaned = uploads_cleaned + output_cleaned + images_cleaned
            if total_cleaned > 0:
                print(f"Scheduled cleanup at {datetime.now()}: removed {total_cleaned} files ({uploads_cleaned} uploads, {output_cleaned} output files, {images_cleaned} images)")
            else:
                print(f"Scheduled cleanup at {datetime.now()}: no files needed cleaning")
                
        except Exception as e:
            print(f"Error during scheduled cleanup: {e}")
        
        # Sleep for 30 minutes before next cleanup
        time.sleep(30 * 60)

def clean_directory(directory, cutoff_time):
    """
    Remove files in the specified directory that are older than the cutoff time.
    
    Args:
        directory (str): Directory to clean
        cutoff_time (datetime): Files older than this will be deleted
    """
    if not os.path.exists(directory):
        return
        
    file_count = 0
    deleted_count = 0
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        # Skip directories
        if os.path.isdir(filepath) and filepath != IMAGES_FOLDER:
            continue
            
        file_count += 1
        # Check file modification time
        file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
        
        # Different cleanup rules based on file type
        should_delete = False
        
        # More aggressive cleanup for text files (30 minutes older)
        if filepath.lower().endswith('.txt'):
            text_cutoff = cutoff_time + timedelta(minutes=30)  # 30 minutes more recent
            should_delete = file_mtime < text_cutoff
        else:
            # Standard cleanup for non-text files
            should_delete = file_mtime < cutoff_time
        
        # Delete if file meets deletion criteria
        if should_delete:
            try:
                if os.path.isfile(filepath):
                    os.remove(filepath)
                    deleted_count += 1
                    print(f"Deleted old file: {filepath}")
            except Exception as e:
                print(f"Error deleting {filepath}: {e}")
    
    if deleted_count > 0:
        print(f"Cleaned {deleted_count}/{file_count} files from {directory}")
        
    return deleted_count

# Start the cleanup thread when the app starts
cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
cleanup_thread.start()

def render_pdf_to_image(pdf_path, page_num=0, zoom=2):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img

def extract_text_from_pdf(pdf_path, page_num=None):
    doc = fitz.open(pdf_path)
    text = ""
    
    if page_num is not None:
        # Extract text from specific page
        if 0 <= page_num < doc.page_count:
            text = doc[page_num].get_text()
    else:
        # Extract text from all pages
        for page in doc:
            text += page.get_text()
    
    doc.close()
    return text.strip()

def highlight_differences(img1, img2):
    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")
    if img1.size != img2.size:
        img2 = img2.resize(img1.size)

    highlight_a = img1.copy()
    highlight_b = img2.copy()

    px1 = img1.load()
    px2 = img2.load()
    ha = highlight_a.load()
    hb = highlight_b.load()

    for y in range(img1.height):
        for x in range(img1.width):
            if px1[x, y] != px2[x, y]:
                if px1[x, y] != (255, 255, 255) and px2[x, y] == (255, 255, 255):
                    ha[x, y] = (255, 0, 0)
                elif px1[x, y] == (255, 255, 255) and px2[x, y] != (255, 255, 255):
                    hb[x, y] = (0, 255, 0)

    return highlight_a, highlight_b

def diff_texts(text1, text2):
    differ = difflib.Differ()
    
    # Convert to lowercase for comparison to ignore case, but keep original words for display
    words1 = text1.split()
    words2 = text2.split()
    
    # Create a mapping of lowercase words to original words
    lowercase_to_original1 = {word.lower(): word for word in words1}
    lowercase_to_original2 = {word.lower(): word for word in words2}
    
    # Compare the lowercase versions
    diff = list(differ.compare([w.lower() for w in words1], [w.lower() for w in words2]))
    
    text_a = []
    text_b = []

    for word in diff:
        if word.startswith("- "):
            # Use the original case for display
            original_word = lowercase_to_original1.get(word[2:], word[2:])
            text_a.append(f"<span class='removed'>{original_word}</span>")
        elif word.startswith("+ "):
            # Use the original case for display
            original_word = lowercase_to_original2.get(word[2:], word[2:])
            text_b.append(f"<span class='added'>{original_word}</span>")
        elif word.startswith("  "):
            # For unchanged words, use the original from the first document
            original_word1 = lowercase_to_original1.get(word[2:], word[2:])
            original_word2 = lowercase_to_original2.get(word[2:], word[2:])
            text_a.append(original_word1)
            text_b.append(original_word2)

    return ' '.join(text_a), ' '.join(text_b)

def is_drawing_pdf(text):
    plain_text = re.sub(r'[^a-zA-Z\n\s]', '', text)
    word_count = len(plain_text.split())
    avg_word_length = sum(len(word) for word in plain_text.split()) / word_count if word_count else 0

    if word_count < 30 or avg_word_length < 4:
        return True

    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You are an assistant. Determine if the following PDF content is a drawing or a text document.\n"
                        "If it's just symbols or technical lines like '² X X X X X X X X X', then it's a drawing.\n"
                        "If it's normal language like 'this is important for humanity', then it's text.\n"
                        "Return only 'drawing' or 'text'.\n\n"
                        f"Content:\n{text[:1000]}"
                    )
                }
            ]
        )
        result = response.choices[0].message.content.strip().lower()
        return 'drawing' in result
    except Exception as e:
        print("LLM analysis failed:", e)
        return True

def extract_images_from_pdf(pdf_path, prefix="pdf"):
    """
    Extract images from a PDF file, save them to disk, and perform OCR on them.
    
    Args:
        pdf_path (str): Path to the PDF file
        prefix (str): Prefix for the output image filenames
        
    Returns:
        list: List of dictionaries containing image information
    """
    if not os.path.exists(pdf_path):
        return []
        
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    
    # Extract images
    images_list = []
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Generate a unique filename
            img_filename = f"{prefix}_page{page_num}_img{img_index}_{uuid.uuid4().hex[:8]}.png"
            img_path = os.path.join(IMAGES_FOLDER, img_filename)
            
            # Save the image
            image.save(img_path)
            
            # Perform OCR on the image
            try:
                img_text = pytesseract.image_to_string(image)
                # Save OCR text to file if text was extracted
                if img_text.strip():
                    text_filename = f"{prefix}_page{page_num}_img{img_index}_{uuid.uuid4().hex[:8]}_ocr.txt"
                    text_path = os.path.join(OUTPUT_FOLDER, text_filename)
                    with open(text_path, "w", encoding="utf-8") as f:
                        f.write(img_text)
                else:
                    text_filename = None
            except Exception as e:
                print(f"OCR failed for image {img_index} on page {page_num}: {e}")
                text_filename = None
                img_text = ""
            
            # Add image info to the list
            images_list.append({
                "page_num": page_num,
                "img_index": img_index,
                "width": base_image["width"],
                "height": base_image["height"],
                "filename": img_filename,
                "path": img_path,
                "ocr_text": img_text,
                "ocr_filename": text_filename
            })
    
    pdf_document.close()
    return images_list

def save_extracted_content(text_content, file_prefix, output_dir=OUTPUT_FOLDER):
    """
    Save the extracted text content to a file.
    
    Args:
        text_content (str): Extracted text content
        file_prefix (str): Prefix for the output file name
        output_dir (str): Directory to save the output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save text content
    filename = f"{file_prefix}_{uuid.uuid4().hex[:8]}.txt"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text_content)
    
    return filename

@app.route('/')
def index():
    # Clean up user files from the current session
    if 'user_files' in session:
        for file_path in session.get('user_files', []):
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Cleaned up user file: {file_path}")
            except Exception as e:
                print(f"Error removing file {file_path}: {e}")
    
    # Also perform an immediate cleanup of any old files
    try:
        # More aggressive cleanup on page load - 15 minutes for all files and even more aggressive for text
        cutoff_time = datetime.now() - timedelta(minutes=15)
        
        uploads_cleaned = clean_directory(UPLOAD_FOLDER, cutoff_time)
        output_cleaned = clean_directory(OUTPUT_FOLDER, cutoff_time)
        images_cleaned = clean_directory(IMAGES_FOLDER, cutoff_time)
        
        total_cleaned = uploads_cleaned + output_cleaned + images_cleaned
        if total_cleaned > 0:
            print(f"Index page cleanup: removed {total_cleaned} total old files")
            
    except Exception as e:
        print(f"Error during immediate cleanup: {e}")
    
    session.clear()  # Clear the session
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file1 = request.files['pdf1']
    file2 = request.files['pdf2']
    
    # No longer need document type selection, we'll process both text and images
    
    file1_path = os.path.join(UPLOAD_FOLDER, f"pdf1_{uuid.uuid4().hex}.pdf")
    file2_path = os.path.join(UPLOAD_FOLDER, f"pdf2_{uuid.uuid4().hex}.pdf")

    file1.save(file1_path)
    file2.save(file2_path)

    session['file1_path'] = file1_path
    session['file2_path'] = file2_path
    
    # Initialize list to track files for this user session
    session['user_files'] = [file1_path, file2_path]

    return redirect(url_for('processing'))

@app.route('/processing')
def processing():
    return render_template('processing.html')

@app.route('/processing/done')
def processing_done():
    path1 = session.get('file1_path')
    path2 = session.get('file2_path')

    if not path1 or not path2:
        return redirect(url_for('index'))

    # Get total pages from both PDFs
    doc1 = fitz.open(path1)
    doc2 = fitz.open(path2)
    total_pages = max(doc1.page_count, doc2.page_count)
    doc1.close()
    doc2.close()

    # Extract text from PDFs
    text1 = extract_text_from_pdf(path1)
    text2 = extract_text_from_pdf(path2)
    
    # Extract images from PDFs and perform OCR
    images1 = extract_images_from_pdf(path1, prefix="pdf1")
    images2 = extract_images_from_pdf(path2, prefix="pdf2")
    
    # Combine extracted text with OCR text from images
    for img in images1:
        if img["ocr_text"].strip():
            text1 += f"\n\n[OCR from image on page {img['page_num'] + 1}]:\n{img['ocr_text']}"
    
    for img in images2:
        if img["ocr_text"].strip():
            text2 += f"\n\n[OCR from image on page {img['page_num'] + 1}]:\n{img['ocr_text']}"
    
    # Save the extracted text to files
    text1_filename = save_extracted_content(text1, "pdf1_text")
    text2_filename = save_extracted_content(text2, "pdf2_text")
    session['text1_filename'] = text1_filename
    session['text2_filename'] = text2_filename
    
    # Track generated files for cleanup
    user_files = session.get('user_files', [])
    user_files.append(os.path.join(OUTPUT_FOLDER, text1_filename))
    user_files.append(os.path.join(OUTPUT_FOLDER, text2_filename))
    session['user_files'] = user_files
    
    # Store image information in session
    session['images1'] = [{'filename': img['filename'], 'page_num': img['page_num']} for img in images1]
    session['images2'] = [{'filename': img['filename'], 'page_num': img['page_num']} for img in images2]
    
    # Add image paths to user_files for cleanup
    for img in images1 + images2:
        user_files.append(img['path'])
        if img.get('ocr_filename'):
            user_files.append(os.path.join(OUTPUT_FOLDER, img['ocr_filename']))
    
    # Process text differences
    diff_text_a, diff_text_b = diff_texts(text1, text2)
    diff_text_a_filename = save_extracted_content(diff_text_a, "pdf1_diff_text")
    diff_text_b_filename = save_extracted_content(diff_text_b, "pdf2_diff_text")
    session['diff_text_a_filename'] = diff_text_a_filename
    session['diff_text_b_filename'] = diff_text_b_filename
    
    # Track diff files
    user_files.append(os.path.join(OUTPUT_FOLDER, diff_text_a_filename))
    user_files.append(os.path.join(OUTPUT_FOLDER, diff_text_b_filename))
    
    # Process visual differences for first page by default
    show_visual = True
    img1 = render_pdf_to_image(path1, page_num=0)
    img2 = render_pdf_to_image(path2, page_num=0)
    highlight_a, highlight_b = highlight_differences(img1, img2)
    
    highlight_a_path = os.path.join(OUTPUT_FOLDER, "pdf_a_highlighted.png")
    highlight_b_path = os.path.join(OUTPUT_FOLDER, "pdf_b_highlighted.png")
    
    highlight_a.save(highlight_a_path)
    highlight_b.save(highlight_b_path)
    
    # Track highlighted images
    user_files.append(highlight_a_path)
    user_files.append(highlight_b_path)
    session['user_files'] = user_files
    
    # Remove AI analysis - no longer needed
    
    # Pass the required data to the template without the analysis
    return render_template(
        "results.html",
        show_visual=show_visual,
        show_text=True,  # Always show text differences
        diff_ready=True,
        diff_text_a=diff_text_a,
        diff_text_b=diff_text_b,
        text1_filename=text1_filename,
        text2_filename=text2_filename,
        diff_text_a_filename=diff_text_a_filename,
        diff_text_b_filename=diff_text_b_filename,
        images1=session['images1'],
        images2=session['images2'],
        image_count1=len(images1),
        image_count2=len(images2),
        total_pages=total_pages
    )

@app.route('/image/<filename>')
def get_image(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/get_page/<int:page_num>')
def get_page(page_num):
    path1 = session.get('file1_path')
    path2 = session.get('file2_path')

    if not path1 or not path2:
        return jsonify({'success': False, 'error': 'No PDF files found'})

    try:
        # Generate images for the requested page
        img1 = render_pdf_to_image(path1, page_num=page_num-1)
        img2 = render_pdf_to_image(path2, page_num=page_num-1)
        
        # Generate unique filenames for this page
        filename1 = f"pdf_a_page_{page_num}.png"
        filename2 = f"pdf_b_page_{page_num}.png"
        
        # Save the images
        img1_path = os.path.join(OUTPUT_FOLDER, filename1)
        img2_path = os.path.join(OUTPUT_FOLDER, filename2)
        
        img1.save(img1_path)
        img2.save(img2_path)
        
        # Track these files for cleanup
        user_files = session.get('user_files', [])
        user_files.extend([img1_path, img2_path])
        
        # Generate highlighted versions
        highlight_a, highlight_b = highlight_differences(img1, img2)
        highlight_a_path = os.path.join(OUTPUT_FOLDER, f"highlighted_{filename1}")
        highlight_b_path = os.path.join(OUTPUT_FOLDER, f"highlighted_{filename2}")
        
        highlight_a.save(highlight_a_path)
        highlight_b.save(highlight_b_path)
        
        # Track highlighted files
        user_files.extend([highlight_a_path, highlight_b_path])
        session['user_files'] = user_files
        
        return jsonify({
            'success': True,
            'img1_url': url_for('get_image', filename=f"highlighted_{filename1}"),
            'img2_url': url_for('get_image', filename=f"highlighted_{filename2}")
        })
    except Exception as e:
        print(f"Error generating page {page_num}:", e)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_text_page/<int:page_num>')
def get_text_page(page_num):
    path1 = session.get('file1_path')
    path2 = session.get('file2_path')

    if not path1 or not path2:
        return jsonify({'success': False, 'error': 'No PDF files found'})

    try:
        # Get text from the specific page (0-based index)
        text1 = extract_text_from_pdf(path1, page_num - 1)
        text2 = extract_text_from_pdf(path2, page_num - 1)
        
        # Generate diff for this page
        diff_text_a, diff_text_b = diff_texts(text1, text2)
        
        # Save text for download if needed
        page_diff_a_filename = save_extracted_content(diff_text_a, f"pdf1_page{page_num}_diff_text")
        page_diff_b_filename = save_extracted_content(diff_text_b, f"pdf2_page{page_num}_diff_text")
        
        # Track these files for cleanup
        user_files = session.get('user_files', [])
        user_files.append(os.path.join(OUTPUT_FOLDER, page_diff_a_filename))
        user_files.append(os.path.join(OUTPUT_FOLDER, page_diff_b_filename))
        session['user_files'] = user_files
        
        return jsonify({
            'success': True,
            'text_a': diff_text_a,
            'text_b': diff_text_b
        })
    except Exception as e:
        print(f"Error generating text page {page_num}:", e)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download/<filename>')
def download_file(filename):
    """Download a file from the output folder"""
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

@app.route('/cleanup')
def cleanup():
    """API endpoint to clean up files when user leaves the page"""
    try:
        files_cleaned = 0
        
        # Clean up user files from the current session
        if 'user_files' in session:
            for file_path in session.get('user_files', []):
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        files_cleaned += 1
                except Exception as e:
                    print(f"Error removing file {file_path}: {e}")
            
            # Clear the user_files list from session
            session.pop('user_files', None)
        
        # Also cleanup any old .txt files that might be left
        cutoff_time = datetime.now() - timedelta(minutes=10)  # Very aggressive cleanup
        txt_cleaned = clean_directory(OUTPUT_FOLDER, cutoff_time)
        
        return jsonify({
            'success': True, 
            'cleaned': files_cleaned + txt_cleaned
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

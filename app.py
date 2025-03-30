import os
import tempfile
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename
import difflib
import cv2
import numpy as np
from PIL import Image
import io
from openai import OpenAI
import base64
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
import traceback  # For better error debugging
import json
import time
import threading
import uuid
import datetime
from io import BytesIO
from tempfile import NamedTemporaryFile

app = Flask(__name__)
app.secret_key = 'pdfsecretkey2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour

# Create uploads directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Global dictionaries to store processing status and results
processing_status = {}
session_results = {}

# Simple in-memory storage for demonstration
uploaded_files = {}

class PDFComparator:
    def __init__(self):
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def extract_pdf_content(self, pdf_path):
        """Extract text and images from a PDF file using PyPDF2 and PyMuPDF"""
        text_content = ""
        image_paths = []
        page_images = []
        page_texts = []  # Store text for each page separately
        
        try:
            # Extract text using PyPDF2
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                page_text = page.extract_text()
                page_texts.append(page_text)  # Add text for this page
                text_content += page_text + "\n"
            
            # Extract images using PyMuPDF (fitz)
            doc = fitz.open(pdf_path)
            for page_index, page in enumerate(doc):
                # Get page as image for visual comparison
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_images.append(img)
                
                # Save the page image
                page_filename = f"{os.path.basename(pdf_path)}_page{page_index+1}.png"
                page_path = os.path.join(app.config['UPLOAD_FOLDER'], page_filename)
                img.save(page_path)
                image_paths.append(page_path)
                
                # Extract embedded images if needed
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Save the embedded image
                    img_filename = f"{os.path.basename(pdf_path)}_page{page_index+1}_img{img_index+1}.png"
                    img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
                    
                    with open(img_path, "wb") as img_file:
                        img_file.write(image_bytes)
            
            return text_content, image_paths, page_images, page_texts
            
        except Exception as e:
            print(f"Error extracting PDF content: {str(e)}")
            traceback.print_exc()  # Print full traceback for debugging
            return "", [], [], []
    
    def extract_words(self, text):
        """Extract words from text, handling various separators and special characters."""
        # Remove special characters and normalize spaces
        text = text.lower()
        text = ''.join(c for c in text if c.isalnum() or c.isspace())
        # Split into words and filter out empty strings
        words = [word for word in text.split() if word]
        return words
    
    def compare_words(self, words1, words2):
        """Compare two lists of words and return differences with HTML formatting."""
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
    
    def compare_text(self, texts1, texts2, per_page=False):
        """Compare two text documents using difflib
        
        If per_page is True, texts1 and texts2 should be lists of page texts.
        Returns page-by-page comparison if per_page is True.
        """
        if not per_page:
            # Traditional line-by-line comparison for the whole document
            text1 = "\n".join(texts1) if isinstance(texts1, list) else texts1
            text2 = "\n".join(texts2) if isinstance(texts2, list) else texts2
            
            text1_lines = text1.splitlines()
            text2_lines = text2.splitlines()
            
            # Generate diff
            diff = difflib.HtmlDiff().make_file(text1_lines, text2_lines, 'PDF 1', 'PDF 2')
            
            # Word-level comparison
            words1 = self.extract_words(text1)
            words2 = self.extract_words(text2)
            word_diff = self.compare_words(words1, words2)
            
            return diff, [word_diff]  # Return as a single-item list for consistency
        else:
            # Page-by-page comparison
            diffs = []
            word_diffs = []
            
            # Make sure we have matching arrays
            min_pages = min(len(texts1), len(texts2))
            
            for i in range(min_pages):
                page_text1 = texts1[i]
                page_text2 = texts2[i]
                
                # Skip processing if either page is empty
                if not page_text1.strip() and not page_text2.strip():
                    word_diffs.append("<p>Page is empty or contains no text that can be extracted.</p>")
                    continue
                    
                text1_lines = page_text1.splitlines()
                text2_lines = page_text2.splitlines()
                
                # Generate diff for this page
                diff = difflib.HtmlDiff().make_file(text1_lines, text2_lines, f'PDF 1 - Page {i+1}', f'PDF 2 - Page {i+1}')
                diffs.append(diff)
                
                # Word-level comparison for this page
                words1 = self.extract_words(page_text1)
                words2 = self.extract_words(page_text2)
                
                # First use heuristics to detect trivial differences
                if self.is_meaningful_diff(words1, words2):
                    # For borderline cases or complex content, use LLM
                    if len(words1) > 50 and abs(len(words1) - len(words2)) > 10:
                        # Only use LLM for non-trivial cases to save API calls
                        if not self.analyze_diff_with_llm(page_text1, page_text2):
                            word_diff = f"<p>Page {i+1}: No significant text differences found that would be meaningful to display.</p>"
                            word_diffs.append(word_diff)
                            continue
                    
                    word_diff = self.compare_words(words1, words2)
                else:
                    word_diff = f"<p>Page {i+1}: No significant text differences found that would be meaningful to display.</p>"
                
                word_diffs.append(word_diff)
            
            return diffs, word_diffs
            
    def is_meaningful_diff(self, words1, words2):
        """Use heuristics and LLM to determine if a difference is meaningful to show."""
        # Quick heuristic check - if lengths are very similar and only a few words different
        if abs(len(words1) - len(words2)) < 5 and min(len(words1), len(words2)) > 10:
            # Count differences
            matcher = difflib.SequenceMatcher(None, words1, words2)
            diff_count = sum(1 for tag, _, _, _, _ in matcher.get_opcodes() if tag != 'equal')
            
            # If the only differences are a few instances of the same word (like 'x', 'the', etc.)
            if diff_count <= 5:
                # Get the actual different words
                different_words = []
                diff_types = {'replace': 0, 'delete': 0, 'insert': 0}
                
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag != 'equal':
                        diff_types[tag] += 1
                        different_words.extend(words1[i1:i2])
                        different_words.extend(words2[j1:j2])
                
                # Count word frequency to detect repeat occurrences
                word_freq = {}
                for word in different_words:
                    word_freq[word] = word_freq.get(word, 0) + 1
                
                # Check for repeated differences of the same word
                repeated_diffs = sum(1 for word, count in word_freq.items() if count > 1)
                max_freq = max(word_freq.values()) if word_freq else 0
                
                # Check if they're trivial differences
                trivial_words = [w for w in different_words if len(w) <= 2 or w.lower() in 
                                ['the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'and', 'or', 'for']]
                
                trivial_percentage = len(trivial_words) / len(different_words) if different_words else 0
                
                # Multiple instances of the same trivial word (like 'x') being added/removed
                if max_freq > 2 and repeated_diffs > 0 and trivial_percentage > 0.3:
                    return False
                    
                # Few trivial changes
                if trivial_percentage > 0.7:
                    return False
                
                # Small difference in one-letter abbreviations/symbols
                special_chars = [w for w in different_words if len(w) == 1 and not w.isalnum()]
                if len(special_chars) > len(different_words) * 0.5:
                    return False
                
                # Check if changes are spread across the text or concentrated in one area
                # If many small changes across a large document, they might be insignificant
                if diff_count > 3 and diff_count < len(words1) * 0.05:
                    diff_positions = []
                    for tag, i1, i2, _, _ in matcher.get_opcodes():
                        if tag != 'equal':
                            diff_positions.extend(range(i1, i2))
                    
                    # If changes are scattered widely, they may be insignificant
                    if diff_positions and (max(diff_positions) - min(diff_positions)) > len(words1) * 0.8:
                        return trivial_percentage < 0.4  # Show only if changes are not trivial
        
        # If we get here, we'll consider the difference meaningful
        return True
    
    def analyze_drawing_changes(self, changes):
        """Analyze the drawing changes and return a detailed description."""
        if not changes:
            return "No changes detected in drawings."
        
        total_changes = len(changes)
        additions = sum(1 for c in changes if c['type'] == 'addition')
        deletions = sum(1 for c in changes if c['type'] == 'deletion')
        
        # Categorize changes by size 
        small_changes = sum(1 for c in changes if 10 < c['area'] <= 100)
        medium_changes = sum(1 for c in changes if 100 < c['area'] <= 500)
        large_changes = sum(1 for c in changes if c['area'] > 500)
        
        # Group changes by position for more detailed reporting
        positions = []
        position_map = {
            'top_left': {'count': 0, 'types': {'addition': 0, 'deletion': 0}},
            'top_right': {'count': 0, 'types': {'addition': 0, 'deletion': 0}},
            'bottom_left': {'count': 0, 'types': {'addition': 0, 'deletion': 0}},
            'bottom_right': {'count': 0, 'types': {'addition': 0, 'deletion': 0}},
            'center': {'count': 0, 'types': {'addition': 0, 'deletion': 0}}
        }
        
        # Define significant changes (changes with larger areas)
        significant_changes = []
        
        for change in changes:
            if change.get('position'):
                x, y, w, h = change['position']
                area = change['area']
                change_type = change['type']
                
                # Determine position in the document
                # Image is divided into 5 regions: top-left, top-right, bottom-left, bottom-right, center
                horizontal = "left" if x < 400 else "right"
                vertical = "top" if y < 400 else "bottom"
                
                # Center area determination
                if 200 <= x <= 600 and 200 <= y <= 600:
                    position = "center"
                else:
                    position = f"{vertical}_{horizontal}"
                
                # Update position map
                position_map[position]['count'] += 1
                position_map[position]['types'][change_type] += 1
                
                # Record significant changes
                if area > 100:  # Only report significant changes
                    significant_changes.append({
                        'position': position,
                        'type': change_type,
                        'area': area,
                        'coords': (x, y, w, h)
                    })
        
        # Generate position descriptions
        position_descriptions = []
        for pos, data in position_map.items():
            if data['count'] > 0:
                pos_name = pos.replace('_', ' ')
                additions_in_pos = data['types']['addition']
                deletions_in_pos = data['types']['deletion']
                
                if additions_in_pos > 0 and deletions_in_pos > 0:
                    position_descriptions.append(
                        f"{data['count']} changes in {pos_name} area ({additions_in_pos} additions, {deletions_in_pos} deletions)"
                    )
                elif additions_in_pos > 0:
                    position_descriptions.append(
                        f"{additions_in_pos} additions in {pos_name} area"
                    )
                elif deletions_in_pos > 0:
                    position_descriptions.append(
                        f"{deletions_in_pos} deletions in {pos_name} area"
                    )
        
        # Highlight most significant changes
        significant_change_descriptions = []
        sorted_significant = sorted(significant_changes, key=lambda x: x['area'], reverse=True)
        
        for i, change in enumerate(sorted_significant[:5]):  # Top 5 most significant changes
            pos = change['position'].replace('_', ' ')
            change_type = "added" if change['type'] == 'addition' else "removed"
            size_desc = "large" if change['area'] > 500 else "medium"
            significant_change_descriptions.append(
                f"{size_desc} content {change_type} in {pos} area"
            )
        
        # Build the final analysis
        analysis = f"Found {total_changes} changes in the drawings: {additions} additions (shown in green) and {deletions} deletions (shown in red)."
        
        # Add size classification if we have different sizes
        if small_changes or medium_changes or large_changes:
            size_details = []
            if large_changes:
                size_details.append(f"{large_changes} large")
            if medium_changes:
                size_details.append(f"{medium_changes} medium")
            if small_changes:
                size_details.append(f"{small_changes} small")
            
            analysis += f"\nChange sizes: {', '.join(size_details)}."
            
        # Add position information
        if position_descriptions:
            analysis += "\nLocations: " + "; ".join(position_descriptions[:4])
            if len(position_descriptions) > 4:
                analysis += f" and {len(position_descriptions) - 4} more location groups"
        
        # Add significant change details
        if significant_change_descriptions:
            analysis += "\nMost significant changes: " + "; ".join(significant_change_descriptions)
        
        return analysis
    
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

        # Calculate the percentage of difference
        difference_percentage = (np.count_nonzero(thresh) / thresh.size) * 100

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

        return result_original_pil, result_modified_pil, changes, difference_percentage
    
    def is_technical_drawing(self, image):
        """Detect if an image is likely to be a technical drawing or diagram."""
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                np_img = np.array(image)
            else:
                np_img = image
                
            # Ensure image is in RGB format
            if len(np_img.shape) == 2:
                np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
                
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
            
            # Enhance image contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Apply Canny edge detection with improved parameters
            edges = cv2.Canny(enhanced, 50, 150)
            
            # Apply morphological operations to connect broken lines
            kernel = np.ones((2, 2), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Count edge pixels and calculate edge percentage
            edge_percentage = (np.count_nonzero(edges) / edges.size) * 100
            
            # Calculate intensity histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()
            
            # Calculate statistical features
            white_percentage = np.sum(hist[200:]) * 100  # Percentage of white/bright pixels
            black_percentage = np.sum(hist[:50]) * 100   # Percentage of black/dark pixels
            mid_tones = np.sum(hist[50:200]) * 100       # Percentage of mid-tones
            
            # Detect peaks in histogram for bi/tri-modal distribution (common in drawings)
            peaks = np.sum(hist > np.mean(hist) * 2)
            
            # Calculate straight line features using Hough transform
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=50, maxLineGap=10)
            straight_line_count = 0 if lines is None else len(lines)
            
            # Calculate the ratio of horizontal to vertical lines (technical drawings often have more horizontal/vertical lines)
            horizontal_lines = 0
            vertical_lines = 0
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    if angle < 10 or angle > 170:  # Nearly horizontal lines
                        horizontal_lines += 1
                    elif 80 < angle < 100:  # Nearly vertical lines
                        vertical_lines += 1
            
            orthogonal_line_ratio = (horizontal_lines + vertical_lines) / max(1, straight_line_count)
            
            # Additional structure detection: count circles (common in technical drawings)
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                param1=50, param2=30, minRadius=5, maxRadius=60
            )
            circle_count = 0 if circles is None else circles.shape[1]
            
            # Decision logic with improved criteria
            is_drawing = False
            
            # Strong indicators of technical drawings
            if (edge_percentage > 5 and white_percentage > 60 and black_percentage > 5 and peaks < 10):
                is_drawing = True
            
            # Straight line features common in diagrams and technical drawings
            if straight_line_count > 20 and orthogonal_line_ratio > 0.5:
                is_drawing = True
                
            # Architectural or engineering drawings often have a lot of circles
            if circle_count > 5:
                is_drawing = True
                
            # Text-heavy pages typically have more mid-tones and fewer strong edges
            if mid_tones > 50 and edge_percentage < 8:
                is_drawing = False
            
            # Very high white percentage with minimal black could be a mostly blank page
            if white_percentage > 90 and black_percentage < 3:
                is_drawing = False
                
            return is_drawing
            
        except Exception as e:
            print(f"Error in is_technical_drawing: {str(e)}")
            traceback.print_exc()  # More detailed error logging
            return False
    
    def get_llm_analysis(self, text_diff, image_diff_percentages, drawing_changes=None):
        """Get analysis from OpenAI about the differences in tabular format"""
        drawing_analysis = ""
        if drawing_changes:
            drawing_analysis = self.analyze_drawing_changes(drawing_changes)
            
        prompt = f"""
        Analyze the following differences between two PDF documents:
        
        Text differences summary:
        {text_diff[:500]}... (truncated)
        
        Image differences:
        {image_diff_percentages}
        
        {"Drawing analysis: " + drawing_analysis if drawing_analysis else ""}
        
        Provide a structured summary of the key differences between these documents in an HTML table format.
        Create a table with the following columns:
        1. Category (Text, Images, Drawings)
        2. Type of Change (Addition, Deletion, Modification)
        3. Description (concise description of the change)
        4. Significance (High/Medium/Low based on your assessment)
        
        Use <table class="table table-striped">, <thead>, <tbody>, <tr>, <th>, and <td> tags for formatting.
        For example:
        <table class="table table-striped">
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Type of Change</th>
                    <th>Description</th>
                    <th>Significance</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Text</td>
                    <td>Addition</td>
                    <td>New paragraph about project timeline added</td>
                    <td>High</td>
                </tr>
                <!-- more rows -->
            </tbody>
        </table>
        
        After the table, add a brief executive summary paragraph.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an assistant that analyzes differences between PDF documents, including both text and technical drawings."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error getting LLM analysis: {str(e)}"

    def assess_text_quality(self, text_diff):
        """Use LLM to determine if the text comparison is meaningful."""
        try:
            # First try simple heuristics to avoid unnecessary API calls
            # Strip HTML tags for basic analysis
            stripped_text = text_diff.replace('<span class="removed-word">', '')
            stripped_text = stripped_text.replace('<span class="added-word">', '')
            stripped_text = stripped_text.replace('<span class="unchanged-word">', '')
            stripped_text = stripped_text.replace('</span>', '')
            
            # Check for repetitive patterns that suggest garbage OCR
            if len(stripped_text) < 50:
                return False
                
            # Count character frequencies
            char_counts = {}
            for char in stripped_text[:300]:  # Check first 300 chars
                char_counts[char] = char_counts.get(char, 0) + 1
                    
            # If a few characters dominate, likely not meaningful text
            total_chars = sum(char_counts.values())
            top_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            top_chars_percentage = sum(count for _, count in top_chars) / total_chars
            
            if top_chars_percentage > 0.8:  # If top 5 chars make up >80% of text, likely garbage
                return False
                
            # Check for common words that should appear in meaningful text
            common_words = ['the', 'a', 'of', 'and', 'to', 'in', 'that', 'is', 'for', 'with']
            text_lower = stripped_text.lower()
            common_word_count = sum(1 for word in common_words if word in text_lower)
            
            # If very few common words are found, text is likely not meaningful
            if common_word_count < 2 and len(stripped_text) > 200:
                return False
            
            # If we're still unsure, use the LLM to analyze the text
            # Sample a portion of the text to check for meaningfulness
            text_sample = stripped_text[:800] if len(stripped_text) > 800 else stripped_text
            
            prompt = f"""
            Analyze this text which was extracted from a PDF comparison:
            
            "{text_sample}"
            
            Does this text appear to be meaningful human language text (like paragraphs, sentences, etc.), 
            or does it look like garbage text/OCR errors/random characters?
            
            Please analyze and respond with only one of these exact responses:
            - MEANINGFUL: If the text contains actual human language content with real words and sentences
            - GARBAGE: If the text appears to be mostly random characters, OCR errors, or non-linguistic content
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an assistant that analyzes text quality. Answer with exactly MEANINGFUL or GARBAGE based on whether the text contains real human language or just random characters/OCR errors."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50
            )
            
            result = response.choices[0].message.content.strip().upper()
            
            # Return true if it contains the word "MEANINGFUL", false otherwise
            return "MEANINGFUL" in result
            
        except Exception as e:
            print(f"Error in assess_text_quality: {str(e)}")
            traceback.print_exc()
            return True  # Default to showing text if error occurs

    def analyze_diff_with_llm(self, text1, text2):
        """Use LLM to determine if differences between two texts are meaningful enough to display."""
        try:
            # Prepare a sample of both texts
            sample1 = text1[:300] if len(text1) > 300 else text1
            sample2 = text2[:300] if len(text2) > 300 else text2
            
            prompt = f"""
            Compare these two text extracts from PDF documents:
            
            TEXT 1:
            {sample1}
            
            TEXT 2:
            {sample2}
            
            Are there meaningful differences that would be helpful to show to a user? 
            Consider the following factors:
            1. If the only differences are formatting, spacing, or punctuation, these are NOT meaningful.
            2. If there are multiple instances of the same trivial word (like 'x' or 'a') being added or removed, these are NOT meaningful.
            3. If there appears to be OCR errors or garbled text in both, highlighting differences is NOT meaningful.
            4. If the text appears to be mostly random characters, line breaks, or non-linguistic content, the differences are NOT meaningful.
            5. If there are actual content changes like different words, phrases, numbers, or sentences, these ARE meaningful.
            
            ONLY respond with one of these values:
            - MEANINGFUL: If the differences reflect actual content changes worth showing to the user
            - NOT_MEANINGFUL: If the differences are trivial, formatting-related, or the text is too garbled to make sense
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You analyze PDF text differences and determine if they are meaningful."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=20,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip().upper()
            
            return "MEANINGFUL" in result
            
        except Exception as e:
            print(f"Error in analyze_diff_with_llm: {str(e)}")
            traceback.print_exc()
            return True  # Default to showing differences if error occurs

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'file1' not in request.files or 'file2' not in request.files:
        flash('Both files are required', 'error')
        return redirect(request.url)
    
    file1 = request.files['file1']
    file2 = request.files['file2']
    
    if file1.filename == '' or file2.filename == '':
        flash('Both files are required', 'error')
        return redirect(request.url)
    
    # Create a unique session ID
    session_id = str(uuid.uuid4())
    
    # Initialize processing status
    processing_status[session_id] = {
        'step': 'upload',
        'message': 'Files uploaded successfully',
        'progress': 10
    }
    
    try:
        # Save files
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)
        
        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename1}")
        filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename2}")
        
        file1.save(filepath1)
        file2.save(filepath2)
        
        # Save file paths in session
        session['filepath1'] = filepath1
        session['filepath2'] = filepath2
        
        # Update processing status
        processing_status[session_id] = {
            'step': 'save',
            'message': 'Files saved successfully!',
            'progress': 20
        }
        
        # Redirect to processing page
        return render_template('processing.html', session_id=session_id)
        
    except Exception as e:
        traceback.print_exc()
        flash(f'Error processing files: {str(e)}', 'error')
        return redirect(request.url)

@app.route('/process_pdfs/<session_id>')
def process_pdfs(session_id):
    if session_id not in processing_status:
        return jsonify({'status': 'error', 'message': 'Invalid session ID'})
    
    try:
        # Use a global variable for development to access session data
        filepath1 = session.get('filepath1')
        filepath2 = session.get('filepath2')
        
        print(f"Processing PDFs: {filepath1}, {filepath2}")
        
        if not filepath1 or not filepath2:
            return jsonify({'status': 'error', 'message': 'File paths not found in session'})
        
        # Start PDF comparison in the background
        processing_thread = threading.Thread(
            target=compare_pdfs_async, 
            args=(session_id, filepath1, filepath2)
        )
        processing_thread.daemon = True
        processing_thread.start()
        
        # Update processing status to indicate the thread has started
        processing_status[session_id]['message'] = 'PDF processing started'
        
        # Immediately start extraction
        processing_status[session_id] = {
            'step': 'extract',
            'message': 'Beginning PDF content extraction...',
            'progress': 25
        }
        
        return jsonify({'status': 'started', 'message': 'PDF processing started'})
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})

def compare_pdfs_async(session_id, filepath1, filepath2):
    try:
        # Initialize the comparator
        comparator = PDFComparator()
        
        # Update the processing status
        processing_status[session_id] = {
            'step': 'extract',
            'message': 'Extracting PDF content...',
            'progress': 20
        }

        # Step 1: Extract text from both PDFs
        text1, images1, page_images1, page_texts1 = comparator.extract_pdf_content(filepath1)
        text2, images2, page_images2, page_texts2 = comparator.extract_pdf_content(filepath2)

        # Update the processing status
        processing_status[session_id] = {
            'step': 'compare_text',
            'message': 'Comparing text content...',
            'progress': 40
        }

        # Step 2: Compare text content
        text_diffs, word_diffs = comparator.compare_text(page_texts1, page_texts2, per_page=True)
        
        # Determine if text comparison is meaningful
        is_text_meaningful = comparator.assess_text_quality(word_diffs[0])
        
        # Update the processing status
        processing_status[session_id] = {
            'step': 'compare_images',
            'message': 'Comparing visual content...',
            'progress': 60
        }

        # Step 3: Compare images in a more efficient way
        image_diffs = []
        image_diff_percentages = []
        original_images = []
        modified_images = []
        diff_images = []
        
        # Compare page by page
        min_pages = min(len(page_images1), len(page_images2))
        all_drawing_changes = []
        
        for i in range(min_pages):
            processing_status[session_id]['message'] = f'Comparing page {i+1} of {min_pages}...'
            processing_status[session_id]['progress'] = 60 + (i / min_pages * 20)
            
            # Detect if the page is likely a technical drawing
            is_drawing = comparator.is_technical_drawing(page_images1[i])
            
            # Compare page images and get highlighted versions
            result_original, result_modified, changes, diff_percentage = comparator.compare_images(
                page_images1[i], 
                page_images2[i]
            )
            
            if is_drawing and changes:
                all_drawing_changes.extend(changes)
            
            # Convert images to base64 for embedding in HTML - reduce quality for faster processing
            buffered = BytesIO()
            result_original.save(buffered, format="JPEG", quality=75, optimize=True)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            diff_images.append(img_str)
            
            buffered = BytesIO()
            result_modified.save(buffered, format="JPEG", quality=75, optimize=True)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            original_images.append(img_str)
            
            buffered = BytesIO()
            result_modified.save(buffered, format="JPEG", quality=75, optimize=True)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            modified_images.append(img_str)
            
            image_diffs.append(changes)
            image_diff_percentages.append(f"Page {i+1}: {diff_percentage:.2f}% different")
        
        # Update the processing status
        processing_status[session_id] = {
            'step': 'analyze',
            'message': 'Analyzing differences with AI...',
            'progress': 80
        }

        # Step 4: AI analysis of differences
        llm_analysis = comparator.get_llm_analysis(
            word_diffs[0], 
            "\n".join(image_diff_percentages),
            all_drawing_changes if all_drawing_changes else None
        )

        # Update the processing status
        processing_status[session_id] = {
            'step': 'render',
            'message': 'Rendering results...',
            'progress': 95
        }

        # Prepare original pages as base64 for side-by-side view (original versions without highlights)
        original_pages = []
        modified_pages = []
        
        for i in range(min_pages):
            buffered = BytesIO()
            page_images1[i].save(buffered, format="JPEG", quality=75, optimize=True)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            original_pages.append(img_str)
            
            buffered = BytesIO()
            page_images2[i].save(buffered, format="JPEG", quality=75, optimize=True)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            modified_pages.append(img_str)

        # Store results in session
        session_results[session_id] = {
            'text_diffs': text_diffs,
            'word_diffs': word_diffs,
            'image_diffs': image_diffs,
            'image_diff_percentages': image_diff_percentages,
            'llm_analysis': llm_analysis,
            'original_highlighted': original_images,
            'modified_highlighted': modified_images,
            'diff_images': diff_images,
            'original_pages': original_pages,
            'modified_pages': modified_pages,
            'page_count': min_pages,
            'text_page_count': len(word_diffs),
            'show_text_comparison': is_text_meaningful
        }
        
        # Update the processing status
        processing_status[session_id] = {
            'step': 'completed',
            'message': 'Processing completed successfully!',
            'progress': 100
        }

    except Exception as e:
        traceback.print_exc()
        processing_status[session_id] = {
            'step': 'error',
            'message': f'Error processing PDFs: {str(e)}',
            'progress': 0
        }

@app.route('/check_processing_status/<session_id>')
def check_processing_status(session_id):
    if session_id not in processing_status:
        return jsonify({'status': 'error', 'message': 'Invalid session ID'})
    
    status = processing_status[session_id]
    print(f"Status check: {status}")
    
    # Check if processing is complete
    if status.get('step') in ['completed', 'complete']:
        return jsonify({
            'status': 'completed',
            'message': 'Processing completed successfully',
            'step': status.get('step'),
            'redirect': url_for('results', session_id=session_id)
        })
    
    # Check if there was an error
    if status.get('step') == 'error':
        return jsonify({
            'status': 'error',
            'message': status.get('message', 'An error occurred'),
            'step': 'error'
        })
    
    # Processing is still ongoing
    return jsonify({
        'status': 'processing',
        'message': status.get('message', 'Processing in progress...'),
        'step': status.get('step', 'unknown'),
        'progress': status.get('progress', 0)
    })

@app.route('/results/<session_id>')
def results(session_id):
    if session_id not in session_results:
        flash('Results not found. Please try again.', 'error')
        return redirect(url_for('index'))
    
    comparison_results = session_results[session_id]
    return render_template('results.html', **comparison_results)

def cleanup_old_files(age_hours=24):
    """Remove files older than the specified number of hours from the uploads directory."""
    try:
        now = datetime.datetime.now()
        count = 0
        
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Get file's last modification time
            file_mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
            age = now - file_mod_time
            
            # If file is older than the specified age, delete it
            if age.total_seconds() > age_hours * 3600:
                os.remove(file_path)
                count += 1
                
        print(f"Cleanup complete: {count} old files removed")
        return count
    except Exception as e:
        print(f"Error during cleanup: {e}")
        traceback.print_exc()
        return 0

def run_scheduled_cleanup():
    """Run cleanup every hour in a background thread."""
    while True:
        cleanup_old_files()
        # Sleep for 1 hour
        time.sleep(3600)

# Start cleanup thread when app initializes
cleanup_thread = threading.Thread(target=run_scheduled_cleanup)
cleanup_thread.daemon = True
cleanup_thread.start()

if __name__ == '__main__':
    app.run(debug=True) 
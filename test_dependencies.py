#!/usr/bin/env python
"""
Test script to verify that critical dependencies for PDFusion are working correctly.
Run this script with: python test_dependencies.py
"""

import sys
import os

def test_opencv():
    print("Testing OpenCV...")
    try:
        import cv2
        import numpy as np
        
        # Create a simple image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:50, :50] = [0, 0, 255]  # Red square
        
        # Apply some basic operations
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        
        print(f"✓ OpenCV {cv2.__version__} is working correctly")
        return True
    except Exception as e:
        print(f"✗ OpenCV test failed: {str(e)}")
        return False

def test_pymupdf():
    print("Testing PyMuPDF...")
    try:
        import fitz
        
        # Get version
        version = fitz.version
        
        # Try to create a simple PDF
        doc = fitz.open()
        page = doc.new_page()
        
        # Add some text
        text_point = fitz.Point(50, 50)
        page.insert_text(text_point, "PyMuPDF Test", fontsize=12)
        
        print(f"✓ PyMuPDF {version} is working correctly")
        return True
    except Exception as e:
        print(f"✗ PyMuPDF test failed: {str(e)}")
        return False

def test_pil():
    print("Testing PIL/Pillow...")
    try:
        from PIL import Image, ImageDraw
        
        # Create a simple image
        img = Image.new('RGB', (100, 100), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([(10, 10), (90, 90)], fill=(255, 0, 0))
        
        print(f"✓ Pillow {Image.__version__} is working correctly")
        return True
    except Exception as e:
        print(f"✗ PIL/Pillow test failed: {str(e)}")
        return False

def test_flask():
    print("Testing Flask...")
    try:
        import flask
        
        print(f"✓ Flask {flask.__version__} is working correctly")
        return True
    except Exception as e:
        print(f"✗ Flask test failed: {str(e)}")
        return False

def test_pypdf2():
    print("Testing PyPDF2...")
    try:
        from PyPDF2 import PdfReader, PdfWriter
        
        # Try to create a simple PDF
        writer = PdfWriter()
        page = writer.add_blank_page(width=100, height=100)
        
        print(f"✓ PyPDF2 is working correctly")
        return True
    except Exception as e:
        print(f"✗ PyPDF2 test failed: {str(e)}")
        return False

def check_directories():
    print("Checking required directories...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check templates directory
    templates_dir = os.path.join(base_dir, 'templates')
    if os.path.exists(templates_dir) and os.path.isdir(templates_dir):
        print(f"✓ Templates directory exists: {templates_dir}")
    else:
        print(f"✗ Templates directory missing: {templates_dir}")
    
    # Check static/uploads directory
    uploads_dir = os.path.join(base_dir, 'static', 'uploads')
    if os.path.exists(uploads_dir) and os.path.isdir(uploads_dir):
        print(f"✓ Uploads directory exists: {uploads_dir}")
    else:
        print(f"✗ Uploads directory missing: {uploads_dir}")
        # Try to create it
        try:
            os.makedirs(uploads_dir, exist_ok=True)
            print(f"  Created uploads directory: {uploads_dir}")
        except Exception as e:
            print(f"  Failed to create uploads directory: {str(e)}")

if __name__ == "__main__":
    print("PDFusion Dependency Test")
    print("=======================")
    
    tests_passed = 0
    tests_total = 5
    
    if test_opencv():
        tests_passed += 1
    
    if test_pymupdf():
        tests_passed += 1
    
    if test_pil():
        tests_passed += 1
    
    if test_flask():
        tests_passed += 1
    
    if test_pypdf2():
        tests_passed += 1
    
    check_directories()
    
    print("\nTest Summary")
    print("===========")
    print(f"Passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("\nAll dependencies are working correctly! PDFusion should work properly.")
        sys.exit(0)
    else:
        print("\nSome dependencies have issues. Please fix them before running PDFusion.")
        sys.exit(1) 
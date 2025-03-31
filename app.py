from flask import Flask, request, render_template, redirect, url_for, send_from_directory, session
import os
import fitz  # PyMuPDF
from PIL import Image
import uuid
import difflib
import openai
import re

app = Flask(__name__)
app.secret_key = 'pdfusion-secret'
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

openai.api_key = os.getenv("OPENAI_API_KEY")

def render_pdf_to_image(pdf_path, page_num=0, zoom=2):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
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
    diff = list(differ.compare(text1.split(), text2.split()))
    text_a = []
    text_b = []

    for word in diff:
        if word.startswith("- "):
            text_a.append(f"<span class='removed'>{word[2:]}</span>")
        elif word.startswith("+ "):
            text_b.append(f"<span class='added'>{word[2:]}</span>")
        elif word.startswith("  "):
            text_a.append(word[2:])
            text_b.append(word[2:])

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

@app.route('/')
def index():
    session.clear()  # <-- This clears any previous PDF paths or comparison results
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file1 = request.files['pdf1']
    file2 = request.files['pdf2']

    file1_path = os.path.join(UPLOAD_FOLDER, f"pdf1_{uuid.uuid4().hex}.pdf")
    file2_path = os.path.join(UPLOAD_FOLDER, f"pdf2_{uuid.uuid4().hex}.pdf")

    file1.save(file1_path)
    file2.save(file2_path)

    session['file1_path'] = file1_path
    session['file2_path'] = file2_path

    return redirect(url_for('processing'))

@app.route('/processing')
def processing():
    return render_template('processing.html')

@app.route('/process')
def process():
    path1 = session.get('file1_path')
    path2 = session.get('file2_path')

    if not path1 or not path2:
        return redirect(url_for('index'))

    text1 = extract_text_from_pdf(path1)
    text2 = extract_text_from_pdf(path2)
    is_drawing = is_drawing_pdf(text1 + text2)

    show_visual = False
    diff_text_a = ""
    diff_text_b = ""

    if is_drawing:
        show_visual = True
        img1 = render_pdf_to_image(path1)
        img2 = render_pdf_to_image(path2)
        highlight_a, highlight_b = highlight_differences(img1, img2)
        highlight_a.save(os.path.join(OUTPUT_FOLDER, "pdf_a_highlighted.png"))
        highlight_b.save(os.path.join(OUTPUT_FOLDER, "pdf_b_highlighted.png"))
    else:
        diff_text_a, diff_text_b = diff_texts(text1, text2)

    # ✅ Save results in session
    session['show_visual'] = show_visual
    session['diff_text_a'] = diff_text_a
    session['diff_text_b'] = diff_text_b

    # ✅ Redirect to final results page
    return redirect(url_for('done'))

@app.route('/processing/done')
def done():
    return render_template(
        "results.html",
        show_visual=session.get('show_visual', False),
        diff_ready=True,
        diff_text_a=session.get('diff_text_a', ''),
        diff_text_b=session.get('diff_text_b', '')
    )

@app.route('/image/<filename>')
def get_image(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)

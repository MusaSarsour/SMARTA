from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory, flash
import os
import subprocess
from werkzeug.utils import secure_filename
import shutil
import firebase_admin
from firebase_admin import credentials, firestore
import threading

app = Flask(__name__)
app.secret_key = 'supersecretkey'

cred = credentials.Certificate('firebase-key2.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

doc_refer= {"omar_alkadi":"AWg1kV9Y44elp8Kt6dqU"}
doc_name = doc_refer['omar_alkadi']
doc_ref = db.collection("ModelAssests").document(doc_name)
attendance_data = {
    'Students': ['abd', 'musa', 'moh'],
    'Present': 3,
    'Absent': 1,
    'Absent_Rate': 0.25,
    'Attendance_Rate': 0.75
}

UPLOAD_FOLDER = 'uploaded/'
EXTRACTED_IMG_FOLDER = 'comp_img/'
LABELED_IMG_FOLDER = 'labeled_images/'

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXTRACTED_IMG_FOLDER, exist_ok=True)
os.makedirs(LABELED_IMG_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['EXTRACTED_IMG_FOLDER'] = EXTRACTED_IMG_FOLDER
app.config['LABELED_IMG_FOLDER'] = LABELED_IMG_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

def run_scripts(file_path):
    """Function to run processing scripts."""
    try:
        subprocess.run(['python', 'extract2.py', file_path], check=True)
        subprocess.run(['python', 'detect.py', file_path, app.config['LABELED_IMG_FOLDER']], check=True)
        doc_ref.set(attendance_data)
    except subprocess.CalledProcessError as e:
        print("Script failed:", e)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash("No file part in the request", "danger")
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash("No file selected for uploading", "danger")
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Start script execution in a new thread
        thread = threading.Thread(target=run_scripts, args=(file_path,))
        thread.start()

        # Create the full URL for the results page
        full_link = url_for('show_results', filename=filename, _external=True)

        # Create an HTML response that includes the full URL and redirects
        response_html = f'''
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <title>Redirecting...</title>
            <meta http-equiv="refresh" content="0; url={full_link}">
        </head>
        <body>
            <h1>Redirecting...</h1>
            <p>File uploaded successfully, processing started. You should be redirected automatically to the target URL: <a href="{full_link}"></a>. If not, click the link.</p>
        </body>
        </html>
        '''

        return response_html
    else:
        flash("File type not allowed", "danger")
        return redirect(url_for('index'))

@app.route('/results/<filename>')
def show_results(filename):
    extracted_files = [f for f in os.listdir(EXTRACTED_IMG_FOLDER) if allowed_file(f)]
    labeled_files = [f for f in os.listdir(LABELED_IMG_FOLDER) if allowed_file(f)]
    extracted_urls = [url_for('extracted_image', filename=f) for f in extracted_files]
    labeled_urls = [url_for('labeled_image', filename=f) for f in labeled_files]
    return render_template(
        'result.html',
        original_image=url_for('uploaded_file', filename=filename),
        extracted_images=extracted_urls,
        labeled_images=labeled_urls
    )

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/reset', methods=['POST'])
def reset():
    """Delete files from comp_img and labeled_matches folders."""
    try:
        for folder in [app.config['EXTRACTED_IMG_FOLDER'], app.config['LABELED_IMG_FOLDER']]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        flash('Files reset successfully!', 'success')
    except Exception as e:
        flash(f'Error while resetting files: {str(e)}', 'danger')

    # Redirect to home page after reset
    return redirect(url_for('index'))

@app.route('/comp_img/<filename>')
def extracted_image(filename):
    return send_from_directory(EXTRACTED_IMG_FOLDER, filename)

@app.route('/labeled_imgs/<filename>')
def labeled_image(filename):
    return send_from_directory(LABELED_IMG_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

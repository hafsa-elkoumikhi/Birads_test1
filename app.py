from flask import Flask, request, render_template, redirect, url_for, session
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from models import db, Feedback
import pydicom
from PIL import Image
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash


# Initialize the Flask application
app = Flask(__name__)
app.secret_key = 'secret_key'

# Load the trained model
model = load_model('model_bi_aTL.h5')

# Configure the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///feedback.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Define the path for the uploads
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define your User model

class Users(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=False, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    user_type = db.Column(db.String(20), nullable=False)

# Update the signup route to handle user type
@app.route('/sign-up', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        user_type = request.form['user_type']

        existing_user = Users.query.filter_by(email=email).first()
        if existing_user:
            error_message = 'Email already exists. Please use a different email.'
            return render_template('pages/sign-up.html', error_message=error_message)
        
        # Create a new user with the plaintext password
        new_user = Users(username=username, email=email, password=password, user_type=user_type)
        db.session.add(new_user)
        db.session.commit()
        
        return redirect(url_for('index'))
    return render_template('pages/sign-up.html')
# Routes
@app.route('/')
def index():
    return render_template('pages/sign-in.html')

@app.route('/sign-in', methods=['POST'])
def signin():
    email = request.form['email']
    password = request.form['password']

    # Query the database for the user
    user = Users.query.filter_by(email=email, password=password).first()
        
    if user:
        session['username'] = user.username
        return redirect(url_for('dashboard'))
    else:
        error_message = 'Invalid email or password'
        return render_template('pages/sign-in.html', error_message=error_message)


# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to convert DICOM to PNG
def convert_dcm_to_png(dcm_path, png_path):
    # Read the DICOM file
    dicom = pydicom.dcmread(dcm_path)
    
    # Apply modality LUT if available
    if 'ModalityLUTSequence' in dicom:
        modality_lut = dicom.ModalityLUTSequence[0].LUTDescriptor
        pixel_array = dicom.pixel_array * modality_lut.RescaleSlope + modality_lut.RescaleIntercept
    else:
        pixel_array = dicom.pixel_array
    
    # Normalize pixel values to 8-bit depth
    pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255
    pixel_array = pixel_array.astype(np.uint8)
    
    # Convert pixel array to image
    image = Image.fromarray(pixel_array)
    
    # Save the image as PNG
    image.save(png_path)

# Define a function to preprocess the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/classify', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save the file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            print(f"File saved to {file_path}")

            # Check if the file is a DICOM file
            if filename.lower().endswith('.dcm'):
                # Convert DICOM to PNG
                png_filename = filename.replace('.dcm', '.png')
                png_path = os.path.join(app.config['UPLOAD_FOLDER'], png_filename)
                convert_dcm_to_png(file_path, png_path)
                file_path = png_path

            # Preprocess the image
            img_array = preprocess_image(file_path)
            # Make prediction
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)
            # Map the predicted class index to the class name
            class_labels = {0: 'BIRADS 1', 1: 'BIRADS 2', 2: 'BIRADS 3', 3: 'BIRADS 4', 4: 'BIRADS 5'}
            result = class_labels[predicted_class[0]]

            # Get the relative file path
            relative_file_path = os.path.join('uploads', os.path.basename(file_path)).replace("\\", "/")
            print(f"Relative file path: {relative_file_path}")
            return render_template('pages/tables.html', result=result, image_path=relative_file_path)
    return render_template('pages/tables.html')

@app.route('/feedback', methods=['POST'])
def feedback():
    file_path = request.form['file_path']
    prediction = request.form['prediction']
    feedback = request.form['feedback']
    correct_class = request.form.get('correct_class')
    # Store feedback in the database
    feedback_entry = Feedback(file_path=file_path, prediction=prediction, feedback=feedback, correct_class=correct_class)
    db.session.add(feedback_entry)
    db.session.commit()
    return redirect(url_for('upload_file'))

@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        return render_template('pages/dashboard.html', username=session['username'])
    else:
        return redirect(url_for('index'))
    

@app.route('/classify')
def classify():
    return render_template('pages/tables.html')

@app.route('/logout')
def logout():
    # Remove username from the session if it exists
    session.pop('username', None)
    # Redirect to login page or wherever you want after logout
    return redirect(url_for('index'))


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

import os
import logging
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, current_user, login_user, logout_user, login_required
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('medai.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize extensions
db = SQLAlchemy()
login_manager = LoginManager()

# Configuration
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'super-secret-key'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///medai.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/uploads')
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models/final_model.keras')
    CLASS_INDICES_PATH = os.path.join(os.path.dirname(__file__), 'models/class_indices.npy')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    DEFAULT_CLASSES = {
        0: 'COVID-19',
        1: 'Normal', 
        2: 'Pneumonia',
        3: 'Tuberculosis'
    }

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    role = db.Column(db.String(20), default='radiologist')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    patients = db.relationship('Patient', backref='doctor', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(20), unique=True)
    full_name = db.Column(db.String(150))
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    medical_history = db.Column(db.Text)
    scans = db.relationship('Scan', backref='patient', lazy=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'))

class Scan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200))
    prediction = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    probabilities = db.Column(db.JSON)
    clinical_notes = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'))

# Application Factory
def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'login'

    # Database initialization command
    @app.cli.command("init-db")
    def init_db():
        """Initialize the database"""
        with app.app_context():
            db.create_all()
            print("Database initialized")

    # Configure user loader
    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    # Clinical notes generator
    def generate_clinical_notes(diagnosis, confidence):
        notes = {
            'COVID-19': f"COVID-19 detected with {confidence:.1%} confidence. Recommendations include: RT-PCR confirmation, chest CT scan, and antiviral treatment protocol. Patient isolation required.",
            'Pneumonia': f"Bacterial pneumonia identified ({confidence:.1%} confidence). Recommended: Sputum culture, antibiotic therapy, and oxygenation monitoring.",
            'Tuberculosis': f"Tuberculosis manifestations observed ({confidence:.1%} confidence). Require: AFB culture, long-term antibiotic regimen, and contact tracing.",
            'Normal': f"No abnormalities detected ({confidence:.1%} confidence). Routine follow-up recommended."
        }
        return notes.get(diagnosis, "Clinical evaluation recommended.")

    # Load AI model
    with app.app_context():
        try:
            app.logger.info("Initializing AI model...")
            app.model = load_model(app.config['MODEL_PATH'])
            
            # Load class indices
            class_indices = app.config['DEFAULT_CLASSES']
            if os.path.exists(app.config['CLASS_INDICES_PATH']):
                try:
                    loaded = np.load(app.config['CLASS_INDICES_PATH'], allow_pickle=True)
                    class_indices = loaded.item() if isinstance(loaded, np.ndarray) else loaded
                except Exception as e:
                    app.logger.warning(f"Using default classes: {str(e)}")
            
            app.class_indices = class_indices
            app.logger.info(f"Loaded class indices: {class_indices}")
            
        except Exception as e:
            app.logger.error(f"Model loading failed: {str(e)}")
            raise RuntimeError("Failed to initialize core components") from e

    # Authentication routes
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
            
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            user = User.query.filter_by(username=username).first()
            
            if user and user.check_password(password):
                login_user(user)
                return redirect(url_for('dashboard'))
            flash('Invalid username or password', 'danger')
            
        return render_template('login.html')

    @app.route('/logout')
    @login_required
    def logout():
        logout_user()
        return redirect(url_for('login'))

    # Patient management
    @app.route('/add-patient', methods=['POST'])
    @login_required
    def add_patient():
        try:
            new_patient = Patient(
                patient_id=request.form['patient_id'],
                full_name=request.form['full_name'],
                age=request.form['age'],
                gender=request.form['gender'],
                medical_history=request.form.get('medical_history', ''),
                doctor_id=current_user.id
            )
            db.session.add(new_patient)
            db.session.commit()
            flash('Patient added successfully', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'Error adding patient: {str(e)}', 'danger')
        return redirect(url_for('dashboard'))

    # Scan management
    @app.route('/analyze', methods=['POST'])
    @login_required
    def analyze_scan():
        try:
            if 'scan' not in request.files:
                flash('No file selected', 'danger')
                return redirect(url_for('dashboard'))
            
            file = request.files['scan']
            if file.filename == '':
                flash('No selected file', 'danger')
                return redirect(url_for('dashboard'))
            
            if file and allowed_file(file.filename):
                # Create upload directory if not exists
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                
                # Save file
                filename = secure_filename(f"{datetime.now().timestamp()}_{file.filename}")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Verify and process image
                try:
                    img = Image.open(filepath).convert('RGB')
                    img = img.resize((224, 224))
                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                except Exception as e:
                    flash('Invalid image file', 'danger')
                    return redirect(url_for('dashboard'))
                
                # Make prediction
                try:
                    predictions = app.model.predict(img_array, verbose=0)[0]
                except Exception as e:
                    app.logger.error(f"Prediction error: {str(e)}")
                    flash('Error analyzing scan', 'danger')
                    return redirect(url_for('dashboard'))
                
                # Get class information
                class_indices = app.class_indices
                if not isinstance(class_indices, dict):
                    class_indices = app.config['DEFAULT_CLASSES']
                
                # Process results
                pred_class = np.argmax(predictions)
                confidence = float(np.max(predictions))
                diagnosis = class_indices.get(pred_class, 'Unknown')
                
                # Store probabilities with actual class names
                probabilities = {
                    class_indices.get(i, f'Class_{i}'): float(prob)
                    for i, prob in enumerate(predictions)
                }
                
                # Save to database
                new_scan = Scan(
                    filename=filename,
                    prediction=diagnosis,
                    confidence=confidence,
                    probabilities=probabilities,
                    clinical_notes=generate_clinical_notes(diagnosis, confidence),
                    patient_id=request.form.get('patient_id')
                )
                db.session.add(new_scan)
                db.session.commit()
                
                return redirect(url_for('scan_details', scan_id=new_scan.id))
            
            flash('Invalid file type', 'danger')
            return redirect(url_for('dashboard'))
        
        except Exception as e:
            app.logger.error(f"Analysis error: {str(e)}", exc_info=True)
            flash('Error processing scan', 'danger')
            return redirect(url_for('dashboard'))

    @app.route('/scan/<int:scan_id>')
    @login_required
    def scan_details(scan_id):
        scan = Scan.query.get_or_404(scan_id)
        return render_template('scan_details.html', scan=scan)

    # Main dashboard
    @app.route('/')
    @login_required
    def dashboard():
        recent_scans = Scan.query.order_by(Scan.timestamp.desc()).limit(5).all()
        patients = Patient.query.filter_by(doctor_id=current_user.id).all()
        return render_template('dashboard.html', 
                             recent_scans=recent_scans,
                             patients=patients)

    # Helper functions
    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000)
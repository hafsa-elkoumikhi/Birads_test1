from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    file_path = db.Column(db.String(256), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    feedback = db.Column(db.String(10), nullable=False)
    correct_class = db.Column(db.String(50))

    def __init__(self, file_path, prediction, feedback, correct_class=None):
        self.file_path = file_path
        self.prediction = prediction
        self.feedback = feedback
        self.correct_class = correct_class

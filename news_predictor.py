import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.metrics import accuracy_score
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QTextEdit, QPushButton, QLabel, QProgressBar, QDialog, QMessageBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QFont
import threading
import os
from PyQt6.QtGui import QFont, QTextOption
# Create a signal class for thread communication
class ModelSignals(QObject):
    accuracy_ready = pyqtSignal(float)
    error_occurred = pyqtSignal(str)
    model_loaded = pyqtSignal()

class ModernTheme:
    """Modern color theme and style definitions with softer accents"""
    BG_WHITE = "#ffffff"
    BG_LIGHT_GRAY = "#f5f5f5"
    BG_GRAY = "#e0e0e0"
    ACCENT_SOFT_BLUE = "#4a90e2"  # Softer blue for a more subdued look
    TEXT_GRAY = "#666666"
    TEXT_DARK = "#333333"

class WarningDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Warning")
        self.setFixedSize(400, 200)
        self.setStyleSheet(f"background-color: {ModernTheme.BG_WHITE};")

        # Layout for the dialog
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Warning message
        self.warning_label = QLabel("Please enter text to analyze.")
        self.warning_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.warning_label.setStyleSheet(f"color: {ModernTheme.ACCENT_SOFT_BLUE};")
        self.warning_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.warning_label)

        # OK button
        self.ok_button = QPushButton("OK")
        self.ok_button.setFont(QFont("Arial", 12))
        self.ok_button.setStyleSheet(f"""
            background-color: {ModernTheme.BG_GRAY};
            color: {ModernTheme.TEXT_DARK};
            border: 1px solid {ModernTheme.BG_GRAY};
            padding: 10px;
            min-width: 100px;
        """)
        self.ok_button.clicked.connect(self.accept)
        layout.addWidget(self.ok_button, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setLayout(layout)

class AboutMeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About Me")
        self.setFixedSize(500, 300)
        self.setStyleSheet(f"background-color: {ModernTheme.BG_WHITE};")

        # Layout for the dialog
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # About Me information
        self.about_label = QLabel("""My name is Aryan Mohammadi, an NLP and Machine Learning developer and a Computer Applications student at HAMK University in Finland. I specialize in AI, Natural Language Processing (NLP), and backend development, working with technologies like Python, Node.js, and React.js. This project is part of my thesis, titled "Automated News Categorization Using Small Language Models, NLP, and Pre-Trained Models." It focuses on leveraging NLP techniques and Small Language models to efficiently categorize news articles, providing a scalable and accurate solution for news analysis and classification.""")
        self.about_label.setFont(QFont("Arial", 12))
        self.about_label.setStyleSheet(f"""
            color: {ModernTheme.TEXT_DARK};
            text-align: justify;  
            qproperty-alignment: AlignLeft;  
        """)
        self.about_label.setAlignment(Qt.AlignmentFlag.AlignJustify | Qt.AlignmentFlag.AlignLeft)  
        self.about_label.setWordWrap(True)
        layout.addWidget(self.about_label)

        # Return button
        self.return_button = QPushButton("Return to Main Page")
        self.return_button.setFont(QFont("Arial", 12))
        self.return_button.setStyleSheet(f"""
            background-color: {ModernTheme.BG_GRAY};
            color: {ModernTheme.TEXT_DARK};
            border: 1px solid {ModernTheme.BG_GRAY};
            padding: 10px;
            min-width: 150px;
        """)
        self.return_button.clicked.connect(self.accept)
        layout.addWidget(self.return_button, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setLayout(layout)

class NewsCategoryPredictor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Predict")
        self.setGeometry(100, 100, 900, 700)
        self.setStyleSheet(f"background-color: {ModernTheme.BG_WHITE};")
        
        # Setup signals for thread communication
        self.signals = ModelSignals()
        self.signals.accuracy_ready.connect(self.update_accuracy_display)
        self.signals.error_occurred.connect(self.show_error)
        self.signals.model_loaded.connect(self.enable_predict_button)

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(20)

        # Header with About Me button
        self.header_frame = QWidget()
        header_layout = QHBoxLayout(self.header_frame)

        self.header_label = QLabel("Predict")
        self.header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.header_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.header_label.setStyleSheet(f"color: {ModernTheme.TEXT_DARK};")
        header_layout.addWidget(self.header_label)

        # About Me button (top right)
        self.about_button = QPushButton("About Me")
        self.about_button.setFont(QFont("Arial", 12))
        self.about_button.setStyleSheet(f"""
            background-color: {ModernTheme.BG_GRAY};
            color: {ModernTheme.TEXT_DARK};
            border: 1px solid {ModernTheme.BG_GRAY};
            padding: 5px 15px;
        """)
        self.about_button.clicked.connect(self.show_about)
        header_layout.addStretch()  # Push the About Me button to the right
        header_layout.addWidget(self.about_button)

        self.layout.addWidget(self.header_frame)

        # Input section
        self.input_frame = QWidget()
        input_layout = QVBoxLayout(self.input_frame)

        self.input_label = QLabel("Enter News Content")
        self.input_label.setFont(QFont("Arial", 12))
        self.input_label.setStyleSheet(f"color: {ModernTheme.TEXT_GRAY};")
        input_layout.addWidget(self.input_label)

        self.text_input = QTextEdit()
        self.text_input.setFont(QFont("Arial", 13))  # Slightly larger font for readability
        self.text_input.setMinimumHeight(250)  # Set a minimum height to show more text
        self.text_input.setAcceptRichText(False)  # Plain text only for simplicity
        self.text_input.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)  # Enable word wrapping
        self.text_input.setWordWrapMode(QTextOption.WrapMode.WordWrap)  # Ensure proper wrapping
        self.text_input.setPlaceholderText("Paste your news article here...")
        self.text_input.setStyleSheet(f"""
            background-color: {ModernTheme.BG_WHITE};
            color: {ModernTheme.TEXT_DARK};
            border: 1px solid {ModernTheme.BG_GRAY};
            border-radius: 5px;  /* Softer corners */
            padding: 15px;       /* More padding for comfort */
        """)
        input_layout.addWidget(self.text_input)

        self.layout.addWidget(self.input_frame, stretch=1)  # Add stretch to give more space to input
        self.text_input.setPlaceholderText("Paste your news article here...")
        input_layout.addWidget(self.text_input)

        self.layout.addWidget(self.input_frame)

        # Button frame for Predict, Restart, and Exit
        self.button_frame = QWidget()
        button_layout = QHBoxLayout(self.button_frame)
        button_layout.setSpacing(10)

        # Predict button
        self.predict_button = QPushButton("Predict")
        self.predict_button.setFont(QFont("Arial", 12))
        self.predict_button.setStyleSheet(f"""
            background-color: {ModernTheme.BG_GRAY};
            color: {ModernTheme.TEXT_DARK};
            border: 1px solid {ModernTheme.BG_GRAY};
            padding: 10px;
            min-width: 100px;
        """)
        self.predict_button.clicked.connect(self.predict)
        self.predict_button.setEnabled(False)  # Disabled until model is loaded
        button_layout.addWidget(self.predict_button)

        # Restart button
        self.restart_button = QPushButton("Restart")
        self.restart_button.setFont(QFont("Arial", 12))
        self.restart_button.setStyleSheet(f"""
            background-color: {ModernTheme.BG_GRAY};
            color: {ModernTheme.TEXT_DARK};
            border: 1px solid {ModernTheme.BG_GRAY};
            padding: 10px;
            min-width: 100px;
        """)
        self.restart_button.clicked.connect(self.restart)
        button_layout.addWidget(self.restart_button)

        # Exit button
        self.exit_button = QPushButton("Exit")
        self.exit_button.setFont(QFont("Arial", 12))
        self.exit_button.setStyleSheet(f"""
            background-color: {ModernTheme.BG_GRAY};
            color: {ModernTheme.TEXT_DARK};
            border: 1px solid {ModernTheme.BG_GRAY};
            padding: 10px;
            min-width: 100px;
        """)
        self.exit_button.clicked.connect(self.close)  # Close the application
        button_layout.addWidget(self.exit_button)

        self.layout.addWidget(self.button_frame, alignment=Qt.AlignmentFlag.AlignCenter)

                # Progress bar with accuracy label (combined into a single white box with light curved border)
        self.accuracy_frame = QWidget()
        accuracy_layout = QVBoxLayout(self.accuracy_frame)
        accuracy_layout.setContentsMargins(0, 0, 0, 0)
        accuracy_layout.setSpacing(5)  # Reduce spacing for a compact look

        # Style the accuracy frame (white background, light curved border)
        self.accuracy_frame.setStyleSheet("""
            background-color: #FFFFFF;  /* White background */
            border: 1px solid #E0E0E0;  /* Light gray border for a subtle look */
            border-radius: 8px;         /* Light curved border */
            padding: 10px;             /* Small padding for spacing */
        """)

        # "Prediction Accuracy" label at the top, aligned left
        self.accuracy_title_label = QLabel("Prediction Accuracy")
        self.accuracy_title_label.setFont(QFont("Arial", 12))
        self.accuracy_title_label.setStyleSheet("""
            color: #333333;  /* Dark gray text for readability */
            qproperty-alignment: AlignLeft;  /* Align left as requested */
        """)
        accuracy_layout.addWidget(self.accuracy_title_label)

        # Horizontal layout for progress bar and percentage (single row in the box)
        accuracy_row = QWidget()
        accuracy_row_layout = QHBoxLayout(accuracy_row)
        accuracy_row_layout.setContentsMargins(0, 0, 0, 0)
        accuracy_row_layout.setSpacing(10)  # Small spacing between bar and text

        # Progress bar (gray background, black fill, small height)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)  # Initialize to 0
        self.progress_bar.setTextVisible(False)  # Hide the percentage text
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #E0E0E0;  /* Gray background for the bar */
                border: 1px solid #E0E0E0;  /* Gray border to match */
                border-radius: 3px;         /* Slight rounding for a clean look */
                height: 8px;               /* Small height for the bar */
                min-width: 150px;          /* Ensure some minimum width */
            }
            QProgressBar::chunk {
                background-color: #000000;  /* Black fill for the progress */
                border-radius: 3px;
            }
        """)
        accuracy_row_layout.addWidget(self.progress_bar)

        # Accuracy percentage label (black text, right of the bar)
        self.accuracy_value_label = QLabel("Loading model...")
        self.accuracy_value_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.accuracy_value_label.setStyleSheet("""
            color: #000000;  /* Black text */
            min-width: 80px;  /* Ensure enough width for the percentage */
            qproperty-alignment: AlignLeft;  /* Align left for consistency */
        """)
        accuracy_row_layout.addWidget(self.accuracy_value_label)

        accuracy_layout.addWidget(accuracy_row)

        self.layout.addWidget(self.accuracy_frame)

        # Results section
        self.result_frame = QWidget()
        result_layout = QVBoxLayout(self.result_frame)
        self.result_label = QLabel("Classification Result")
        self.result_label.setFont(QFont("Arial", 12))
        self.result_label.setStyleSheet(f"color: {ModernTheme.TEXT_GRAY};")
        result_layout.addWidget(self.result_label)

        self.category_result = QLabel("")
        self.category_result.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.category_result.setStyleSheet(f"""
            color: #28A745;  /* Green text color, matching the image */
            background-color: #E6F9E9;  /* Light green background for the oval */
            border: 1px solid #28A745;  /* Green border to match the text */
            border-radius: 20px;  /* Create an oval shape with larger radius */
            padding: 10px 20px;  /* Padding for horizontal and vertical spacing */
            min-height: 40px;    /* Ensure enough height for the text */
            min-width: 150px;    /* Ensure enough width for the text */
            qproperty-alignment: AlignCenter;  /* Center the text */
        """)
        self.category_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_layout.addWidget(self.category_result)

        self.layout.addWidget(self.result_frame)
        # Footer
        self.footer_label = QLabel("""
            This AI model classifies news articles into different categories with high accuracy.  
            Simply paste your article and click "Predict" to see the results.  
        """)
        self.footer_label.setFont(QFont("Arial", 12))
        self.footer_label.setStyleSheet(f"""
            color: {ModernTheme.TEXT_GRAY};
            qproperty-alignment: AlignLeft;
        """)
        self.footer_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.footer_label.setWordWrap(True)
        self.layout.addWidget(self.footer_label)

        # Model state
        self.model = None
        self.vectorizer = None
        self.is_model_loaded = False
        self.accuracy = 0

        # Start loading model in a separate thread
        self.load_model_thread = threading.Thread(target=self.load_model)
        self.load_model_thread.daemon = True
        self.load_model_thread.start()

    def load_model(self):
        """Load and train the model in a separate thread"""
        try:
            # Add print for debugging
            print("Starting to load model...")
            
            # NLTK requirements
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()

            # Read dataset
            data = pd.read_csv('MNDS_preprocessed.csv')
            print(f"Dataset loaded with {len(data)} rows")

            # Fill NaN values in 'new_content' with an empty string
            X = data['new_content'].fillna('')
            Y = data['category_level_1']

            # Convert text to TF-IDF features
            self.vectorizer = TfidfVectorizer(max_features=5000)
            X_tfidf = self.vectorizer.fit_transform(X)
            print(f"TF-IDF vectorization complete with {X_tfidf.shape[1]} features")

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X_tfidf, Y, test_size=0.2, random_state=42)
            print(f"Data split: training={X_train.shape[0]}, testing={X_test.shape[0]}")

            # Train Logistic Regression model
            self.model = LogisticRegression(random_state=42, max_iter=1000)
            self.model.fit(X_train, y_train)
            print("Model training complete")

            # Evaluate model
            y_pred_lr = self.model.predict(X_test)
            lr_accuracy = accuracy_score(y_test, y_pred_lr)
            self.accuracy = lr_accuracy
            print(f"Model accuracy: {lr_accuracy:.4f}")
            
            # Emit signal with accuracy (as percentage)
            self.signals.accuracy_ready.emit(lr_accuracy * 100)
            
            # Set flag that model is loaded
            self.is_model_loaded = True
            self.signals.model_loaded.emit()
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.signals.error_occurred.emit(f"Model initialization failed: {e}")

    def enable_predict_button(self):
        """Enable the predict button once the model is loaded"""
        self.predict_button.setEnabled(True)

    def update_accuracy_display(self, accuracy_percent):
        """Update the progress bar and accuracy label with the model's accuracy"""
        print(f"Updating accuracy display to {accuracy_percent:.1f}%")
        # Make sure the value is an integer for the progress bar
        self.progress_bar.setValue(int(accuracy_percent))
        # Format the accuracy to one decimal place
        self.accuracy_value_label.setText(f"{accuracy_percent:.1f}%")

    def preprocess_text(self, text):
        """Preprocess the input text"""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        tokens = word_tokenize(text)
        cleaned_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(cleaned_tokens)

    def predict_category(self, text):
        """Predict category for the given text"""
        cleaned_text = self.preprocess_text(text)
        text_tfidf = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(text_tfidf)[0]
        return prediction

    def predict(self):
        """Handle predict button click"""
        if not self.is_model_loaded:
            self.show_warning("Model is still initializing. Please wait.")
            return

        text = self.text_input.toPlainText().strip()
        if text == "Paste your news article here...":
            text = ""

        if not text:
            self.show_warning()  # Show custom warning dialog
            return

        try:
            pred_category = self.predict_category(text)
            self.category_result.setText(pred_category)
        except Exception as e:
            self.show_error(f"Prediction failed: {e}")
            print(f"Error during prediction: {e}")

    def restart(self):
        """Reset the text input and classification result"""
        self.text_input.clear()
        self.text_input.setPlaceholderText("Paste your news article here...")
        self.category_result.setText("")

    def show_warning(self, message=None):
        """Show a custom warning dialog"""
        dialog = WarningDialog(self)
        dialog.exec()

    def show_error(self, message):
        """Show an error message box"""
        QMessageBox.critical(self, "Error", message)

    def show_about(self):
        """Show the About Me dialog"""
        dialog = AboutMeDialog(self)
        dialog.exec()

if __name__ == "__main__":
    # Clear console (for Windows)
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

    app = QApplication(sys.argv)
    window = NewsCategoryPredictor()
    window.show()
    sys.exit(app.exec())
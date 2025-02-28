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
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, font
import threading
import os

class CyberpunkTheme:
    """Cyberpunk color theme and style definitions"""
    # Main colors
    BG_DARK = "#0a0a12"
    BG_MEDIUM = "#13132a"
    BG_LIGHT = "#1a1a3a"
    
    # Accent colors
    NEON_BLUE = "#00f2ff"
    NEON_PINK = "#ff00f2"
    NEON_PURPLE = "#9000ff"
    NEON_GREEN = "#00ff9f"
    NEON_YELLOW = "#ffee00"
    
    # Text colors
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#aaaadd"
    
    # Font definitions
    FONT_MAIN = "Courier New"
    FONT_HEADER = "Arial Black"

class NewsCategoryPredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("CYBERPREDICTOR v1.0")
        self.root.geometry("900x700")
        self.root.configure(bg=CyberpunkTheme.BG_DARK)
        
        # Custom fonts
        self.custom_font = font.Font(family=CyberpunkTheme.FONT_MAIN, size=10)
        self.header_font = font.Font(family=CyberpunkTheme.FONT_HEADER, size=16, weight="bold")
        self.subheader_font = font.Font(family=CyberpunkTheme.FONT_MAIN, size=12, weight="bold")
        self.result_font = font.Font(family=CyberpunkTheme.FONT_MAIN, size=14, weight="bold")
        
        # Configure styles
        self.configure_styles()
        
        # Create the UI
        self.create_ui()
        
        # Model state
        self.model = None
        self.vectorizer = None
        self.is_model_loaded = False
        
        # Start loading model in a separate thread
        self.load_model_thread = threading.Thread(target=self.load_model)
        self.load_model_thread.daemon = True
        self.load_model_thread.start()
        self.progress.start()
        self.check_model_loaded()
        
        # Blinking effect for neon elements
        self.blink_state = False
        self.run_blink_effect()
    
    def configure_styles(self):
        """Configure ttk styles for cyberpunk theme"""
        self.style = ttk.Style()
        
        # Configure frame styles
        self.style.configure("Cyber.TFrame", background=CyberpunkTheme.BG_DARK)
        self.style.configure("CyberInner.TFrame", background=CyberpunkTheme.BG_MEDIUM)
        
        # Configure label styles
        self.style.configure("Cyber.TLabel", 
                            background=CyberpunkTheme.BG_DARK, 
                            foreground=CyberpunkTheme.TEXT_PRIMARY, 
                            font=self.custom_font)
        
        self.style.configure("CyberHeader.TLabel", 
                            background=CyberpunkTheme.BG_DARK, 
                            foreground=CyberpunkTheme.NEON_BLUE, 
                            font=self.header_font)
        
        self.style.configure("CyberResult.TLabel", 
                            background=CyberpunkTheme.BG_MEDIUM, 
                            foreground=CyberpunkTheme.NEON_YELLOW, 
                            font=self.result_font)
        
        self.style.configure("CyberSubHeader.TLabel", 
                            background=CyberpunkTheme.BG_MEDIUM, 
                            foreground=CyberpunkTheme.NEON_GREEN, 
                            font=self.subheader_font)
        
        # Configure button style
        self.style.configure("Cyber.TButton", 
                            font=self.subheader_font)
        
        self.style.map("Cyber.TButton",
                      foreground=[('pressed', CyberpunkTheme.BG_DARK), ('active', CyberpunkTheme.TEXT_PRIMARY)],
                      background=[('pressed', CyberpunkTheme.NEON_GREEN), ('active', CyberpunkTheme.NEON_BLUE)])
        
        # Configure progressbar style
        self.style.configure("Cyber.Horizontal.TProgressbar", 
                            background=CyberpunkTheme.NEON_PURPLE,
                            troughcolor=CyberpunkTheme.BG_LIGHT)
    
    def create_ui(self):
        """Create the UI components"""
        # Main frame
        self.main_frame = ttk.Frame(self.root, style="Cyber.TFrame", padding=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header with animated text effect
        self.header_frame = ttk.Frame(self.main_frame, style="Cyber.TFrame")
        self.header_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.header_label = ttk.Label(self.header_frame, 
                                      text="// NEURAL NEWS ANALYZER //", 
                                      style="CyberHeader.TLabel",
                                      anchor="center")
        self.header_label.pack(fill=tk.X)
        
        self.subheader_label = ttk.Label(self.header_frame, 
                                        text="QUANTUM PREDICTION ENGINE", 
                                        style="Cyber.TLabel",
                                        foreground=CyberpunkTheme.NEON_PINK,
                                        font=self.custom_font,
                                        anchor="center")
        self.subheader_label.pack(fill=tk.X)
        
        # Input section
        self.input_frame = ttk.Frame(self.main_frame, style="CyberInner.TFrame", padding=15)
        self.input_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        self.input_header = ttk.Label(self.input_frame, 
                                     text=">> INPUT DATA STREAM", 
                                     style="CyberSubHeader.TLabel")
        self.input_header.pack(anchor="w", pady=(0, 10))
        
        # Custom text input with cyberpunk styling
        self.text_input = scrolledtext.ScrolledText(
            self.input_frame, 
            wrap=tk.WORD, 
            height=10,
            bg=CyberpunkTheme.BG_LIGHT,
            fg=CyberpunkTheme.TEXT_PRIMARY,
            insertbackground=CyberpunkTheme.NEON_GREEN,  # Cursor color
            font=self.custom_font,
            bd=0,
            padx=10,
            pady=10
        )
        self.text_input.pack(fill=tk.BOTH, expand=True)
        
        # Add placeholder text
        self.text_input.insert("1.0", "Enter news text here for analysis...")
        self.text_input.bind("<FocusIn>", self.clear_placeholder)
        
        # Progress and status section
        self.status_frame = ttk.Frame(self.main_frame, style="Cyber.TFrame")
        self.status_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.progress = ttk.Progressbar(self.status_frame, 
                                       style="Cyber.Horizontal.TProgressbar",
                                       mode="indeterminate")
        self.progress.pack(fill=tk.X)
        
        self.status_label = ttk.Label(self.status_frame, 
                                     text="INITIALIZING NEURAL NETWORK...", 
                                     style="Cyber.TLabel",
                                     foreground=CyberpunkTheme.NEON_PURPLE)
        self.status_label.pack(pady=(5, 0))
        
        # Control section with glowing button
        self.control_frame = ttk.Frame(self.main_frame, style="Cyber.TFrame")
        self.control_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.predict_button = ttk.Button(
            self.control_frame, 
            text="ANALYZE CONTENT", 
            style="Cyber.TButton",
            command=self.predict
        )
        self.predict_button.pack(pady=10)
        
        # Results section with animated border
        self.result_container = tk.Frame(self.main_frame, bg=CyberpunkTheme.NEON_BLUE, padx=2, pady=2)
        self.result_container.pack(fill=tk.BOTH, pady=(0, 10))
        
        self.result_frame = ttk.Frame(self.result_container, style="CyberInner.TFrame", padding=15)
        self.result_frame.pack(fill=tk.BOTH, expand=True)
        
        self.result_header = ttk.Label(self.result_frame, 
                                      text=">> PREDICTION RESULTS", 
                                      style="CyberSubHeader.TLabel")
        self.result_header.pack(anchor="w", pady=(0, 15))
        
        # Category result with cyberpunk styling
        self.category_container = ttk.Frame(self.result_frame, style="CyberInner.TFrame")
        self.category_container.pack(fill=tk.X, pady=(0, 10))
        
        self.category_label = ttk.Label(self.category_container, 
                                       text="CATEGORY:", 
                                       style="Cyber.TLabel",
                                       foreground=CyberpunkTheme.NEON_GREEN)
        self.category_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.category_result = ttk.Label(self.category_container, 
                                        text="AWAITING INPUT", 
                                        style="CyberResult.TLabel")
        self.category_result.pack(side=tk.LEFT)
        
        # Footer with cyber styling
        self.footer_frame = ttk.Frame(self.main_frame, style="Cyber.TFrame")
        self.footer_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.footer_text = ttk.Label(self.footer_frame, 
                                    text="NEURAL ENGINE v2.0 | CYBERSYSTEMS CORP", 
                                    style="Cyber.TLabel",
                                    foreground=CyberpunkTheme.TEXT_SECONDARY,
                                    anchor="center")
        self.footer_text.pack(fill=tk.X)
    
    def clear_placeholder(self, event):
        """Clear placeholder text when input is focused"""
        if self.text_input.get("1.0", "end-1c") == "Enter news text here for analysis...":
            self.text_input.delete("1.0", tk.END)
    
    def run_blink_effect(self):
        """Create a blinking effect for neon elements"""
        # Toggle blink state
        self.blink_state = not self.blink_state
        
        # Apply blinking to header
        if self.blink_state:
            self.header_label.configure(foreground=CyberpunkTheme.NEON_BLUE)
            self.result_container.configure(bg=CyberpunkTheme.NEON_BLUE)
        else:
            self.header_label.configure(foreground=CyberpunkTheme.NEON_PURPLE)
            self.result_container.configure(bg=CyberpunkTheme.NEON_PINK)
        
        # Schedule next blink
        self.root.after(1000, self.run_blink_effect)
    
    def load_model(self):
        """Load and train the model in a separate thread"""
        try:
            # NLTK requirements
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            
            # Read dataset
            data = pd.read_csv('MNDS_preprocessed.csv')
            
            # Fill NaN values in 'new_content' with an empty string
            X = data['new_content'].fillna('')
            Y = data['category_level_1']
            
            # Convert text to TF-IDF features
            self.vectorizer = TfidfVectorizer(max_features=5000)
            X_tfidf = self.vectorizer.fit_transform(X)
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X_tfidf, Y, test_size=0.2, random_state=42)
            
            # Train Logistic Regression model
            self.model = LogisticRegression(random_state=42, max_iter=1000)
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred_lr = self.model.predict(X_test)
            lr_accuracy = accuracy_score(y_test, y_pred_lr)
            self.accuracy = lr_accuracy
            
            # Set flag that model is loaded
            self.is_model_loaded = True
        except Exception as e:
            print(f"Error loading model: {e}")
            messagebox.showerror("SYSTEM ERROR", f"NEURAL NETWORK INITIALIZATION FAILED: {e}")
    
    def check_model_loaded(self):
        """Check if the model is loaded and update UI"""
        if self.is_model_loaded:
            self.progress.stop()
            self.progress.pack_forget()
            self.status_label.config(
                text=f"NEURAL ENGINE ONLINE | ACCURACY: {self.accuracy:.2%}",
                foreground=CyberpunkTheme.NEON_GREEN
            )
            self.predict_button.state(['!disabled'])
        else:
            self.root.after(100, self.check_model_loaded)
    
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
        # Preprocess the input text
        cleaned_text = self.preprocess_text(text)
        # Transform using the same TF-IDF vectorizer
        text_tfidf = self.vectorizer.transform([cleaned_text])
        # Predict category
        prediction = self.model.predict(text_tfidf)[0]
        return prediction
    
    def predict(self):
        """Handle prediction button click"""
        if not self.is_model_loaded:
            messagebox.showwarning("SYSTEM WARNING", "NEURAL ENGINE STILL INITIALIZING. PLEASE WAIT.")
            return
        
        # Get text from input
        text = self.text_input.get("1.0", tk.END).strip()
        if text == "Enter news text here for analysis...":
            text = ""
            
        if not text:
            messagebox.showwarning("SYSTEM WARNING", "DATA STREAM EMPTY. PLEASE ENTER TEXT TO ANALYZE.")
            return
        
        try:
            # Set status to processing
            self.status_label.config(
                text="PROCESSING DATA STREAM...",
                foreground=CyberpunkTheme.NEON_PURPLE
            )
            self.root.update()
            
            # Add typing animation effect
            self.category_result.config(text="ANALYZING")
            self.root.update()
            self.root.after(100)
            
            # Get prediction
            pred_category = self.predict_category(text)
            
            # Flash effect before showing result
            for _ in range(3):
                self.result_container.configure(bg=CyberpunkTheme.NEON_YELLOW)
                self.root.update()
                self.root.after(100)
                self.result_container.configure(bg=CyberpunkTheme.NEON_BLUE)
                self.root.update()
                self.root.after(100)
            
            # Update result with typing effect
            self.category_result.config(text=pred_category)
            
            # Reset status
            self.status_label.config(
                text=f"ANALYSIS COMPLETE | ACCURACY: {self.accuracy:.2%}",
                foreground=CyberpunkTheme.NEON_GREEN
            )
        except Exception as e:
            messagebox.showerror("SYSTEM ERROR", f"ANALYSIS FAILED: {e}")
            print(f"Error during prediction: {e}")

if __name__ == "__main__":
    # Clear console (for Windows)
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')
    
    # Create main window
    root = tk.Tk()
    app = NewsCategoryPredictor(root)
    root.mainloop()
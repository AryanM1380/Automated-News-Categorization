# Automated News Categorization

This project classifies news articles into categories using small language models and pre-trained NLP models. It is part of my Bachelor's thesis.

## Features
- Categorizes news articles
- Uses NLP models
- Simple data preprocessing

## Technologies Used
- Python
- NLP Libraries: SpaCy, Transformers, NLTK
- Machine Learning: Scikit-Learn, TensorFlow/PyTorch

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AryanM1380/Automated-News-Categorization.git
   cd Automated-News-Categorization
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Preprocess data:
   ```bash
   python preprocess.py
   ```
2. Train model:
   ```bash
   python train.py
   ```
3. Predict category:
   ```bash
   python predict.py --input "news_article.txt"
   ```

## License
MIT License.


# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from deep_translator import GoogleTranslator

from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import cv2
import numpy as np


# Mushroom Model

df = pd.read_csv("mushrooms.csv")

# فصل الهدف عن الخصائص
X = df.drop("class", axis=1)
y = df["class"]

# تحويل القيم النصية إلى أرقام
encoders = {}
for column in X.columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    encoders[column] = le

# ترميز الــ target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# تدريب الموديل
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# حفظ الموديل و الـ encoders
with open("mushroom_model.pkl", "wb") as f:
    pickle.dump({"model": model, "encoders": encoders, "target_encoder": target_encoder}, f)
print("✅ Model trained and saved successfully!")

# Spam Model

def clean_text(text):

    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

# Load Dataset

@st.cache_data
def load_data(csv_path="spam.csv"):
    """
    Load spam dataset and preprocess it.
    Maps 'ham' -> 0 and 'spam' -> 1, and applies text cleaning.
    """
    df = pd.read_csv(csv_path, encoding="latin-1")[['Category', 'Message']]
    df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})
    df['clean_Message'] = df['Message'].apply(clean_text)
    return df


# Train Model

@st.cache_resource
def train_model(df):

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_Message'], df['Category'], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train_tfidf, y_train)

    y_pred = rf.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return rf, vectorizer, acc, report


# Text Summarization

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from deep_translator import GoogleTranslator


def summarize_text(text, method="LSA", lang="english", sentence_count=3):

    info_msg = None

    # Detect input language (simple ASCII check)
    is_english = all(ord(c) < 128 for c in text)

    text_to_summarize = text

    # Translate if needed
    if is_english and lang == "arabic":
        text_to_summarize = GoogleTranslator(source="en", target="ar").translate(text)
        info_msg = "🌍 Input detected as English → Translated to Arabic before summarizing."
    elif not is_english and lang == "english":
        text_to_summarize = GoogleTranslator(source="ar", target="en").translate(text)
        info_msg = "🌍 Input detected as Arabic → Translated to English before summarizing."

    # Summarization
    parser = PlaintextParser.from_string(text_to_summarize, Tokenizer(lang))

    if method == "LSA":
        summarizer = LsaSummarizer()
    elif method == "LexRank":
        summarizer = LexRankSummarizer()
    else:
        summarizer = LuhnSummarizer()

    summary_sentences = [str(sentence) for sentence in summarizer(parser.document, sentence_count)]

    return summary_sentences, info_msg

# Image Processing

from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import cv2
import numpy as np

def rotate_image(image, angle):
    return image.rotate(angle)

def grayscale_image(image):
    return image.convert("L")

def flip_horizontal(image):
    return ImageOps.mirror(image)

def flip_vertical(image):
    return ImageOps.flip(image)

def blur_image(image, level):
    return image.filter(ImageFilter.GaussianBlur(level))

def sharpen_image(image):
    return image.filter(ImageFilter.SHARPEN)

def adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def adjust_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def cartoon_effect(image):
    img_cv = np.array(image)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.medianBlur(img_gray, 5)
    edges = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )
    color = cv2.bilateralFilter(img_cv, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return Image.fromarray(cartoon)

def denoise_image(image):
    img_cv = np.array(image)
    denoised = cv2.fastNlMeansDenoisingColored(img_cv, None, 10, 10, 7, 21)
    return Image.fromarray(denoised)

def edge_detection(image, mode="Canny"):
    img_cv = np.array(image)
    if mode == "Canny":
        edges = cv2.Canny(img_cv, 100, 200)
        return Image.fromarray(edges)
    elif mode == "Contours":
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(img_cv, contours, -1, (0, 255, 0), 2)
        return Image.fromarray(img_cv)
    elif mode == "Sobel":
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        sobel = cv2.magnitude(sobelx, sobely)
        sobel = np.uint8(sobel)
        return Image.fromarray(sobel)
    elif mode == "Laplacian":
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        return Image.fromarray(laplacian)

def smooth_image(image):
    img_cv = np.array(image)
    smooth = cv2.GaussianBlur(img_cv, (7, 7), 0)
    return Image.fromarray(smooth)

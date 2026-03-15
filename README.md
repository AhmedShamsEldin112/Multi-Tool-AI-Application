Multi-Tool AI App

Hey! I'm **Ahmed Shams**, a Data Scientist passionate about building things that actually work and look good doing it.

This project is something I'm genuinely proud of — it started as a way to practice different areas of Data Science and Machine Learning, and ended up becoming a full multi-tool app that brings four completely different AI-powered tools under one clean interface.

---

What's this about?

Ever wanted one place where you can classify mushrooms, detect spam, summarize long articles, *and* edit images — all without switching between five different tools?

Yeah, me too. So I built it.

---

What's inside?

Mushroom Classifier
I trained a **Random Forest** model on 22 physical features of mushrooms to predict whether they're safe to eat or will ruin your day. Spoiler: some of them really will.

Spam Classifier
Fed up with spam? This tool uses **TF-IDF + Random Forest** to tell you instantly if a message is spam or not — and even shows you the probability. Quite satisfying, honestly.

Text Summarizer
Long article? No problem. This summarizer supports three different NLP methods (**LSA, LexRank, Luhn**) and works in both **English and Arabic** with automatic translation. You can even download the summary.

Image Processing
12 different operations — from basic stuff like rotate and grayscale, to fun things like cartoon effect and edge detection. Upload an image, pick what you want to do with it, download the result. Simple.

---

Tech Stack

| Tool | Why I used it |
|------|--------------|
| Streamlit | Fast, clean UI without writing a single line of HTML |
| Scikit-learn | Reliable ML models that just work |
| OpenCV + Pillow | The duo you need for serious image processing |
| Sumy | Solid NLP summarization library |
| Deep Translator | Seamless Arabic ↔ English translation |
| Pandas + NumPy | Because data science without these two isn't data science |

---

How to run it locally

```bash
git clone https://github.com/AhmedShamsEldin112/multi-tool-app.git
cd multi-tool-app
pip install -r requirements.txt
streamlit run main.py
```

That's it. No complicated setup, no environment headaches.

---

## Project Structure

```
multi-tool-app/
├── main.py            # The entire Streamlit UI lives here
├── model.py           # All ML models + processing functions
├── mushrooms.csv      # Mushroom dataset (UCI)
├── spam.csv           # Spam/Ham dataset
└── requirements.txt   # All dependencies
```

---

About Me

I'm a Data Scientist who likes building end-to-end projects — from raw data all the way to a working interface that anyone can use. This project is one of those.

Feel free to reach out, fork the repo, or just star it if you found it useful.

**GitHub:** [AhmedShamsEldin112](https://github.com/AhmedShamsEldin112)

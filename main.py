# main_app.py
import streamlit as st
import pandas as pd
from PIL import Image
import io
import pickle
from model import *
import model as im  # لوظائف معالجة الصور


#  إعدادات عامة للستريمليت

st.set_page_config(
    page_title=" Multi-Tool App",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CSS لتجميل الواجهة + Dark Mode + Animations

st.markdown("""
<style>
/* Sidebar Gradient + Dark Mode */
[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #1e3c72, #2a5298);
    color: white;
    font-weight: bold;
}

/* Card style with animation */
.card {
    background-color: #2b2b2b;
    padding: 20px;
    margin: 10px;
    border-radius: 12px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.5);
    transition: transform 0.3s, box-shadow 0.3s;
}
.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 30px rgba(0,0,0,0.7);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(to right, #4facfe, #00f2fe);
    color: white;
    font-size: 16px;
    padding: 10px 24px;
    border-radius: 8px;
    border: none;
    transition: transform 0.2s;
}
.stButton>button:hover {
    transform: scale(1.05);
    cursor: pointer;
}

/* Titles */
h1 {
    color: #4facfe;
    text-align: center;
}

/* Dark Mode general */
body, .stApp, .css-18e3th9 {
    background-color: #121212;
    color: #e0e0e0;
}

/* TextAreas and Inputs */
textarea, input, select {
    background-color: #1f1f1f;
    color: #e0e0e0;
    border-radius: 8px;
    border: 1px solid #444;
    padding: 5px;
}
</style>
""", unsafe_allow_html=True)


# Sidebar Gradient + اختيار المشروع

st.sidebar.markdown("""
    <div style='background: linear-gradient(to bottom, #1e3c72, #2a5298);
                padding: 20px; border-radius:12px; color:white; text-align:center; font-size:20px'>
         Multi-Tool App
    </div>
""", unsafe_allow_html=True)

app_choice = st.sidebar.selectbox(
    "Select Project",
    ["Mushroom Classifier", "Spam Classifier", "Text Summarizer", "Image Processing"]
)


# 🍄 Mushroom Classifier

if app_choice == "Mushroom Classifier":
    st.markdown("<h1>🍄 Mushroom Classification</h1>", unsafe_allow_html=True)
    st.write("Select mushroom features to predict if it is **Edible** 🟢 or **Poisonous** ☠️")

    with open("mushroom_model.pkl", "rb") as f:
        data = pickle.load(f)

    model = data["model"]
    encoders = data["encoders"]
    target_encoder = data["target_encoder"]

    attributes = {
    "cap-shape": {"Bell": "b", "Conical": "c", "Convex": "x", "Flat": "f", "Knobbed": "k", "Sunken": "s"},
    "cap-surface": {"Fibrous": "f", "Grooves": "g", "Scaly": "y", "Smooth": "s"},
    "cap-color": {"Brown": "n", "Buff": "b", "Cinnamon": "c", "Gray": "g", "Green": "r",
                  "Pink": "p", "Purple": "u", "Red": "e", "White": "w", "Yellow": "y"},
    "bruises": {"Yes": "t", "No": "f"},
    "odor": {"Almond": "a", "Anise": "l", "Creosote": "c", "Fishy": "y", "Foul": "f",
             "Musty": "m", "None": "n", "Pungent": "p", "Spicy": "s"},
    "gill-attachment": {"Attached": "a", "Descending": "d", "Free": "f", "Notched": "n"},
    "gill-spacing": {"Close": "c", "Crowded": "w", "Distant": "d"},
    "gill-size": {"Broad": "b", "Narrow": "n"},
    "gill-color": {"Black": "k", "Brown": "n", "Buff": "b", "Chocolate": "h", "Gray": "g", "Green": "r",
                   "Orange": "o", "Pink": "p", "Purple": "u", "Red": "e", "White": "w", "Yellow": "y"},
    "stalk-shape": {"Enlarging": "e", "Tapering": "t"},
    "stalk-root": {"Bulbous": "b", "Club": "c", "Cup": "u", "Equal": "e", "Rhizomorphs": "z",
                   "Rooted": "r", "Missing": "?"},
    "stalk-surface-above-ring": {"Fibrous": "f", "Scaly": "y", "Silky": "k", "Smooth": "s"},
    "stalk-surface-below-ring": {"Fibrous": "f", "Scaly": "y", "Silky": "k", "Smooth": "s"},
    "stalk-color-above-ring": {"Brown": "n", "Buff": "b", "Cinnamon": "c", "Gray": "g", "Orange": "o",
                               "Pink": "p", "Red": "e", "White": "w", "Yellow": "y"},
    "stalk-color-below-ring": {"Brown": "n", "Buff": "b", "Cinnamon": "c", "Gray": "g", "Orange": "o",
                               "Pink": "p", "Red": "e", "White": "w", "Yellow": "y"},
    "veil-type": {"Partial": "p", "Universal": "u"},
    "veil-color": {"Brown": "n", "Orange": "o", "White": "w", "Yellow": "y"},
    "ring-number": {"None": "n", "One": "o", "Two": "t"},
    "ring-type": {"Cobwebby": "c", "Evanescent": "e", "Flaring": "f", "Large": "l", "None": "n",
                  "Pendant": "p", "Sheathing": "s", "Zone": "z"},
    "spore-print-color": {"Black": "k", "Brown": "n", "Buff": "b", "Chocolate": "h", "Green": "r",
                          "Orange": "o", "Purple": "u", "White": "w", "Yellow": "y"},
    "population": {"Abundant": "a", "Clustered": "c", "Numerous": "n", "Scattered": "s",
                   "Several": "v", "Solitary": "y"},
    "habitat": {"Grasses": "g", "Leaves": "l", "Meadows": "m", "Paths": "p",
                "Urban": "u", "Waste": "w", "Woods": "d"}
}

    user_input = {}
    cols = st.columns(2)
    for i, (feature, options) in enumerate(attributes.items()):
        choice = cols[i % 2].selectbox(f"{feature.replace('-', ' ').title()}:", list(options.keys()))
        user_input[feature] = options[choice]

    input_df = pd.DataFrame([user_input])

    # Encode inputs
    for column in input_df.columns:
        le = encoders[column]
        input_df[column] = input_df[column].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        input_df[column] = le.transform(input_df[column])

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if st.button("🔮 Predict Mushroom"):
        prediction = model.predict(input_df)[0]
        result = target_encoder.inverse_transform([prediction])[0]
        if result == "e":
            st.success("🟢 This mushroom is **Edible**.", icon="✅")
        else:
            st.error("☠️ This mushroom is **Poisonous**!", icon="⚠️")
    st.markdown("</div>", unsafe_allow_html=True)


# 📧 Spam Classifier

elif app_choice == "Spam Classifier":
    st.markdown("<h1>📧 Spam Classifier</h1>", unsafe_allow_html=True)

    df = load_data()
    model_spam, vectorizer, acc, report = train_model(df)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    user_input = st.text_area("✍️ Enter your message:", height=120)

    if st.button("Classify Message"):
        if user_input.strip() != "":
            cleaned = clean_text(user_input)
            vectorized = vectorizer.transform([cleaned])
            prediction = model_spam.predict(vectorized)[0]
            prob = model_spam.predict_proba(vectorized)[0]

            if prediction == 1:
                st.error(f"🚨 This message is SPAM! (Spam Probability: {prob[1]*100:.2f}%)", icon="⚠️")
            else:
                st.success(f"✅ This message is HAM (Not Spam). (Ham Probability: {prob[0]*100:.2f}%)", icon="✅")
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("📊 Model Performance on Test Data"):
        st.write(f"**Accuracy:** {acc:.4f}")
        st.dataframe(pd.DataFrame(report).transpose())


# 📝 Text Summarizer

elif app_choice == "Text Summarizer":
    # text_summ/app.py (embedded)
    import io
    from model import summarize_text  # تأكد أن ملف model.py يحتوي على الدالة summarize_text

    # Streamlit UI for summarization
    st.title("📝 Smart Text Summarization App (with Translation)")

    # User Input
    user_input = st.text_area("✍️ Enter your text here:")

    # Choose summarization method
    method = st.selectbox("Choose summarization method:", ["LSA", "LexRank", "Luhn"])

    # Choose language for summary
    lang = st.selectbox("Choose summary language:", ["english", "arabic"])

    # Number of sentences
    sentence_count = st.number_input(
        "Enter number of sentences in summary:", min_value=1, max_value=20, value=3
    )

    # Summarize button
    if st.button("Summarize"):
        if user_input.strip() != "":
            # Call summarize_text function
            summary, info_msg = summarize_text(
                user_input, method=method, lang=lang, sentence_count=sentence_count
            )

            # Show translation info if exists
            if info_msg:
                st.info(info_msg)

            # Display summary
            st.subheader("📌 Summary:")
            result_text = ""
            for sentence in summary:
                st.success(sentence)
                result_text += sentence + "\n"

            # Download summary as text file
            buf = io.BytesIO(result_text.encode("utf-8"))
            st.download_button(
                label="📥 Download Summary",
                data=buf,
                file_name="summary.txt",
                mime="text/plain"
            )


# 🌟 Image Processing

elif app_choice == "Image Processing":
    st.markdown("<h1>🌟 Image Processing</h1>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original Image", use_column_width=True)

        option = st.selectbox(
            "Choose an operation",
            ["Rotate", "Grayscale", "Flip Horizontal", "Flip Vertical",
             "Blur", "Sharpen", "Brightness", "Contrast",
             "Cartoon Effect", "Denoised", "Edge Detection", "Smooth"]
        )

        if option == "Rotate":
            angle = st.slider("Rotation Angle", 0, 360, 90)
        elif option == "Blur":
            blur_level = st.slider("Blur Strength", 1, 10, 2)
        elif option == "Brightness":
            brightness = st.slider("Brightness", 0.1, 3.0, 1.0)
        elif option == "Contrast":
            contrast = st.slider("Contrast", 0.1, 3.0, 1.0)
        elif option == "Edge Detection":
            mode = st.selectbox("Edge Detection Mode", ["Canny", "Contours", "Sobel", "Laplacian"])

        if st.button("Apply Operation"):
            if option == "Rotate":
                result = im.rotate_image(image, angle)
            elif option == "Grayscale":
                result = im.grayscale_image(image)
            elif option == "Flip Horizontal":
                result = im.flip_horizontal(image)
            elif option == "Flip Vertical":
                result = im.flip_vertical(image)
            elif option == "Blur":
                result = im.blur_image(image, blur_level)
            elif option == "Sharpen":
                result = im.sharpen_image(image)
            elif option == "Brightness":
                result = im.adjust_brightness(image, brightness)
            elif option == "Contrast":
                result = im.adjust_contrast(image, contrast)
            elif option == "Cartoon Effect":
                result = im.cartoon_effect(image)
            elif option == "Denoised":
                result = im.denoise_image(image)
            elif option == "Edge Detection":
                result = im.edge_detection(image, mode)
            elif option == "Smooth":
                result = im.smooth_image(image)

            st.image(result, caption=f"Result: {option}", use_column_width=True)

            buf = io.BytesIO()
            result.save(buf, format="PNG")
            st.download_button(
                label="📥 Download Result",
                data=buf.getvalue(),
                file_name="processed_image.png",
                mime="image/png"
            )
    st.markdown("</div>", unsafe_allow_html=True)

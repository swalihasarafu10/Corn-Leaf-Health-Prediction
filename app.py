import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------- Page setup ----------
st.set_page_config(page_title="Corn Leaf Health Checker", page_icon="🌽", layout="centered")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corn_leaf_model.h5")
IMG_SIZE = (224, 224)

# ---------- Dark aesthetic styling ----------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Poppins', sans-serif;
}

/* background with soft green glow */
.stApp {
    background:
        radial-gradient(900px 500px at 15% -10%, rgba(52, 211, 153, 0.12), transparent 60%),
        radial-gradient(700px 400px at 100% 0%, rgba(251, 191, 36, 0.08), transparent 55%),
        #0b0f0d;
}

/* hide default chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2.5rem; max-width: 760px; }

/* hero */
.hero { text-align: center; margin-bottom: 1.6rem; }
.hero .badge {
    display: inline-block; padding: 0.3rem 0.9rem; border-radius: 999px;
    background: rgba(52, 211, 153, 0.12); border: 1px solid rgba(52, 211, 153, 0.35);
    color: #6ee7b7; font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase;
}
.hero h1 {
    margin: 0.9rem 0 0.4rem 0; font-size: 2.5rem; font-weight: 700; line-height: 1.15;
    background: linear-gradient(90deg, #fbbf24 0%, #a3e635 50%, #34d399 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { color: #8fa398; font-weight: 300; font-size: 1rem; margin: 0; }

/* mode switcher (radio styled as pills) */
[data-testid="stRadio"] [role="radiogroup"] {
    display: flex; gap: 0.5rem; background: #131a16; padding: 0.35rem; border-radius: 14px;
    border: 1px solid #1f2b25; width: 100%;
}
[data-testid="stRadio"] [role="radiogroup"] > label {
    flex: 1; justify-content: center; align-items: center; margin: 0; padding: 0.7rem 0.5rem;
    border-radius: 10px; cursor: pointer; color: #8fa398; font-weight: 500;
    transition: background 0.2s ease, color 0.2s ease;
}
[data-testid="stRadio"] [role="radiogroup"] > label > div:first-child { display: none; }
[data-testid="stRadio"] [role="radiogroup"] > label:hover { color: #e7efe9; }
[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(135deg, rgba(52,211,153,0.22), rgba(163,230,53,0.12));
    color: #e7efe9;
}
[data-testid="stRadio"] [role="radiogroup"] > label p { font-size: 1rem; }

/* file uploader */
[data-testid="stFileUploader"] section {
    background: #101612; border: 1.5px dashed #2c4036; border-radius: 16px; padding: 1.6rem;
    transition: border-color 0.2s ease, background 0.2s ease;
}
[data-testid="stFileUploader"] section:hover { border-color: #34d399; background: #121b16; }
[data-testid="stFileUploader"] button {
    background: #34d399; color: #04130c; border: none; border-radius: 10px; font-weight: 600;
}

/* camera */
[data-testid="stCameraInput"] > div { border-radius: 16px; overflow: hidden; }
[data-testid="stCameraInput"] button {
    background: #34d399; color: #04130c; border: none; border-radius: 10px; font-weight: 600;
}

/* images */
[data-testid="stImage"] img {
    border-radius: 16px; border: 1px solid #1f2b25;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
}
[data-testid="stImage"] { margin-top: 1rem; }

/* result card */
.result-card {
    margin-top: 1.2rem; padding: 1.5rem 1.6rem; border-radius: 18px;
    background: #131a16; border: 1px solid #1f2b25;
}
.result-card.healthy {
    border-color: rgba(52, 211, 153, 0.55);
    box-shadow: 0 0 40px rgba(52, 211, 153, 0.12);
}
.result-card.diseased {
    border-color: rgba(248, 113, 113, 0.55);
    box-shadow: 0 0 40px rgba(248, 113, 113, 0.12);
}
.result-top { display: flex; align-items: center; gap: 1rem; }
.result-icon { font-size: 2.4rem; }
.result-label { font-size: 1.7rem; font-weight: 700; letter-spacing: 0.03em; line-height: 1.1; }
.result-card.healthy .result-label { color: #34d399; }
.result-card.diseased .result-label { color: #f87171; }
.result-sub { color: #8fa398; font-size: 0.9rem; font-weight: 300; }
.bar-row { margin-top: 1.2rem; }
.bar-head { display: flex; justify-content: space-between; color: #b6c4bb; font-size: 0.85rem; margin-bottom: 0.3rem; }
.bar-track { height: 9px; border-radius: 999px; background: #0b0f0d; overflow: hidden; border: 1px solid #1f2b25; }
.bar-fill { height: 100%; border-radius: 999px; }
.note {
    margin-top: 1rem; padding: 0.8rem 1rem; border-radius: 12px;
    background: rgba(251, 191, 36, 0.08); border: 1px solid rgba(251, 191, 36, 0.3);
    color: #fcd34d; font-size: 0.85rem;
}
.footer-note { text-align: center; color: #5d7066; font-size: 0.78rem; margin-top: 2.5rem; font-weight: 300; }

/* spinner */
.stSpinner > div { border-top-color: #34d399 !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------- Load model once ----------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def predict(image):
    # the model rescales pixels itself, so only resize here
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image, dtype="float32")
    arr = np.expand_dims(arr, axis=0)
    # one sigmoid output: close to 1 = healthy, close to 0 = diseased (index 0 = diseased, 1 = healthy)
    return float(model.predict(arr, verbose=0)[0][0])


def bar(label, value, color):
    return f"""
    <div class="bar-row">
        <div class="bar-head"><span>{label}</span><span>{value * 100:.1f}%</span></div>
        <div class="bar-track"><div class="bar-fill" style="width:{value * 100:.1f}%; background:{color};"></div></div>
    </div>
    """


def show_result(image):
    st.image(image)

    with st.spinner("Analysing leaf..."):
        prob_healthy = predict(image)

    prob_diseased = 1 - prob_healthy
    is_healthy = prob_healthy > 0.5
    confidence = prob_healthy if is_healthy else prob_diseased

    css_class = "healthy" if is_healthy else "diseased"
    icon = "🌿" if is_healthy else "🍂"
    label = "HEALTHY" if is_healthy else "DISEASED"
    sub = "This leaf looks healthy." if is_healthy else "This leaf shows signs of disease."

    html = f"""
    <div class="result-card {css_class}">
        <div class="result-top">
            <div class="result-icon">{icon}</div>
            <div>
                <div class="result-label">{label}</div>
                <div class="result-sub">{sub} Confidence {confidence * 100:.1f}%</div>
            </div>
        </div>
        {bar("Healthy", prob_healthy, "linear-gradient(90deg,#10b981,#a3e635)")}
        {bar("Diseased", prob_diseased, "linear-gradient(90deg,#f59e0b,#f87171)")}
    </div>
    """

    if confidence < 0.70:
        html += '<div class="note">The model is not very sure about this one. Try a clearer, closer photo of a single leaf in good light.</div>'

    st.markdown(html, unsafe_allow_html=True)


# ---------- UI ----------
st.markdown(
    """
<div class="hero">
    <span class="badge">AI Crop Health</span>
    <h1>🌽 Corn Leaf Health Checker</h1>
    <p>Upload a leaf photo or snap one with your webcam and find out if it is healthy or diseased.</p>
</div>
""",
    unsafe_allow_html=True,
)

if not os.path.exists(MODEL_PATH):
    st.error("Model file 'corn_leaf_model.h5' was not found. Put it in the same folder as app.py.")
    st.stop()

model = load_model()

mode = st.radio(
    "Choose input",
    ["📁  Upload image", "📷  Use webcam"],
    horizontal=True,
    label_visibility="collapsed",
)

# Only the selected option is rendered, so the webcam stays off until "Use webcam" is chosen
if mode == "📁  Upload image":
    uploaded = st.file_uploader("Drop a leaf image here", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded is not None:
        show_result(Image.open(uploaded))
else:
    st.caption("Hold the leaf close to the camera in good light, then take the photo.")
    photo = st.camera_input("Take a photo of the leaf", label_visibility="collapsed")
    if photo is not None:
        show_result(Image.open(photo))

st.markdown(
    '<div class="footer-note">Trained on close-up photos of single corn leaves. Results on whole plants or cluttered backgrounds may be unreliable.</div>',
    unsafe_allow_html=True,
)

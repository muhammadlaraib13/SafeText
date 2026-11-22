import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

APP_NAME = "SafeText"

st.set_page_config(
    page_title=f"{APP_NAME} | Text & Link Safety Checker",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "msg_box" not in st.session_state:
    st.session_state.msg_box = ""
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


@st.cache_resource
def train_detector():
    df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "message"]
    df = df.dropna().copy()
    df["label"] = df["label"].str.strip().str.lower()
    df["message"] = df["message"].astype(str)

    vectorizer = CountVectorizer(lowercase=True, stop_words="english")
    X_vec = vectorizer.fit_transform(df["message"])

    model = MultinomialNB(alpha=0.5)
    model.fit(X_vec, df["label"])

    return vectorizer, model


def set_sample_message(sample_text: str):
    st.session_state.msg_box = sample_text
    st.session_state.analysis_result = None


def clear_message():
    st.session_state.msg_box = ""
    st.session_state.analysis_result = None


vectorizer, model = train_detector()

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #dbeafe 0%, #ede9fe 50%, #fef3c7 100%);
            color: #0f172a;
        }
        [data-testid="stHeader"] {
            background: transparent !important;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e1b4b 0%, #312e81 55%, #1d4ed8 100%);
        }
        [data-testid="stSidebar"] * {
            color: #f8fafc !important;
        }
        .block-container {
            padding-top: 2.5rem;
            padding-bottom: 3rem;
            max-width: 1020px;
        }
        .top-banner {
            width: 100%;
            background: linear-gradient(120deg, #2563eb 0%, #7c3aed 55%, #db2777 100%);
            color: #ffffff;
            border-radius: 18px;
            padding: 1.5rem 1.75rem;
            margin-top: 0.25rem;
            margin-bottom: 1.35rem;
            box-shadow: 0 10px 24px rgba(79, 70, 229, 0.2);
        }
        .top-banner h1 {
            margin: 0 0 0.4rem 0;
            font-size: clamp(1.75rem, 3.2vw, 2.35rem);
            font-weight: 800;
            line-height: 1.2;
            color: #ffffff;
        }
        .top-banner p {
            margin: 0;
            font-size: clamp(1rem, 1.5vw, 1.12rem);
            color: #f8fafc;
            line-height: 1.55;
        }
        .section-label {
            color: #0f172a !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            margin-bottom: 0.35rem !important;
        }
        .result-safe {
            width: 100%;
            background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
            border: 2px solid #059669;
            border-left: 10px solid #059669;
            color: #064e3b;
            border-radius: 16px;
            padding: 1.3rem 1.5rem;
            margin-top: 1.1rem;
        }
        .result-danger {
            width: 100%;
            background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
            border: 2px solid #dc2626;
            border-left: 10px solid #dc2626;
            color: #7f1d1d;
            border-radius: 16px;
            padding: 1.3rem 1.5rem;
            margin-top: 1.1rem;
        }
        .result-safe h3, .result-danger h3 {
            margin: 0 0 0.5rem 0;
            font-size: clamp(1.2rem, 2vw, 1.45rem);
        }
        .result-safe p, .result-danger p, .result-safe li, .result-danger li {
            font-size: 1.04rem;
            line-height: 1.6;
        }
        .stTextArea textarea {
            background-color: #ffffff !important;
            color: #0f172a !important;
            caret-color: #0f172a !important;
            font-size: 1.05rem !important;
            line-height: 1.55 !important;
            border-radius: 12px !important;
            border: 2px solid #6366f1 !important;
        }
        .stTextArea textarea::placeholder {
            color: #64748b !important;
            opacity: 1 !important;
        }
        .stButton > button {
            width: 100% !important;
            background-color: #ffffff !important;
            color: #0f172a !important;
            border: 2px solid #6366f1 !important;
            border-radius: 12px !important;
            min-height: 2.85rem !important;
            font-weight: 700 !important;
        }
        .stButton > button:hover {
            background-color: #eef2ff !important;
            color: #1e1b4b !important;
            border-color: #4338ca !important;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #2563eb 0%, #7c3aed 100%) !important;
            color: #ffffff !important;
            border: none !important;
            min-height: 3.05rem !important;
            font-size: 1.05rem !important;
        }
        .stButton > button[kind="primary"]:hover {
            filter: brightness(1.06);
            color: #ffffff !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar - Things Users Should Avoid
with st.sidebar:
    st.markdown(f"## 🛡️ {APP_NAME}")
    st.caption("Easy safety check for texts, emails, and links.")
    st.divider()
    st.markdown("### 🚫 What to Avoid")
    st.markdown(
        """
        - **Unknown Links:** Avoid clicking links from strangers or unexpected messages.
        - **Sharing OTPs & PINs:** Never share one-time codes, PINs, passwords, or bank details.
        - **Fake Prize & Urgency Traps:** Be careful with messages claiming instant rewards or urgent account blocks.
        - **Unexpected Money Requests:** Do not send money or gift cards based on sudden texts.
        - **Suspicious Files or Logins:** Avoid downloading unknown files or logging in through random links.
        """
    )

# Top Header Banner
st.markdown(
    f"""
    <div class="top-banner">
        <h1>🛡️ {APP_NAME}</h1>
        <p>Paste any SMS, email, or link below to quickly check whether it looks safe or like a scam.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sample Buttons
st.markdown('<p class="section-label">Try a Sample Message</p>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns([1.15, 1.15, 1.15, 0.75])
with c1:
    st.button(
        "🟢 Safe Sample",
        on_click=set_sample_message,
        args=("Hey! Are we still meeting at 5 PM today? Bring your notebook.",),
    )
with c2:
    st.button(
        "🔴 Bank Scam",
        on_click=set_sample_message,
        args=("URGENT: Your bank account will be blocked today. Verify PIN now: http://bank-alert-check.example.com",),
    )
with c3:
    st.button(
        "🎁 Prize Scam",
        on_click=set_sample_message,
        args=("Congratulations! You won a free cash prize. Click here to claim now: http://free-cash-win.example.com",),
    )
with c4:
    st.button(
        "🧹 Clear",
        on_click=clear_message,
    )

# Input Box & Check Button
st.markdown('<p class="section-label">Paste or Type Message Here</p>', unsafe_allow_html=True)
st.text_area(
    "Paste or Type Message Here",
    label_visibility="collapsed",
    key="msg_box",
    height=140,
    placeholder="Type or paste an SMS, email, or link here...",
)

run_check = st.button("🔍 Check Message", type="primary")

if run_check:
    cleaned_text = st.session_state.msg_box.strip()
    if not cleaned_text:
        st.warning("Please type or paste a message first, or click a sample button above.")
    else:
        vec_msg = vectorizer.transform([cleaned_text])
        prediction = model.predict(vec_msg)[0]
        probs = model.predict_proba(vec_msg)[0]
        classes = list(model.classes_)

        spam_score = probs[classes.index("spam")] * 100 if "spam" in classes else 0.0
        ham_score = probs[classes.index("ham")] * 100 if "ham" in classes else 0.0

        st.session_state.analysis_result = {
            "prediction": prediction,
            "spam_score": spam_score,
            "ham_score": ham_score,
        }

# Result Display
result = st.session_state.analysis_result
if result is not None:
    if result["prediction"] == "spam":
        st.markdown(
            f"""
            <div class="result-danger">
                <h3>🚨 WARNING: Likely Scam or Spam ({result['spam_score']:.1f}% confidence)</h3>
                <p>This message looks risky. Follow these steps to stay safe:</p>
                <ul>
                    <li>Do not click any link in this message.</li>
                    <li>Never share your OTP, PIN, password, or bank info.</li>
                    <li>If you are unsure, ask someone you trust or check the official source directly.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-safe">
                <h3>✅ SAFE: Looks Like a Normal Message ({result['ham_score']:.1f}% confidence)</h3>
                <p>This message appears normal, but always stay careful if a message asks for money, logins, or private codes.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

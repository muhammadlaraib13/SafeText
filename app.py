import re
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

APP_NAME = "SafeText"

st.set_page_config(
    page_title=f"{APP_NAME} | Text & Link Safety Checker",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "safe_msg_input" not in st.session_state:
    st.session_state["safe_msg_input"] = ""
if "check_output" not in st.session_state:
    st.session_state["check_output"] = None


@st.cache_resource
def train_spam_model():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    if "v1" in df.columns and "v2" in df.columns:
        df = df[["v1", "v2"]].copy()
        df.columns = ["label", "message"]
    else:
        df = df.iloc[:, :2].copy()
        df.columns = ["label", "message"]

    df = df.dropna().copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str)

    x_train, x_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    vec = CountVectorizer(lowercase=True, stop_words="english")
    x_train_vec = vec.fit_transform(x_train)
    x_test_vec = vec.transform(x_test)

    clf = MultinomialNB(alpha=0.5)
    clf.fit(x_train_vec, y_train)

    acc = accuracy_score(y_test, clf.predict(x_test_vec)) * 100.0
    return vec, clf, float(acc)


def extract_signals(text: str) -> dict:
    links = re.findall(r"(?:https?://|www\.)\S+", text, flags=re.IGNORECASE)
    risk_words = [
        "urgent",
        "verify",
        "suspended",
        "blocked",
        "winner",
        "won",
        "prize",
        "free",
        "click",
        "password",
        "pin",
        "otp",
        "bank",
        "claim",
    ]
    lowered = text.lower()
    found = [w for w in risk_words if re.search(rf"\b{re.escape(w)}\b", lowered)]
    return {
        "links": links,
        "link_count": len(links),
        "risk_words": found,
        "word_count": len(text.split()),
    }


def create_probability_chart(ham_pct: float, spam_pct: float):
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    categories = ["Safe (Normal)", "Scam / Risky"]
    scores = [ham_pct, spam_pct]
    bar_colors = ["#10b981", "#ef4444"]

    ax.barh(categories, [100, 100], color="#e2e8f0", height=0.52, zorder=1)
    bars = ax.barh(categories, scores, color=bar_colors, height=0.52, zorder=2)

    ax.set_xlim(0, 115)
    ax.set_xticks([])
    ax.set_yticklabels(categories, fontsize=11, fontweight="bold", color="#0f172a")
    ax.invert_yaxis()
    ax.set_title(
        "How Safe or Risky Does This Message Look?",
        fontsize=12.5,
        fontweight="bold",
        color="#0f172a",
        pad=12,
        loc="left",
    )

    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")

    fig.patch.set_facecolor("#ffffff")
    fig.patch.set_alpha(0.95)
    ax.set_facecolor("#ffffff")
    ax.patch.set_alpha(0.95)

    for bar, val in zip(bars, scores):
        ax.text(
            min(val + 2.5, 101.0),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold",
            color="#0f172a",
            zorder=3,
        )

    plt.tight_layout()
    return fig


st.markdown(
    """
    <style>
        html, body, .stApp {
            color-scheme: light !important;
            background: linear-gradient(135deg, #dbeafe 0%, #ede9fe 50%, #fef3c7 100%) !important;
            color: #0f172a !important;
        }
        [data-testid="stHeader"] {
            background: transparent !important;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e1b4b 0%, #312e81 55%, #1d4ed8 100%) !important;
        }
        [data-testid="stSidebar"] * {
            color: #f8fafc !important;
            -webkit-text-fill-color: #f8fafc !important;
        }
        .block-container {
            padding-top: 2.5rem;
            padding-bottom: 3rem;
            max-width: 1020px;
        }
        .top-banner {
            width: 100%;
            background: linear-gradient(120deg, #2563eb 0%, #7c3aed 55%, #db2777 100%) !important;
            color: #ffffff !important;
            border-radius: 18px;
            padding: 1.5rem 1.75rem;
            margin-top: 0.25rem;
            margin-bottom: 1.35rem;
        }
        .top-banner h1, .top-banner p {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            margin: 0;
        }
        .top-banner h1 {
            font-size: clamp(1.75rem, 3.2vw, 2.35rem);
            font-weight: 800;
            margin-bottom: 0.35rem;
        }
        [data-testid="stAppViewContainer"] .main p,
        [data-testid="stAppViewContainer"] .main span,
        [data-testid="stAppViewContainer"] .main label,
        [data-testid="stAppViewContainer"] .main div[role="radiogroup"] * {
            color: #0f172a !important;
            -webkit-text-fill-color: #0f172a !important;
            opacity: 1 !important;
        }
        div[role="radiogroup"] {
            background: rgba(255, 255, 255, 0.88) !important;
            padding: 0.55rem 0.9rem !important;
            border-radius: 12px !important;
            border: 1.5px solid #c7d2fe !important;
        }
        .stTextArea textarea {
            background-color: #ffffff !important;
            color: #0f172a !important;
            -webkit-text-fill-color: #0f172a !important;
            caret-color: #0f172a !important;
            border-radius: 12px !important;
            border: 2px solid #6366f1 !important;
        }
        .stButton button {
            width: 100% !important;
            border-radius: 12px !important;
            min-height: 2.95rem !important;
            font-weight: 700 !important;
            background-color: #312e81 !important;
            color: #ffffff !important;
            border: 2px solid #818cf8 !important;
        }
        .stButton button p, .stButton button span {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }
        .result-safe {
            background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%) !important;
            border: 2px solid #059669 !important;
            border-left: 10px solid #059669 !important;
            color: #064e3b !important;
            border-radius: 16px;
            padding: 1.3rem 1.5rem;
            margin-top: 1rem;
        }
        .result-danger {
            background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%) !important;
            border: 2px solid #dc2626 !important;
            border-left: 10px solid #dc2626 !important;
            color: #7f1d1d !important;
            border-radius: 16px;
            padding: 1.3rem 1.5rem;
            margin-top: 1rem;
        }
        .result-safe *, .result-danger * {
            color: inherit !important;
            -webkit-text-fill-color: inherit !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    vectorizer, model, accuracy = train_spam_model()
except Exception as exc:
    st.error(f"Could not load spam.csv or train model: {exc}")
    st.stop()

with st.sidebar:
    st.markdown(f"## 🛡️ {APP_NAME}")
    st.caption("Easy safety check for texts, emails, and links.")
    st.divider()
    st.metric("Model Accuracy", f"{accuracy:.2f}%")
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

st.markdown(
    f"""
    <div class="top-banner">
        <h1>🛡️ {APP_NAME}</h1>
        <p>Paste any SMS, email, or link below to quickly check whether it looks safe or like a scam.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("**Choose Check Option**")
check_mode = st.radio(
    "Choose Check Option",
    options=["Quick Check", "Check + Score Graph & Signals"],
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown("**Try a Sample Message**")
c1, c2, c3, c4 = st.columns([1.15, 1.15, 1.15, 0.75])
with c1:
    if st.button("🟢 Safe Sample"):
        st.session_state["safe_msg_input"] = "Hey! Are we still meeting at 5 PM today? Bring your notebook."
        st.session_state["check_output"] = None
        st.rerun()
with c2:
    if st.button("🔴 Bank Scam"):
        st.session_state["safe_msg_input"] = "URGENT: Your bank account will be blocked today. Verify PIN now: http://bank-alert-check.example.com"
        st.session_state["check_output"] = None
        st.rerun()
with c3:
    if st.button("🎁 Prize Scam"):
        st.session_state["safe_msg_input"] = "Congratulations! You won a free cash prize. Click here to claim now: http://free-cash-win.example.com"
        st.session_state["check_output"] = None
        st.rerun()
with c4:
    if st.button("🧹 Clear"):
        st.session_state["safe_msg_input"] = ""
        st.session_state["check_output"] = None
        st.rerun()

st.markdown("**Paste or Type Message Here**")
st.text_area(
    "Paste or Type Message Here",
    key="safe_msg_input",
    label_visibility="collapsed",
    height=140,
    placeholder="Type or paste an SMS, email, or link here...",
)

if st.button("🔍 Check Message", type="primary"):
    raw_msg = st.session_state.get("safe_msg_input", "").strip()
    if not raw_msg:
        st.session_state["check_output"] = {"empty": True}
    else:
        msg_vec = vectorizer.transform([raw_msg])
        pred_label = model.predict(msg_vec)[0]
        proba_vals = model.predict_proba(msg_vec)[0]
        class_list = list(model.classes_)

        spam_pct = float(proba_vals[class_list.index("spam")] * 100.0) if "spam" in class_list else 0.0
        ham_pct = float(proba_vals[class_list.index("ham")] * 100.0) if "ham" in class_list else 0.0
        signals = extract_signals(raw_msg)

        st.session_state["check_output"] = {
            "empty": False,
            "prediction": pred_label,
            "spam_pct": spam_pct,
            "ham_pct": ham_pct,
            "signals": signals,
        }

output = st.session_state.get("check_output")
if output is not None:
    if output.get("empty"):
        st.warning("Please type or paste a message first, or click a sample button above.")
    elif output["prediction"] == "spam":
        st.markdown(
            f"""
            <div class="result-danger">
                <h3>🚨 WARNING: Likely Scam or Spam ({output['spam_pct']:.1f}% confidence)</h3>
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
                <h3>✅ SAFE: Looks Like a Normal Message ({output['ham_pct']:.1f}% confidence)</h3>
                <p>This message appears normal, but always stay careful if a message asks for money, logins, or private codes.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not output.get("empty") and check_mode == "Check + Score Graph & Signals":
        chart_fig = create_probability_chart(output["ham_pct"], output["spam_pct"])
        st.pyplot(chart_fig, use_container_width=True)
        plt.close(chart_fig)

        sig = output["signals"]
        st.write(f"**Links Found:** {sig['link_count']} | **Words:** {sig['word_count']}")
        for url in sig["links"]:
            st.code(url, language="text")
        if sig["risk_words"]:
            st.write("**Risk Keywords Found:** " + ", ".join(f"`{w}`" for w in sig["risk_words"]))

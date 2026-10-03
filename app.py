import random
import streamlit as st
from openai import OpenAI

from recommender import MovieRecommender, run_agent, SYSTEM_PROMPT

# The trained ALS model file is too large for a normal GitHub push (~300MB,
# over GitHub's 100MB hard limit), so it's hosted on Hugging Face Hub
# instead. MovieRecommender downloads and caches both files internally via
# hf_hub_download the first time it's constructed — no manual download
# logic needed here.
HF_REPO_ID = "arpit110/cinematch-als-model"

st.set_page_config(
    page_title="CineMatch — AI Movie Recommender",
    page_icon="🎬",
    layout="centered",
)

# ---------- styling ----------
st.markdown(
    """
    <style>
    .main .block-container { padding-top: 2rem; max-width: 780px; }

    .cine-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #E50914, #F5B700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle { color: #9a9fa8; font-size: 0.95rem; margin-top: -0.3rem; margin-bottom: 1.4rem; }

    div[data-testid="stChatMessage"] { padding: 0.35rem 0; }

    .method-badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        margin-top: 0.35rem;
        letter-spacing: 0.02em;
    }
    .badge-personalized { background: #1f3a2e; color: #6fd68a; }
    .badge-similarity   { background: #2a2650; color: #b3a7f7; }
    .badge-popularity   { background: #3a2e1f; color: #f0b96a; }
    .badge-lookup       { background: #1f2f3a; color: #7cc4f0; }
    .badge-none         { background: #2a2a2a; color: #9a9fa8; }

    .footer-pills { margin-top: 2rem; opacity: 0.6; font-size: 0.78rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

USER_AVATAR = "🧑"
ASSISTANT_AVATAR = "🎬"

METHOD_DISPLAY = {
    "personalized_als": ("👤 Personalized for you", "badge-personalized"),
    "item_similarity": ("🔁 Similar movies", "badge-similarity"),
    "popularity_fallback": ("🔥 Popular pick", "badge-popularity"),
    "lookup": ("ℹ️ Catalog lookup", "badge-lookup"),
    "not_found": ("⚠️ Movie not found", "badge-none"),
    "lookup_not_found": ("⚠️ Movie not found", "badge-none"),
}

LOADING_MESSAGES = [
    "Digging through the catalog...",
    "Checking what similar viewers liked...",
    "Running the recommender model...",
    "Finding something worth watching...",
    "Cross-checking the ratings data...",
]


@st.cache_resource
def load_recommender():
    return MovieRecommender(hf_repo_id=HF_REPO_ID)


@st.cache_resource
def load_client():
    return OpenAI(
        api_key=st.secrets["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )


recommender = load_recommender()
client = load_client()

# ---------- session state ----------
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
if "display_history" not in st.session_state:
    st.session_state.display_history = []  # each item: {role, content, method (optional)}
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# A turn is "pending" if the last message is from the user with no reply yet —
# whether that's because it was just added, or because a previous attempt to
# answer it got interrupted (e.g. the user clicked again before it finished).
# Checking this directly from the actual conversation state (instead of a
# separate is_generating flag) means the app can never get permanently stuck:
# any future rerun will just retry answering the same pending turn.
pending_turn = bool(
    st.session_state.display_history and st.session_state.display_history[-1]["role"] == "user"
)

# ---------- sidebar ----------
with st.sidebar:
    st.header("🎬 CineMatch")
    st.caption("A conversational movie recommender")

    st.markdown(
        "This chatbot combines a **real collaborative-filtering model** "
        "(trained on the MovieLens dataset) with an **LLM agent** that "
        "calls it as a tool — so recommendations come from real data, "
        "not the AI's imagination."
    )

    st.divider()
    st.subheader("Try asking:")

    example_prompts = [
        "I loved Inception, what should I watch next?",
        "What genre is Toy Story?",
        "Recommend something for user 1",
        "I'm bored, surprise me",
    ]
    for prompt in example_prompts:
        if st.button(prompt, use_container_width=True, disabled=pending_turn):
            st.session_state.pending_prompt = prompt

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True, disabled=pending_turn):
        st.session_state.conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        st.session_state.display_history = []
        st.rerun()

    st.divider()
    with st.expander("How it decides what to recommend"):
        st.markdown(
            "- **👤 Personalized** — uses your rating history via collaborative filtering\n"
            "- **🔁 Similar movies** — matches based on a movie you mentioned liking\n"
            "- **🔥 Popular pick** — falls back to well-rated catalog favorites\n"
            "- **ℹ️ Catalog lookup** — answers factual questions (genre, etc.)"
        )

    st.markdown(
        '<div class="footer-pills">Built with ALS collaborative filtering + an open-weight LLM (via Groq)</div>',
        unsafe_allow_html=True,
    )

# ---------- main chat area ----------
st.markdown('<p class="cine-title">🎬 CineMatch</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Tell me a movie you liked, or just ask for a recommendation.</p>', unsafe_allow_html=True)

for i, msg in enumerate(st.session_state.display_history):
    avatar = USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])
        if msg.get("method"):
            label, css_class = METHOD_DISPLAY.get(msg["method"], (None, None))
            if label:
                st.markdown(f'<span class="method-badge {css_class}">{label}</span>', unsafe_allow_html=True)
        if msg["role"] == "assistant":
            col1, col2, _ = st.columns([1, 1, 10])
            with col1:
                st.button("👍", key=f"up_{i}", help="Helpful")
            with col2:
                st.button("👎", key=f"down_{i}", help="Not helpful")

typed_input = st.chat_input("Ask me for a movie recommendation...", disabled=pending_turn)

if not pending_turn:
    new_input = st.session_state.pending_prompt or typed_input
    st.session_state.pending_prompt = None
    if new_input:
        st.session_state.display_history.append({"role": "user", "content": new_input})
        st.session_state.conversation_history.append({"role": "user", "content": new_input})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.write(new_input)
        pending_turn = True

if pending_turn:
    method = None
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner(random.choice(LOADING_MESSAGES)):
            try:
                reply, updated_history, tool_info = run_agent(
                    client=client,
                    recommender=recommender,
                    conversation_history=st.session_state.conversation_history,
                )
                st.session_state.conversation_history = updated_history
                method = tool_info[0]["result"].get("method") if tool_info else None
            except Exception as e:
                reply = f"Sorry, something went wrong talking to the model: {e}"
                method = None
                # Keep conversation_history in sync with display_history even on
                # failure — otherwise it's left with a dangling unanswered user
                # turn, and the next real message would stack two user turns in
                # a row when sent to the model.
                st.session_state.conversation_history.append({"role": "assistant", "content": reply})

        st.write(reply)
        if method:
            label, css_class = METHOD_DISPLAY.get(method, (None, None))
            if label:
                st.markdown(f'<span class="method-badge {css_class}">{label}</span>', unsafe_allow_html=True)

    st.session_state.display_history.append({"role": "assistant", "content": reply, "method": method})
    st.rerun()

if not st.session_state.display_history:
    st.info("👈 Try one of the example prompts in the sidebar, or type your own message below.")
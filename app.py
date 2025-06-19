# ────────────────────────────────────────────────────────────────────
# 3. Streamlit UI  – ChatGPT-like look & feel
# ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer-Problem Assistant",
    page_icon="💬",
    layout="centered",
)

# ---- Tiny CSS polish -------------------------------------------------
st.markdown(
    """
    <style>
    /* nicer padding for the whole page */
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    /* personalise assistant bubbles */
    .stChatMessage.avatar  {background:#f1f0f0}
    .stChatMessage.user    {background:#dcf8c6}
    /* hide Streamlit footer / hamburger if you like */
    #MainMenu {visibility:hidden;}
    footer   {visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.header("💬 JE AI Assistant")

if not HF_TOKEN:
    st.error("HF_TOKEN is not set.  Add it under  ➜  Settings → Secrets and reload.")
    st.stop()

# -------- Session state ----------------------------------------------
if "stage" not in st.session_state:
    st.session_state.stage   = "need_problem"          # → need_clarify → done
    st.session_state.chatlog = [
        {"role": "system",
         "content":
         "You are an internal support assistant for our company. "
         "Follow subsequent instructions carefully."}
    ]

# ---------------------------------------------------------------------
# Helper to render every stored turn with bubbles/avatars
# ---------------------------------------------------------------------
def render_chat():
    for m in st.session_state.chatlog:
        if m["role"] == "system":
            # Usually we do NOT show system messages
            continue
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

# ---------------------------------------------------------------------
# The main flow ↓
# ---------------------------------------------------------------------
render_chat()          # always show conversation so far
user_prompt = None     # initialise

# ---- Stage 1 : get the problem --------------------------------------
if st.session_state.stage == "need_problem":
    user_prompt = st.chat_input(
        placeholder="Describe the customer's problem and hit Enter …",
        key="problem_input"
    )
    if user_prompt:
        st.session_state.chatlog.append({"role": "user", "content": user_prompt.strip()})
        st.session_state.chatlog.append({
            "role": "system",
            "content":
            "Ask the user 4-8 concise clarifying questions using the 5W1H method "
            "(Who, What, When, Where, Why, How). Number the questions."
        })
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                assistant = llm_chat(st.session_state.chatlog, max_new_tokens=256)
                st.markdown(assistant)
        st.session_state.chatlog.append({"role": "assistant", "content": assistant})
        st.session_state.stage = "need_clarify"
        st.experimental_rerun()

# ---- Stage 2 : clarify ----------------------------------------------
elif st.session_state.stage == "need_clarify":
    user_prompt = st.chat_input(
        placeholder="Answer the assistant’s numbered questions here …",
        key="clarify_input"
    )
    if user_prompt:
        st.session_state.chatlog.append({"role": "user", "content": user_prompt.strip()})
        st.session_state.chatlog.append({
            "role": "system",
            "content":
            "Analyse the conversation so far.\n"
            "1. List the most plausible root causes of the user's problem (bulleted).\n"
            "2. For each cause, suggest practical solutions or next steps.\n"
            "3. Keep the tone professional and concise."
        })
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                assistant = llm_chat(st.session_state.chatlog, max_new_tokens=512)
                st.markdown(assistant)
        st.session_state.chatlog.append({"role": "assistant", "content": assistant})
        st.session_state.stage = "done"
        st.experimental_rerun()

# ---- Stage 3 : show diagnosis ---------------------------------------
elif st.session_state.stage == "done":
    with st.chat_message("assistant"):
        st.success("Here is my analysis.  Let me know if I can help further:")
        st.markdown(st.session_state.chatlog[-1]["content"])

    if st.button("🔄  Start a new analysis"):
        for k in ("stage", "chatlog"):
            st.session_state.pop(k, None)
        st.experimental_rerun()

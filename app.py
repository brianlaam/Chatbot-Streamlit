# ────────────────────────────────────────────────────────────────────
# 0.  Imports & constants
# ────────────────────────────────────────────────────────────────────
import os
import re
import json
import time
import requests
import streamlit as st

HF_API_URL = (
    "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
)
HF_TOKEN = st.secrets["HUGGINGFACE_API_TOKEN"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# ────────────────────────────────────────────────────────────────────
# 1.  Helper ─ Hugging Face Inference API
# ────────────────────────────────────────────────────────────────────
def hf_generate(
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Send one prompt to the endpoint and return ONLY the newly
    generated text (original prompt stripped)."""
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": True,
            "top_p": 0.95,
            "repetition_penalty": 1.1,
        },
    }

    r = requests.post(
        HF_API_URL, headers=HEADERS, json=payload, timeout=180
    )
    if r.status_code == 503:  # model is loading
        with st.spinner("Model is loading on the HuggingFace server…"):
            time.sleep(10)
            r = requests.post(
                HF_API_URL, headers=HEADERS, json=payload, timeout=180
            )
    r.raise_for_status()

    data = r.json()
    # HF returns: [{"generated_text": "..."}]
    full_text = data[0]["generated_text"]
    return full_text[len(prompt) :].lstrip()  # strip the prompt part


# ────────────────────────────────────────────────────────────────────
# 2.  Prompt builder (Mistral-/Zephyr-style <s>[INST] … [/INST])
# ────────────────────────────────────────────────────────────────────
def build_prompt(messages: list[dict]) -> str:
    """
    Turn messages = [{"role": "...", "content": "..."}] into one prompt
    string that follows the <s>[INST] ... [/INST] chat format.
    """
    prompt = ""
    for m in messages:
        role, content = m["role"], m["content"].strip()
        if role in ("system", "user"):
            prompt += f"<s>[INST] {content} [/INST]"
        elif role == "assistant":
            prompt += f" {content} "
    # The model should now answer the last user/system block:
    return prompt + " "


def llm_chat(messages, **gen_kw):
    prompt = build_prompt(messages)
    reply = hf_generate(prompt, **gen_kw)
    return reply


# ────────────────────────────────────────────────────────────────────
# 3.  Output sanitiser  ← NEW
# ────────────────────────────────────────────────────────────────────
TAG_PATTERN = re.compile(r"\[/?(INST|USER|SYSTEM|ASSISTANT)\]")

def clean_response(text: str) -> str:
    """Remove any leftover tag-like tokens the model may output."""
    return TAG_PATTERN.sub("", text).strip()


# ────────────────────────────────────────────────────────────────────
# 4.  Streamlit UI
# ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JE AI Assistant",
    page_icon="💬",
    layout="centered",
)

# ---- Tiny CSS polish ------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .stChatMessage.avatar  {background:#f1f0f0}
    .stChatMessage.user    {background:#dcf8c6}
    #MainMenu {visibility:hidden;}
    footer   {visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.header("💬 JE AI Assistant")

if not HF_TOKEN:
    st.error(
        "HUGGINGFACE_API_TOKEN is not set. "
        "Add it under ➜  Settings → Secrets and reload."
    )
    st.stop()

# --------------------------------------------------------------------
# 5.  Session state
# --------------------------------------------------------------------
if "stage" not in st.session_state:
    st.session_state.stage = "need_problem"  # → need_clarify → done
    st.session_state.chatlog = [
        {
            "role": "system",
            "content": (
                "You are an experienced quality manager (30 years). "
                "The user may encounter technical problems. "
                "Please guide the user to use the 4M approach "
                "(Man, Machine, Material, Method) within the 8D "
                "Problem-Solving process. "
                "Assist the user in developing Interim Containment "
                "Actions (D3). "
                "Follow subsequent instructions carefully. "
                "Do not show role markers in your answer."
            ),
        }
    ]

# ────────────────────────────────────────────────────────────────────
# 6.  Helper to render the chat so far
# ────────────────────────────────────────────────────────────────────
def render_chat():
    for m in st.session_state.chatlog:
        if m["role"] == "system":  # never show system messages
            continue
        with st.chat_message(m["role"]):
            st.markdown(m["content"])


# ────────────────────────────────────────────────────────────────────
# 7.  Main flow
# ────────────────────────────────────────────────────────────────────
render_chat()

if st.session_state.stage == "need_problem":
    user_prompt = st.chat_input(
        placeholder="Describe your problem in a few sentences…",
        key="problem_input",
    )
    if user_prompt:
        st.session_state.chatlog.append(
            {"role": "user", "content": user_prompt.strip()}
        )
        # Add a hidden instruction for the assistant
        st.session_state.chatlog.append(
            {
                "role": "system",
                "content": (
                    "Use tools like the 5 Whys to identify and verify "
                    "root causes. Propose permanent corrective actions, "
                    "and guide me through their implementation and "
                    "validation. Finally, suggest ways to modify "
                    "processes to prevent recurrence."
                ),
            }
        )
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                raw_answer = llm_chat(
                    st.session_state.chatlog, max_new_tokens=256
                )
                answer = clean_response(raw_answer)  # ← NEW
                st.markdown(answer)
        st.session_state.chatlog.append({"role": "assistant", "content": answer})
        st.session_state.stage = "need_clarify"
        st.rerun()

elif st.session_state.stage == "need_clarify":
    user_prompt = st.chat_input(
        placeholder="Let me know more details or ask follow-up questions…",
        key="clarify_input",
    )
    if user_prompt:
        st.session_state.chatlog.append(
            {"role": "user", "content": user_prompt.strip()}
        )
        st.session_state.chatlog.append(
            {
                "role": "system",
                "content": (
                    "Gather the user's information and analyse the "
                    "conversation. List the most plausible root causes "
                    "of the user's problem in bullet points. For each "
                    "possible cause, suggest practical solutions or "
                    "next steps. Keep the tone professional and concise."
                ),
            }
        )
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                raw_answer = llm_chat(
                    st.session_state.chatlog, max_new_tokens=512
                )
                answer = clean_response(raw_answer)  # ← NEW
                st.markdown(answer)
        st.session_state.chatlog.append({"role": "assistant", "content": answer})
        st.session_state.stage = "done"
        st.rerun()

# (Optional) you may re-enable the "done" stage or a debug expander

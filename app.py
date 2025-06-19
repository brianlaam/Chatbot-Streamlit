# ────────────────────────────────────────────
# 0. Imports & constants
# ────────────────────────────────────────────
import os, re, time, requests, streamlit as st

HF_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HF_TOKEN   = st.secrets["HUGGINGFACE_API_TOKEN"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# ────────────────────────────────────────────
# 1. HF helper
# ────────────────────────────────────────────
def hf_generate(prompt, max_new_tokens=2048, temperature=0.7):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens":     max_new_tokens,
            "temperature":        temperature,
            "do_sample":          True,
            "top_p":              0.95,
            "repetition_penalty": 1.1,
        },
    }
    r = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=180)
    if r.status_code == 503:
        with st.spinner("Model loading…"): time.sleep(10)
        r = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=180)
    r.raise_for_status()
    full = r.json()[0]["generated_text"]
    return full[len(prompt):].lstrip()

# ────────────────────────────────────────────
# 2. Prompt builder  (single system block!)
# ────────────────────────────────────────────
def build_prompt(messages):
    """Combine all system msgs into one, then alternate user/assistant."""
    system_prefix = ""
    chat_pairs    = []
    for m in messages:
        role, txt = m["role"], m["content"].strip()
        if role == "system":
            system_prefix += txt + "\n"
        elif role == "user":
            chat_pairs.append((txt, None))          # placeholder for answer
        else:                                       # assistant
            if chat_pairs:
                chat_pairs[-1] = (chat_pairs[-1][0], txt)

    prompt = f"<s>[INST] {system_prefix.strip()}\n" \
             f"{chat_pairs[-1][0]} [/INST]"         # last user request
    for user, assistant in chat_pairs[:-1]:
        prompt += f" {assistant.strip()} </s><s>[INST] {user} [/INST]"
    return prompt + " "

def llm_chat(msgs, **kw): return hf_generate(build_prompt(msgs), **kw)

# ────────────────────────────────────────────
# 3. Strip stray tags the model might emit
# ────────────────────────────────────────────
TAG_RE = re.compile(r"\[/?(INST|USER|SYSTEM|ASSISTANT)\]")
def clean(text): return TAG_RE.sub("", text).strip()

# ────────────────────────────────────────────
# 4. Streamlit UI
# ────────────────────────────────────────────
st.set_page_config(page_title="JE AI Assistant", page_icon="💬")
st.header("💬 JE AI Assistant")

if not HF_TOKEN:
    st.error("Add HUGGINGFACE_API_TOKEN under Settings → Secrets.")
    st.stop()

if "log" not in st.session_state:
    st.session_state.stage = "need_problem"
    st.session_state.log = [{
        "role": "system",
        "content": (
            "You are an internal support assistant for our company. "
            "Follow subsequent instructions carefully."),
    }]

def show_chat():
    for m in st.session_state.log:
        if m["role"] == "system": continue
        with st.chat_message(m["role"]): st.markdown(m["content"])
show_chat()

# ---------- Stage 1 : get the problem -------------------------------
if st.session_state.stage == "need_problem":
    user_in = st.chat_input("Describe the problem …")
    if user_in:
        st.session_state.log.append({"role": "user", "content": user_in})
        # hidden instruction goes ONLY to this call, not into log
        hidden_sys = {
            "role": "system",
            "content": (
                "Ask the user 4-8 concise clarifying questions using the 5W1H method "
                "(Who, What, When, Where, Why, How). Number the questions."
                "Follow all later instructions and do not show role markers."
            ),
        }
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                reply = clean(
                    llm_chat(st.session_state.log + [hidden_sys], max_new_tokens=300)
                )
                st.markdown(reply)
        st.session_state.log.append({"role": "assistant", "content": reply})
        st.session_state.stage = "need_clarify"
        st.rerun()

# ---------- Stage 2 : follow-up questions ---------------------------
elif st.session_state.stage == "need_clarify":
    user_in = st.chat_input("Any more details or questions?")
    if user_in:
        st.session_state.log.append({"role": "user", "content": user_in})
        hidden_sys = {
            "role": "system",
            "content": (
                "Guide the user to apply the 4M approach (Man, Machine, Material, Method) to solve the problem(s)"
                "Help them develop Interim Containment Actions"
            ),
        }
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                reply = clean(
                    llm_chat(st.session_state.log + [hidden_sys], max_new_tokens=400)
                )
                st.markdown(reply)
        st.session_state.log.append({"role": "assistant", "content": reply})
        st.session_state.stage = "done"
        st.rerun()

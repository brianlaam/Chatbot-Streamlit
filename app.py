import os, time, requests, streamlit as st

# ────────────────────────────── 1. CONFIG ─────────────────────────────
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
HF_TOKEN = os.getenv("HF_TOKEN")                            # via secrets
HEADERS  = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# Tune these so we stay within the public queue limits
MAX_NEW_TOKENS_1 = 128     # for the 5W1H question turn
MAX_NEW_TOKENS_2 = 256     # for the final answer


# ────────────────────────────── 2.  LOW-LEVEL CALL ────────────────────
def call_hf(prompt: str, max_new: int, temperature: float = 0.7) -> str:
    """
    Send a prompt to the HuggingFace inference-endpoint and
    return ONLY the newly-generated text.  Raises a RuntimeError with
    a clean explanation if the endpoint responds with an error.
    """
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens":   max_new,
            "temperature":      temperature,
            "top_p":            0.95,
            "do_sample":        True,
            "repetition_penalty": 1.1,
        },
        # Ask the HF queue to wait (up to 2 min) until a GPU is free
        "options": {"wait_for_model": True}
    }

    try:
        r = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=180)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error while contacting HF API: {e}")

    #  ----- handle non-200 codes ourselves so we can show useful info
    if r.status_code != 200:
        try:
            detail = r.json().get("error", r.text)
        except ValueError:
            detail = r.text
        raise RuntimeError(f"HF API returned {r.status_code}: {detail}")

    data = r.json()

    # HF sometimes returns {"error": "..."} with 200 – catch that too
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"HF API error: {data['error']}")

    full_text = data[0]["generated_text"]
    return full_text[len(prompt):].lstrip()


# ────────────────────────────── 3.  PROMPT UTILS ──────────────────────
def build_prompt(messages: list[dict]) -> str:
    """
    Convert the chat list into Mistral <s>[INST] ... [/INST] format.
    """
    txt = ""
    for m in messages:
        role, content = m["role"], m["content"].strip()
        if role in ("system", "user"):
            txt += f"<s>[INST] {content} [/INST]"
        else:                       # assistant
            txt += f" {content} "
    return txt + " "

def llm_complete(msgs, max_new):
    prompt = build_prompt(msgs)
    return call_hf(prompt, max_new=max_new)


# ────────────────────────────── 4.  STREAMLIT UI ──────────────────────
st.set_page_config(page_title="Customer-Problem Assistant", page_icon="💬")
st.title("💬  Internal AI Troubleshooting Assistant")

if not HF_TOKEN:
    st.error("HF_TOKEN is not set – add it under Settings → Secrets.")
    st.stop()

if "stage" not in st.session_state:
    st.session_state.stage   = "need_problem"  # -> need_clarify -> done
    st.session_state.chatlog = [
        {"role": "system",
         "content": "You are an internal support assistant. "
                    "Follow instructions carefully."}
    ]

# ───────── Stage 1: user describes the problem ────────────────────────
if st.session_state.stage == "need_problem":
    problem = st.text_area(
        "Describe the customer's problem:",
        placeholder="e.g. Mobile app crashes when user tries to upload …"
    )
    if st.button("Submit problem", disabled=not problem.strip()):
        st.session_state.chatlog.append({"role": "user", "content": problem.strip()})
        st.session_state.chatlog.append({
            "role": "system",
            "content": (
                "Ask the user 4-8 concise clarifying questions using the 5W1H "
                "method (Who, What, When, Where, Why, How). Number the questions."
            )
        })
        with st.spinner("Generating clarifying questions…"):
            try:
                reply = llm_complete(st.session_state.chatlog,
                                     max_new=MAX_NEW_TOKENS_1)
            except RuntimeError as e:
                st.error(str(e))
                st.stop()
        st.session_state.chatlog.append({"role": "assistant", "content": reply})
        st.session_state.stage = "need_clarify"
        st.experimental_rerun()

# ───────── Stage 2: user answers, model diagnoses ──────────────────────
elif st.session_state.stage == "need_clarify":
    st.subheader("Assistant questions")
    st.markdown(st.session_state.chatlog[-1]["content"])
    answers = st.text_area("Your answers:")
    if st.button("Submit answers", disabled=not answers.strip()):
        st.session_state.chatlog.append({"role": "user", "content": answers.strip()})
        st.session_state.chatlog.append({
            "role": "system",
            "content": (
                "Analyse the conversation so far.\n"
                "1. Bullet the most plausible root causes.\n"
                "2. For each cause, propose practical solutions.\n"
                "3. Keep tone professional and concise."
            )
        })
        with st.spinner("Thinking…"):
            try:
                reply = llm_complete(st.session_state.chatlog,
                                     max_new=MAX_NEW_TOKENS_2)
            except RuntimeError as e:
                st.error(str(e))
                st.stop()
        st.session_state.chatlog.append({"role": "assistant", "content": reply})
        st.session_state.stage = "done"
        st.experimental_rerun()

# ───────── Stage 3: show final suggestions ─────────────────────────────
elif st.session_state.stage == "done":
    st.success("Possible causes and solutions")
    st.markdown(st.session_state.chatlog[-1]["content"])
    if st.button("Start new analysis"):
        for key in ("stage", "chatlog"):
            st.session_state.pop(key, None)
        st.experimental_rerun()

# ───────── Optional debug view ─────────────────────────────────────────
with st.expander("🔎 Debug conversation log"):
    for m in st.session_state.chatlog:
        st.write(f"**{m['role'].upper()}**: {m['content']}")

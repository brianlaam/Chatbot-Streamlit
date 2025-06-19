import os
import json
import time
import requests
import streamlit as st

# -------------------------------------------------------------------
# 1. Hugging Face Inference-API helper
# -------------------------------------------------------------------
HF_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HF_TOKEN = st.secrets["HUGGINGFACE_API_TOKEN"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
HEADERS    = {"Authorization": f"Bearer {HF_TOKEN}"}

def hf_generate(prompt: str,
                max_new_tokens: int = 2048,
                temperature: float = 0.7) -> str:
    """
    Send one prompt to the endpoint and return only the newly
    generated text (i.e. without the original prompt).
    """
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature":   temperature,
            "do_sample":     True,
            "top_p":         0.95,
            "repetition_penalty": 1.1,
        }
    }

    r = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=180)
    if r.status_code == 503:          # model is loading
        with st.spinner("Model is loading on the HuggingFace server…"):
            time.sleep(10)
            r = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=180)
    r.raise_for_status()

    data = r.json()
    # HF returns a list with 1 dict → {"generated_text": "..."}
    full_text = data[0]["generated_text"]
    return full_text[len(prompt):].lstrip()   # strip the prompt part


# -------------------------------------------------------------------
# 2. Very small template replicating chat format
# -------------------------------------------------------------------
def build_prompt(messages: list[dict]) -> str:
    """
    Turn messages = [{"role": "...", "content": "..."}] into one prompt string
    that follows the <s>[INST] ... [/INST] format Mistral-Instruct expects.
    """
    prompt = ""
    for m in messages:
        role, content = m["role"], m["content"].strip()
        if role in ("system", "user"):
            prompt += f"<s>[INST] {content} [/INST]"
        elif role == "assistant":
            prompt += f" {content} "
    # Last assistant turn is what we want the model to generate now:
    return prompt + " "

def llm_chat(messages, **gen_kw):
    prompt = build_prompt(messages)
    reply  = hf_generate(prompt, **gen_kw)
    return reply


# -------------------------------------------------------------------
# 3. Streamlit UI
# -------------------------------------------------------------------
st.set_page_config(page_title="JE AI Assistant", page_icon="💬")
st.title("JE AI Assistant")

if not HF_TOKEN:
    st.error("HF_TOKEN is not set.  Add it under *Settings → Secrets* and reload.")
    st.stop()

# -------- Session state -------------------------------------------------
if "stage" not in st.session_state:
    st.session_state.stage    = "need_problem"    # → need_clarify → done
    st.session_state.chatlog  = [
        {"role": "system",
         "content":
         "You are an internal support assistant for our company. "
         "Follow subsequent instructions carefully."}
    ]

# -------- Stage 1 : get initial problem --------------------------------
if st.session_state.stage == "need_problem":
    problem = st.text_area(
        "Describe the customer's problem:",
        placeholder="e.g. Mobile app crashes when user tries to upload a file…"
    )
    if st.button("Submit problem", disabled=not problem.strip()):
        st.session_state.chatlog.append({"role": "user", "content": problem.strip()})
        # Tell the model what we want next:
        st.session_state.chatlog.append({
            "role": "system",
            "content":
            "Ask the user 4-8 concise clarifying questions using the 5W1H method "
            "(Who, What, When, Where, Why, How). Number the questions."
        })
        assistant = llm_chat(st.session_state.chatlog, max_new_tokens=256)
        st.session_state.chatlog.append({"role": "assistant", "content": assistant})
        st.session_state.stage = "need_clarify"
        st.rerun()

# -------- Stage 2 : display 5W1H questions, collect answers ------------
elif st.session_state.stage == "need_clarify":
    st.subheader("Assistant questions")
    st.markdown(st.session_state.chatlog[-1]["content"])
    answers = st.text_area("Your answers:")
    if st.button("Submit answers", disabled=not answers.strip()):
        st.session_state.chatlog.append({"role": "user", "content": answers.strip()})
        st.session_state.chatlog.append({
            "role": "system",
            "content":
            "Analyse the conversation so far.\n"
            "1. List the most plausible root causes of the user's problem (bulleted).\n"
            "2. For each cause, suggest practical solutions or next steps.\n"
            "3. Keep the tone professional and concise."
        })
        assistant = llm_chat(st.session_state.chatlog, max_new_tokens=512)
        st.session_state.chatlog.append({"role": "assistant", "content": assistant})
        st.session_state.stage = "done"
        st.rerun()

# # -------- Stage 3 : show diagnosis -------------------------------------
# elif st.session_state.stage == "done":
#     st.success("Possible causes and solutions")
#     st.markdown(st.session_state.chatlog[-1]["content"])
#     if st.button("Start new analysis"):
#         for k in ("stage", "chatlog"):
#             st.session_state.pop(k, None)
#         st.rerun()

# # -------- Optional: expandable debug log -------------------------------
# with st.expander("🔎 Debug conversation log"):
#     for m in st.session_state.chatlog:
#         st.write(f"**{m['role'].upper()}**: {m['content']}")

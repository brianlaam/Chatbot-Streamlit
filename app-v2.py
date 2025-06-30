import os, json, time, yaml, requests, streamlit as st
from dotenv import load_dotenv
load_dotenv()

# ---------- 0. Config & secrets ---------------------------------
with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

HF_MODEL   = os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HF_TOKEN   = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HEADERS    = {"Authorization": f"Bearer {HF_TOKEN}"}

# ---------- 1. Helper functions ---------------------------------
@st.cache_data(show_spinner=False)
def hf_generate(prompt, **p):
    payload = {"inputs": prompt,
               "parameters": {**{
                   "max_new_tokens": 512, "temperature": .7,
                   "top_p": .95, "repetition_penalty": 1.1}, **p}}
    rsp = requests.post(HF_API_URL, headers=HEADERS,
                        json=payload, timeout=180)
    rsp.raise_for_status()
    return rsp.json()[0]["generated_text"][len(prompt):].lstrip()

def build_prompt(msgs):
    out = ""
    for m in msgs:
        role, txt = m["role"], m["content"].strip()
        if role in ("system", "user"):
            out += f"<s>[INST] {txt} [/INST]"
        else:
            out += f" {txt} "
    return out + " "

def export_chat(chatlog):
    st.download_button("⬇️  Export chat (MD)",
        data="\n\n".join(f"**{m['role']}**: {m['content']}"
                         for m in chatlog if m['role']!="system"),
        file_name="chat.md")

# ---------- 2. UI ------------------------------------------------
st.set_page_config(page_title=CFG["app_name"],
                   page_icon=CFG["page_icon"])
if not HF_TOKEN:
    st.error("Add HUGGINGFACEHUB_API_TOKEN to .env")
    st.stop()

if "chat" not in st.session_state:
    st.session_state.chat = [
        {"role":"system",
         "content":CFG["system_prompts"]["quality"]}]

for m in st.session_state.chat:
    if m["role"] != "system":
        st.chat_message(m["role"]).markdown(m["content"])

prompt = st.chat_input("Write your message…")
if prompt:
    st.session_state.chat.append({"role":"user","content":prompt})
    with st.spinner("Thinking…"):
        assistant = hf_generate(build_prompt(st.session_state.chat))
    st.session_state.chat.append({"role":"assistant","content":assistant})
    st.rerun()

export_chat(st.session_state.chat)

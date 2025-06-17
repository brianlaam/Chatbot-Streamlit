import os, torch, streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# ---------- 1. Load model once & cache ----------
@st.cache_resource(show_spinner="Loading Mistral-7B …")
def load_model():
    dtype = torch.float16              # ← fp16 on most GPUs
    # For CPU or low-VRAM GPU you can do:
    #   from transformers import BitsAndBytesConfig
    #   bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    #   model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_cfg, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    return tokenizer, model

tokenizer, model = load_model()

# ---------- 2. Helper to call the LLM ----------
def chat_complete(msgs, max_new=512, temperature=0.7):
    """msgs = list[dict(role,content)]  -> assistant reply str"""
    input_ids = tokenizer.apply_chat_template(
        msgs, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)

    out = model.generate(
        input_ids,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    # The assistant reply is everything after the last <assistant> tag:
    return full.split("<assistant>")[-1].strip()

# ---------- 3. Streamlit UI ----------
st.set_page_config(page_title="Customer-Problem Assistant", page_icon="💬")
st.title("💬  Internal AI Troubleshooting Assistant")

# Session memory ----------------------------------------------------------
if "stage" not in st.session_state:
    # stages: "need_problem" → "need_clarify" → "done"
    st.session_state.stage      = "need_problem"
    st.session_state.messages   = [
        {"role":"system",
         "content":"You are an internal support assistant. You follow the instructions below."}
    ]

# ----------- Stage 1 : initial problem -----------------------------------
if st.session_state.stage == "need_problem":
    problem = st.text_area("Describe the customer's problem",
                           placeholder="e.g. Our mobile app crashes whenever …")
    if st.button("Submit problem", disabled=not problem.strip()):
        st.session_state.messages.append({"role":"user", "content": problem.strip()})
        # Add a system instruction telling the LLM WHAT we want next:
        st.session_state.messages.append({
            "role":"system",
            "content":(
                "Ask the user clarifying questions using the 5W1H method "
                "(Who, What, When, Where, Why, How). "
                "Return 4-8 concise, numbered questions."
            )
        })
        assistant = chat_complete(st.session_state.messages, max_new=256)
        st.session_state.messages.append({"role":"assistant", "content": assistant})
        st.session_state.stage = "need_clarify"
        st.experimental_rerun()    # jump to next stage immediately

# ----------- Stage 2 : user answers 5W1H ---------------------------------
elif st.session_state.stage == "need_clarify":
    st.subheader("Assistant questions")
    st.markdown(st.session_state.messages[-1]["content"])
    clarifications = st.text_area("Your answers",
                                  placeholder="Answer each question here …")
    if st.button("Submit answers", disabled=not clarifications.strip()):
        st.session_state.messages.append({"role":"user", "content": clarifications.strip()})
        # Next instruction: find causes + solutions
        st.session_state.messages.append({
            "role":"system",
            "content":(
                "Analyse the entire conversation.\n"
                "1. List the most plausible root causes (bulleted).\n"
                "2. For each cause, suggest practical solutions or next steps.\n"
                "3. Keep the tone professional and concise."
            )
        })
        assistant = chat_complete(st.session_state.messages, max_new=512)
        st.session_state.messages.append({"role":"assistant", "content": assistant})
        st.session_state.stage = "done"
        st.experimental_rerun()

# ----------- Stage 3 : show diagnostic -----------------------------------
elif st.session_state.stage == "done":
    st.success("Here are possible causes and solutions:")
    st.markdown(st.session_state.messages[-1]["content"])

    if st.button("Start new analysis"):
        for k in ("stage", "messages"): st.session_state.pop(k, None)
        st.experimental_rerun()

# ---------- (Optional) conversation log for admins -----------------------
with st.expander("🗒  Internal debug log", expanded=False):
    for m in st.session_state.messages:
        st.write(f"{m['role'].upper()}: {m['content']}")
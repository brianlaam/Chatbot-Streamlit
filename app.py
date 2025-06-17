import streamlit as st
from transformers import pipeline
from huggingface_hub import InferenceClient

# Access HuggingFace API token from Streamlit secrets
hf_token = st.secrets["HUGGINGFACE_API_TOKEN"]

# Initialize HuggingFace Inference Client
client = InferenceClient(token=hf_token)

# Streamlit app layout
st.title("Customer Problem Solver Chatbot")
st.write("Describe the problem faced by your customer, and I'll help identify reasons and solutions.")

# Initialize session state to store conversation
if "step" not in st.session_state:
    st.session_state.step = 1
    st.session_state.problem = ""
    st.session_state.questions = []
    st.session_state.answers = []

# Step 1: User inputs the customer problem
if st.session_state.step == 1:
    with st.form("problem_form"):
        problem = st.text_area("Enter the customer's problem:", height=100)
        submitted = st.form_submit_button("Submit Problem")
        if submitted and problem:
            st.session_state.problem = problem

            # Generate 5W1H questions using LLM
            prompt = f"""
            A customer reported the following problem: "{problem}"
            Generate 5W1H (Who, What, When, Where, Why, How) questions to gather more details about this problem. Provide the questions as a bulleted list.
            """
            try:
                response = client.text_generation(
                    prompt,
                    model="facebook/blenderbot-400M-distill",
                    max_length=200,
                    temperature=0.7
                )
                questions = response.strip().split("\n")
                st.session_state.questions = [q for q in questions if q.startswith("- ")]
                st.session_state.step = 2
                st.rerun()
            except Exception as e:
                st.error(f"Error generating questions: {e}")

# Step 2: Display questions and collect answers
if st.session_state.step == 2:
    st.write("Please answer the following questions to provide more details:")
    with st.form("answers_form"):
        answers = []
        for i, question in enumerate(st.session_state.questions):
            answer = st.text_input(f"{question}", key=f"q{i}")
            answers.append(answer)
        submitted = st.form_submit_button("Submit Answers")
        if submitted and all(answers):
            st.session_state.answers = answers
            st.session_state.step = 3
            st.rerun()

# Step 3: Generate reasons and solutions
if st.session_state.step == 3:
    st.write("### Analysis of the Problem")
    st.write(f"**Customer Problem**: {st.session_state.problem}")
    st.write("**Provided Details**:")
    for q, a in zip(st.session_state.questions, st.session_state.answers):
        st.write(f"{q}: {a}")

    # Generate reasons and solutions using LLM
    details = "\n".join([f"{q}: {a}" for q, a in zip(st.session_state.questions, st.session_state.answers)])
    prompt = f"""
    A customer reported the following problem: "{st.session_state.problem}"
    Additional details provided:
    {details}

    Analyze the problem and provide:
    - A bulleted list of possible reasons for the problem.
    - A bulleted list of suggested solutions to address the problem.
    """
    try:
        response = client.text_generation(
            prompt,
            model="facebook/blenderbot-400M-distill",
            max_length=400,
            temperature=0.7
        )
        st.write("### Possible Reasons and Suggested Solutions")
        st.markdown(response.strip())
    except Exception as e:
        st.error(f"Error generating reasons: {e}")

    if st.button("Start Over"):
        st.session_state.step = 1
        st.session_state.clear()
        st.rerun()

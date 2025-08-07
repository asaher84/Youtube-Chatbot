import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled  # type:ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- LLM Setup ---

llm = ChatGroq(,
               model_name='gemma2-9b-it')  # type:ignore

# --- Helper Functions ---


def get_transcript_and_retriever(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(
            video_id, languages=["en", "en-US", "hi", "mr",
                                 "de", "ru", "es", "zh-Hans", "Japanese", "ja"]
        )
        transcript = " ".join(chunk["text"] for chunk in transcript_list)

    except TranscriptsDisabled:
        raise ValueError("No caption available for this video.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-small")
    vector_store = FAISS.from_documents(chunks, embeddings)

    retriever = vector_store.as_retriever(
        search_type='similarity', search_kwargs={'k': 4}
    )
    return retriever


def build_qa_chain(retriever):
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.
        {context}
        Question: {question}
        """,
        input_variables=['context', 'question']
    )

    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })
    parser = StrOutputParser()

    return parallel_chain | prompt | llm | parser


# --- Streamlit UI ---
st.set_page_config(page_title="🎬 YouTube ChatBot", layout="centered")
st.title("🎬 YouTube ChatBot")
st.markdown(
    "Ask questions about any YouTube video by providing its video ID below.")

# Initialize session state to preserve retriever and chain across multiple questions
if 'qa' not in st.session_state:
    st.session_state.qa = None
    st.session_state.current_video_id = None

with st.sidebar:
    st.header("📽️ Load YouTube Video")
    video_id = st.text_input(
        "Enter Video ID (e.g., dQw4w9WgXcQ)", max_chars=20)

    if st.button("🔄 Load Transcript"):
        if video_id:

            # Show YouTube video preview and thumbnail

            video_url = f"https://www.youtube.com/watch?v={video_id}"
            # thumbnail_url = f"https://img.youtube.com/vi/{video_id}/0.jpg"

            st.markdown("### 🎞️ Video Preview")
            st.video(video_url)

            # st.markdown("### 🖼️ Thumbnail")
            # st.image(thumbnail_url, width=400)

            try:
                with st.spinner("Loading transcript and setting up QA..."):
                    retriever = get_transcript_and_retriever(video_id)
                    qa_chain = build_qa_chain(retriever)
                    st.session_state.qa = qa_chain
                    st.session_state.current_video_id = video_id
                st.success("✅ Transcript loaded successfully!")
            except Exception as e:
                st.session_state.qa = None
                st.error(f"❌ Failed to load transcript: {e}")
        else:
            st.warning("Please enter a valid YouTube Video ID.")

# Main interaction area
if st.session_state.qa:
    st.subheader("❓ Ask your question about the video:")
    question = st.text_input("Your Question:")

    if question:
        with st.spinner("Thinking..."):
            response = st.session_state.qa.invoke(question)
        st.markdown("### ✅ Answer:")
        st.markdown(f"**{response}**")
else:
    st.info("Load a video from the sidebar to get started.")


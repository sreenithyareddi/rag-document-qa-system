import streamlit as st
import os
import tempfile
from openai import OpenAI
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid

GROQ_API_KEY = st.secrets.get(
    "GROQ_API_KEY",
    os.getenv("GROQ_API_KEY")
)
client = Groq(api_key=GROQ_API_KEY)
if not GROQ_API_KEY:
    st.error("Groq API key not found.")
    st.stop()
CHROMA_DIR     = "./chroma_db"
COLLECTION     = "rag_documents"


@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return embed_model, chroma_client

embed_model, chroma_client = load_models()


def ingest_file(file_path, collection):
    ext = file_path.split(".")[-1].lower()
    if ext == "pdf":
        docs = PyPDFLoader(file_path).load()
    elif ext == "docx":
        docs = Docx2txtLoader(file_path).load()
    else:
        docs = TextLoader(file_path).load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    texts  = [c.page_content for c in chunks]
    metas  = [{"source": file_path, "page": str(c.metadata.get("page", "?"))} for c in chunks]
    embeds = embed_model.encode(texts).tolist()

    
    collection.add(
        ids=[str(uuid.uuid4()) for _ in texts],
        documents=texts,
        embeddings=embeds,
        metadatas=metas
    )
    return len(texts)


def answer_question(query, collection, history):

    summary_keywords = [
    "summary", "summarize", "overview",
    "main idea", "brief", "tldr",
    "what is this document about",
    "explain this document",
    "describe the document"
    ]

    is_summary = any(
        keyword in query.lower()
        for keyword in summary_keywords
    )

    if is_summary:
        results = collection.get(
            include=["documents", "metadatas"]
        )

        context = "\n\n".join(
            results["documents"][:50]  
        )

        sources = results["metadatas"][:50]

    else:
        q_embed = embed_model.encode([query]).tolist()

        results = collection.query(
            query_embeddings=q_embed,
            n_results=10,
            include=["documents", "metadatas"]
        )

        context = "\n\n".join(results["documents"][0][:5])
        sources = results["metadatas"][0] 
    prompt = f"""
You are an intelligent document assistant.

Answer the user's question using the provided context.

Be concise and direct.

If the answer cannot be found in the context, reply exactly:

"I could not find the answer in the uploaded document."

Context:
{context}

Question:
{query}

Answer:
"""
    
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        answer = resp.choices[0].message.content

    except Exception as e:
        answer = f"⚠️ API Error: {str(e)}"
    return answer, sources
# ── UI ───────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Document QA",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 1rem;
    padding-left: 3rem;
    padding-right: 3rem;
    max-width: 1200px;
}

/* Chat messages */
div[data-testid="stChatMessage"] {
    padding: 1rem;
    border-radius: 15px;
    margin-bottom: 1rem;
}

/* User message */
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    text-align: right;
}

/* Assistant message */
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    text-align: left;
}

</style>
""", unsafe_allow_html=True)
st.title("RAG-Based Document QA System")
st.caption("Upload a document and ask questions about its content.")

if "history" not in st.session_state:
    st.session_state.history = []
if "collection_ready" not in st.session_state:
    st.session_state.collection_ready = False

with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/4712/4712109.png",
        width=100
    )

    st.title("📂 Document Hub")

    uploaded = st.file_uploader(
        "Upload PDF/DOCX/TXT",
        type=["pdf","docx","txt"]
    )
    
    if st.button("Clear Chat"):
        st.session_state.history = []
        st.rerun()
    if uploaded and st.button("⚙️ Process"):
        with tempfile.NamedTemporaryFile(delete=False,
                                        suffix="."+uploaded.name.split(".")[-1]) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        try:
            try:
                chroma_client.delete_collection(COLLECTION)
            except:
                pass

            collection = chroma_client.get_or_create_collection(COLLECTION)

            # Ingest document and get number of chunks
            n_chunks = ingest_file(tmp_path, collection)

            st.success(f"Processed {n_chunks} chunks successfully!")
            st.session_state.collection_ready = True

        except Exception as e:
            st.error(f"Error processing document: {e}")

        finally:
            os.remove(tmp_path)

if not st.session_state.collection_ready:
    st.info("Upload and process a document first.")
else:
    for q, a in st.session_state.history:
        st.chat_message("user").write(q)
        st.chat_message("assistant").write(a)

    query = st.chat_input("Ask something about your document...")
    if query:
        with st.spinner("Thinking..."):
            collection = chroma_client.get_collection(COLLECTION)
            answer, sources = answer_question(
                query,
                collection,
                st.session_state.history
            )

        st.session_state.history.append((query, answer))
        st.chat_message("user").write(query)
        st.chat_message("assistant").write(answer)
        
        pages = sorted(
            set(
                str(int(s.get("page", 0)) + 1)
                for s in sources
                if s.get("page") is not None
            )
        )
        st.caption(f"📎 Retrieved from pages: {', '.join(pages)}")
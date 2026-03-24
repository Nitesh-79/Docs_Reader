import streamlit as st
import os
import fitz  # PyMuPDF
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------------------
# PDF Processing
# ---------------------------
def process_pdf(file_path):
    pdf = fitz.open(file_path)
    documents = []

    for i in range(len(pdf)):
        page = pdf.load_page(i)
        text = page.get_text()

        documents.append(
            Document(
                page_content=text,
                metadata={"page": i + 1, "source": file_path}
            )
        )

    pdf.close()
    return documents


# ---------------------------
# Process Uploaded Files
# ---------------------------
def process_documents(uploaded_files):
    documents = []
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    for file in uploaded_files:
        file_path = os.path.join(temp_dir, file.name)

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        if file.name.endswith(".pdf"):
            docs = process_pdf(file_path)

        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            docs = loader.load()

        elif file.name.endswith(".txt"):
            loader = TextLoader(file_path)
            docs = loader.load()

        else:
            st.warning(f"Unsupported file: {file.name}")
            continue

        documents.extend(docs)

        os.remove(file_path)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)
    return chunks


# ---------------------------
# Create Vector Store
# ---------------------------
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    return db


# ---------------------------
# MAIN APP
# ---------------------------
def main():
    st.set_page_config(page_title="Document Search App", page_icon="📄")
    st.title("📄 Document Search (No AI, Exact Answers Only)")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # Sidebar
    with st.sidebar:
        st.header("📁 Upload Documents")

        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, TXT",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )

        if st.button("Process Documents"):
            if uploaded_files:
                with st.spinner("Processing..."):
                    chunks = process_documents(uploaded_files)
                    st.session_state.vector_store = create_vector_store(chunks)
                    st.success(f"Processed {len(chunks)} chunks!")
            else:
                st.warning("Upload files first!")

    # Chat/Search
    if st.session_state.vector_store:
        st.subheader("🔍 Search Your Documents")

        query = st.text_input("Enter your question:")

        if query:
            retriever = st.session_state.vector_store.as_retriever(
                search_kwargs={"k": 3}
            )

            results = retriever.invoke(query)

            st.write("## 📌 Results:")

            for i, doc in enumerate(results):
                st.markdown(f"### 📄 Result {i+1} (Page {doc.metadata.get('page', 'N/A')})")
                st.write(doc.page_content)
                st.divider()

    else:
        st.info("👈 Upload and process documents first.")


# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    main()
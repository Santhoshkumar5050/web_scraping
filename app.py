import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
import google.generativeai as genai

# ----------------- Config -----------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")  # ✅ correct way

# ----------------- Functions -----------------
def scrape_text_from_url(url: str) -> str:
    """Scrape all paragraph text from a webpage."""
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        return "\n".join(paragraphs)
    except Exception as e:
        return f"Error scraping URL: {e}"

def create_vector_store(text: str):
    """Split text into chunks and create FAISS vector store."""
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

def ask_gemini(prompt: str) -> str:
    """Call Google Gemini model via GenerativeModel."""
    response = model.generate_content(prompt)
    return response.text

def identify_domain(text: str) -> str:
    """Identify domain of the content."""
    prompt = f"""
    Read the following text and return ONE word from this list:
    Finance, Education, Health, Technology, Other

    Text:
    {text[:1000]}
    """
    domain = ask_gemini(prompt).strip().capitalize()
    if domain not in ["Finance", "Education", "Health", "Technology"]:
        domain = "Other"
    return domain

# ----------------- Streamlit UI -----------------
st.title("🌐 Web RAG AI – NLP Tasks on Any Website")
st.info("Enter a URL, scrape text, and perform NLP tasks using Google Gemini.")

url = st.text_input("Enter Website URL:")

if url:
    with st.spinner("Scraping website..."):
        web_text = scrape_text_from_url(url)
        if len(web_text.strip()) == 0:
            st.error("No text found on the webpage!")
        else:
            st.success("✅ Text scraped successfully!")
            st.subheader("Preview of Scraped Text:")
            st.write(web_text[:1000] + "..." if len(web_text) > 1000 else web_text)

            # ----------------- NLP Tasks -----------------
            st.subheader("Select NLP Task")
            task = st.selectbox("Choose Task", [
                "Summarization",
                "Sentiment Analysis",
                "Domain Identification",
                "Translation",
                "Q&A"
            ])

            translation_language = None
            if task == "Translation":
                translation_language = st.selectbox(
                    "Select Target Language", ["Telugu", "Tamil", "Hindi", "Kannada"]
                )

            query = None
            if task == "Q&A":
                query = st.text_input("Enter your question:")

            if st.button("Run Task"):
                with st.spinner(f"Running {task}..."):
                    result = ""
                    if task == "Summarization":
                        prompt = f"Summarize the following text in 5 lines:\n\n{web_text}"
                        result = ask_gemini(prompt)
                    elif task == "Sentiment Analysis":
                        prompt = f"Analyze the sentiment (Positive/Negative/Neutral) of the following text:\n\n{web_text}"
                        result = ask_gemini(prompt)
                    elif task == "Domain Identification":
                        result = identify_domain(web_text)
                    elif task == "Translation":
                        prompt = f"Translate the following text into {translation_language}:\n\n{web_text}"
                        result = ask_gemini(prompt)
                    elif task == "Q&A":
                        if query and query.strip() != "":
                            vector_store = create_vector_store(web_text)
                            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
                            docs = retriever.get_relevant_documents(query)
                            combined_text = "\n".join([doc.page_content for doc in docs])
                            prompt = f"Answer the following question based on the text:\n\nText:\n{combined_text}\n\nQuestion: {query}\nAnswer:"
                            result = ask_gemini(prompt)
                        else:
                            st.warning("Please enter a question for Q&A.")
                            result = ""

                    if result:
                        st.subheader(f"✅ {task} Result")
                        st.write(result)

import os
import gradio as gr
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline

# Step 1: Load PDFs and URLs
def load_documents(pdf_directory, urls_file):
    documents = []
    # Load all PDFs from the specified directory
    for file_name in os.listdir(pdf_directory):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, file_name)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load_and_split())
    # Read URLs from the file and load them
    if os.path.exists(urls_file):
        with open(urls_file, "r") as file:
            urls = [line.strip() for line in file.readlines()]
        for url in urls:
            loader = WebBaseLoader(url)
            documents.extend(loader.load())
    return documents

# Step 2: Build FAISS Retriever
def create_retriever(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore.as_retriever()

# Step 3: Load LLaMA 2 Model
def load_llama2():
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Change this to the appropriate LLaMA model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, device=0)
    return HuggingFacePipeline(pipeline=pipe)

# Step 4: Create RAG Pipeline
def create_qa_pipeline(retriever, llm):
    return RetrievalQA(retriever=retriever, llm=llm)

# Step 5: User Query Handler
def handle_query(query, retriever, qa_pipeline):
    retriever_result = retriever.get_relevant_documents(query)
    answer = qa_pipeline.run(query)
    return {
        "Query": query,
        "Answer": answer,
        "Relevant Documents": [doc.page_content for doc in retriever_result]
    }

# Step 6: Gradio Interface
def main():
    # Path to the directory containing PDF files
    pdf_directory = r"C:\Users\roman\Desktop\hydro-grant\data\docs"
    # Path to the file containing URLs
    urls_file = r"C:\Users\roman\Desktop\hydro-grant\data\urls.txt"

    # Load documents and set up retriever
    documents = load_documents(pdf_directory, urls_file)
    retriever = create_retriever(documents)

    # Load LLaMA 2 model
    llm = load_llama2()

    # Create RAG pipeline
    qa_pipeline = create_qa_pipeline(retriever, llm)

    # Define Gradio inputs and outputs
    query_input = gr.Textbox(label="Enter your query:")
    output_display = gr.JSON(label="Response")

    # Gradio function
    def respond(query):
        return handle_query(query, retriever, qa_pipeline)

    # Launch Gradio interface
    interface = gr.Interface(fn=respond, inputs=query_input, outputs=output_display, title="RAG Chatbot with LLaMA 2")
    interface.launch()

if __name__ == "__main__":
    main()
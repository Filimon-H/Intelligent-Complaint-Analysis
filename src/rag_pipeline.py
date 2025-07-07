
# src/rag_pipeline.py

import os
import pickle
from typing import List
from dotenv import load_dotenv

from transformers import pipeline
from huggingface_hub import login
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# Step 1: Load the FAISS vector store (using the saved index.pkl)
#def load_vector_store(path: str = "../vector_store/index.pkl") -> FAISS:
    #with open(path, "rb") as f:
        #return pickle.load(f)
#def load_vector_store(path: str = "../vector_store/index.pkl") -> FAISS:
    #with open(path, "rb") as f:
        #vector_store, _ = pickle.load(f)  # Unpack and ignore 2nd item
       # return vector_store



from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def load_vector_store(path="vector_store") -> FAISS:
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(
        folder_path=path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )


#def load_vector_store(path="vector_store") -> FAISS:
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(folder_path=path, embeddings=embedding_model)


# Step 2: Retrieve top-k context chunks using similarity search
def retrieve_context(query: str, vector_store: FAISS, k: int = 5) -> List[str]:
    docs = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]

# Step 3: Define the RAG prompt template
def get_prompt_template() -> PromptTemplate:
    template = """You are a financial analyst assistant for CrediTrust.
Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer.
If the context doesn't contain the answer, say you don't have enough information.

Context:
{context}

Question: {question}
Answer:"""
    return PromptTemplate.from_template(template)

# Step 4: Generate an answer using context + prompt + user question
def generate_answer(question: str, context_chunks: List[str], llm) -> str:
    # Join the context chunks
    context = "\n\n".join(context_chunks)

    # Create the prompt
    prompt_template = get_prompt_template()

    # Create a LangChain LLMChain manually (no retriever)
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # Generate answer
    output = chain.run({"context": context, "question": question})
    return output.strip()

########################3


######################

# Step 5: Load the LLM for mistralai model
#def load_llm_Mistral(model_name="mistralai/Mistral-7B-Instruct-v0.1"):
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    if hf_token:
        login(hf_token)
    else:
        raise ValueError("HUGGINGFACE_TOKEN not found in .env file.")

    pipe = pipeline(
        "text-generation",
        model=model_name,
        token=hf_token,
        max_new_tokens=256
    )
    return HuggingFacePipeline(pipeline=pipe)


#def load_llm_Mistral(local_dir="../models/Mistral-7B-Instruct-v0.1"):
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(local_dir)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256
    )
    return HuggingFacePipeline(pipeline=pipe)


def load_llm_Mistral(local_dir="../models/Mistral-7B-Instruct-v0.1"):
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    model = AutoModelForCausalLM.from_pretrained(local_dir)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.95
    )
    return HuggingFacePipeline(pipeline=pipe)

##########################
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline

def load_llm_falcon(local_dir="../models/falcon-rw-1b"):
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    model = AutoModelForCausalLM.from_pretrained(local_dir)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.95
    )

    return HuggingFacePipeline(pipeline=pipe)


###########################


from transformers import pipeline
from langchain.llms import HuggingFacePipeline



from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.llms import HuggingFacePipeline

def load_llm(local_dir="../models/flan-t5-base"):
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(local_dir)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256
    )
    return HuggingFacePipeline(pipeline=pipe)



def load_llm_large(local_dir="../models/flan-t5-large"):
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(local_dir)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256
    )
    return HuggingFacePipeline(pipeline=pipe)






##########################
import requests
import os
from dotenv import load_dotenv

# Load Hugging Face token from .env for safety
load_dotenv()
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-rw-1b"
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

def load_llm_falcon():
    def llm_api_call(prompt: str):
        payload = {
            "inputs": prompt,
            "options": {"use_cache": True, "wait_for_model": True}
        }
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        if response.status_code == 200:
            return response.json()[0]["generated_text"]
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")
    
    return llm_api_call





# src/rag_pipeline.py (or rag_interface.py)

import os
import requests
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
API_URL_MISTRAL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

def load_llm_mistral():
    """
    Returns a function that queries the Mistral-7B-Instruct API.
    """
    def llm_api_call(prompt: str):
        # Wrap prompt in Mistral's required INST tokens
        wrapped = f"<s>[INST] {prompt} [/INST]"
        payload = {
            "inputs": wrapped,
            "parameters": {"return_full_text": False, "max_new_tokens": 256},
            "options": {"wait_for_model": True}
        }
        resp = requests.post(API_URL_MISTRAL, headers=HEADERS, json=payload)
        if resp.status_code == 200:
            return resp.json()[0]["generated_text"]
        else:
            raise Exception(f"Mistral API error {resp.status_code}: {resp.text}")
    return llm_api_call

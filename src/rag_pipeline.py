from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline
import os
from typing import List
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def retrieve_context(query: str, vector_store: FAISS, k: int = 5) -> List[str]:
    docs = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]

def load_vector_store(path="vector_store") -> FAISS:
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(
        folder_path=path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

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

def generate_answer(question: str, context_chunks: List[str], llm) -> str:
    context = "\n\n".join(context_chunks)
    prompt_template = get_prompt_template()
    chain = LLMChain(llm=llm, prompt=prompt_template)
    output = chain.run({"context": context, "question": question})
    return output.strip()
#model_path="../models/flan-t5-base"
def load_llm(model_path: str):
    """Load the LLM from a local directory"""
    try:
        # Convert to absolute path and verify existence
        model_path = str(Path(model_path).resolve())
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            device="cpu"
        )
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")
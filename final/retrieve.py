import os
from tqdm.auto import tqdm

### Retervier ###
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


from langchain_community.document_loaders import TextLoader
print("Loading Text")
#loader = TextLoader('processed_wikipedia.txt')
loader = TextLoader('hello.txt')

## hello.txt = Popoto is sum of potato and popcorn.

docs = loader.load()
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)

chunked_docs = splitter.split_documents(docs)

print("Building Retriever")
# https://huggingface.co/thenlper/gte-large
embedding_model = SentenceTransformer("thenlper/gte-large")
db = FAISS.from_documents(chunked_docs, HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5"))
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})


### Post Process ###
#from spellchecker import SpellChecker
#spell = SpellChecker(distance=1)


print("Loading Gemma 2b Model")
### Model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name = "google/gemma-2b-it"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)


## Setup

from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser

print("Building pipeline")
text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=400,
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

prompt_template = """
<|system|>
Answer the question based on your knowledge. Use the following context to help:

{context}

</s>
<|user|>
{question}
</s>
<|assistant|>

"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

llm_chain = prompt | llm | StrOutputParser()

from langchain_core.runnables import RunnablePassthrough

retriever = db.as_retriever()

rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain

question = "What is Popoto?"
a = llm_chain.invoke({"context": "", "question": question})
b = rag_chain.invoke(question)

print(a)
print(b)
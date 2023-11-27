from operator import itemgetter

from langchain.chat_models import ChatOpenAI
#from langchain.embeddings import OpenAIEmbeddings

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.vectorstores import FAISS

from utils import llama2Model
import fitz 

def create_retriever_from_pdf(pdf_paths):
    texts = []
    for pdf_path in pdf_paths:
        with fitz.open(pdf_path) as document:
            text = ""
            for page in document:
                text += page.get_text()
        texts = []
        max_length = 1000
        while text:
            if len(text) <= max_length:
                texts.append(text)
                break
            else:
                idx = text.rfind(' ', 0, max_length)

                if idx == -1:
                    idx = max_length

                texts.append(text[:idx])
                text = text[idx:].lstrip()

    vectorstore = FAISS.from_texts(
            texts,
            HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={"device": "cuda"}
                )
            )

    retriever = vectorstore.as_retriever()
    return retriever


if __name__=="__main__":
    pdf_paths = ["2307.09288.pdf"]
    retriever = create_retriever_from_pdf(pdf_paths)

    template = """<s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    <</SYS>>
    Answer the question based only on the following context:
    {context}
    Question: {question}
    [/INST]

    """
    prompt = ChatPromptTemplate.from_template(template)
    model = llama2Model()

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    #res = chain.invoke("How many parameters does llama2 have?")
    res = chain.invoke("What can llama 2 do?")
    print(res)







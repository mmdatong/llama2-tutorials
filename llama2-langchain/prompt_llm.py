from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from utils import llama2Model

if __name__=="__main__":
    prompt = """<s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    <</SYS>>

    What does {foo} look like? [/INST]
    """


    prompt = ChatPromptTemplate.from_template("What does {foo} look like?")
    model = llama2Model()

    chain = prompt | model
    res = chain.invoke({"foo": "bears"})
    print(res)
    import pdb; pdb.set_trace()


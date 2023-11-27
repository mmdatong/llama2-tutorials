## llama 2 系列教程（二） —— 用 langchain 和 rag 创建 chatpdf（个人阅读助手）

本文使用 llama2 语言大模型，结合 langchain 和 rag （retrieval augmented generation) 技术创建一个类似于 chatpdf 的个人阅读助手。本文的实验可以直接在[美美大同平台](https://mmdatong.com/ModelDetail#1)学习和运行使用。同时，文本的实验代码在 [mmdatong/llama2-tutorials](https://github.com/mmdatong/llama2-tutorials/blob/master/llama2-langchain-make-your-own-reading-assistant/rag.py)

### 一， langchain 和 rag 简介
langchain 是一个构建大模型应用的框架，它可以增强大模型的能力，帮助大模型获得更多背景信息，利用外部信息（文本，网络，代码函数等）来增强应用的能力。 langchain 作为语言大模型的一个重要补充，使得语言大模型可以地满足实际应用的需求。

rag （retrieval augmented generation） 是 langchain 中一个重要的方法，它可以利用外部信息作为知识库，增强语言大模型的能力。

本文利用 langchain 和 rag 技术，创建一个类似于 chatpdf 的个人阅读助手。

### 2. 实验介绍
#### 2.1 llama 2 模型转换
我们首先需要将原始的 meta-llama/Llama-2-7b-hf 转化为 langchain 部署可以用的模型，使用以下命令进行模型转换：

```
ct2-transformers-converter --model meta-llama/Llama-2-7b-chat-hf --output_dir /root/autodl-tmp/llama-2-7b-ct2 --force
```

得到转化后的模型之后，我们用 `CTranslate2` 创建llama 2 模型：
```python
def llama2Model():
    # infer to https://github.com/langchain-ai/langchain/blob/0efa59cbb8633357129083594ff6222fbc27726a/docs/docs/integrations/llms/ctranslate2.ipynb#L54
    model = CTranslate2(
            model_path="/root/autodl-tmp/llama-2-7b-ct2/",
            tokenizer_name="meta-llama/Llama-2-7b-hf",
            device="cuda",
            device_index=[0],
            compute_type="bfloat16",
            max_length=2048
            )
    return model
```

#### 2.2 提取 pdf 内容和构建知识库
本文以 [llama2](https://arxiv.org/pdf/2307.09288.pdf) 的原始论文 作为实验的pdf文本，使用 `fitz` 提取 pdf 中的内容，并且将pdf 中文本进行简单地分割，每1000个字符进行一次分割。用 `FAISS` 作为向量检索工具，`HuggingFaceEmbeddings` 作为embedding 的提取器。

#### 2.3 构建chain
我们首先定义llama2 模型的prompt，prompt 格式如下，其中 context 表示从pdf 文本中检索出来的相关知识，question 表示用户交互的提问。
```python
template = """<s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    <</SYS>>
    Answer the question based only on the following context:
    {context}
    Question: {question}
    [/INST]

    """
```

chain 的构建如下：
```python
chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
```

chain 构建之后，即可用 `chain.invoke(question)` 来使用langchain 增强后的llama2 模型了。


### 三，个人 chatpdf 的结果展示

以下是本文实验的结果展示，我们针对实验构建的chatpdf提了三个不同问题，以下是三个问题对应的回答。

| question | answer |
 -------- | ------- |
|what is llama2? | Hello! I'm here to help you with your question. Llama2 is a language model that has been developed by Facebook Research. It is designed to generate human-like text based on the input it receives, and it has been trained on a wide range of texts. However, it is important to note that Llama2 is a machine learning model, and it may not always produce accurate or appropriate responses. <br> As the developers of Llama2 have noted, the model may in some instances produce inaccurate or objectionable responses to user prompts. Therefore, before deploying any applications of Llama2, developers should perform safety testing and tuning tailored to their specific applications of the model. <br> It is also important to note that Llama2 is intended for commercial and research use in English, and it is not intended for use in any other languages. Additionally, it is important to comply with the terms o                        f the provided license and the Acceptable Use Policy when using Llama2. <br> Overall, Llama2 is a powerful tool for natural language generation, but it is important to use it responsibly and ethically. If you have any further questions about Llama2, please feel free to ask!|
|What are limitations of human evaluations? | Human evaluations have several limitations that should be considered when using them as a metric for assessing the helpfulness and safety of AI language models. Here are some of the key limitations:<br> 1. Subjectivity: Human evaluations are inherently subjective, as different annotators may have different interpretations of what is helpful or safe. This can lead to inconsistencies in the ratings given by different annotators, and can make it difficult to obtain a reliable measure of helpfulness or safety. <br> 2. Limited scope: Human evaluations are typically limited to a small subset of the possible responses that an AI language model could generate. This can make it difficult to evaluate the model's performance on a wide range of topics or in different contexts.<br>3. Time-consuming and expensive: Obtaining high-quality annotations from human evaluators can be time-consuming and expensive, particularly for large and complex datasets. This can make it difficult to scale up the evaluation process to include a large number of models or to evaluate models in a timely manner.<br>4. Annotator fatigue: Human evaluators may become fatigued or biased after evaluating a large number of responses, which can lead to inconsistent or inaccurate ratings.<br>5. Lack of context: Human evaluators may not have access to the full context of the conversation or the user's intent, which can make it difficult to evaluate the helpfulness or safety of a particular response.<br>6. Difficulty in defining helpfulness: It can be challenging to define what constitutes a helpful response, particularly in complex or nuanced situations. This can lead to inconsistencies in the ratings given by different annotators.<br>7. Difficulty in defining safety: It can be challenging to define what constitutes a safe response, particularly in situations where the AI language model is generating responses that are not explicitly harmful or offensive.8. Limited generalizability: Human evaluations may not generalize well to other contexts or populations, which can limit the applicability of the results.<br>Overall, while human evaluations can provide valuable insights into the helpfulness and safety of AI language models, they have several limitations that should be taken into account when interpreting the results.|
|How many parameters does llama2 have?|According to the document, Llama 2 has 77 million parameters.|

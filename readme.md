# llama 2 系列教程（1）—— 在自定义数据集上微调和模型预测

以 chatgpt 和 gpt4 为代表的语言大模型向人们展示了 AI 的强大能力，并且在实际生活中逐渐成为人类的重要助手，在很多场景下极大提升了工作效率，甚至能完全取代很多工作。相比于 chatgpt 和 gpt4，Meta 公司开源的 llama 2 允许用户可以更灵活地使用语言大模型，满足用户自定义的需求，并且可以商用。

本文主要向对语言大模型感兴趣，希望使用语言大模型的朋友提供一个基础的入门版教程，帮助朋友们快速地入门llama模型的使用。本文的主要部分包括：（1） llama2 语言大模型的简介；（2）在 samsum 数据集和自定义数据集（arithmetirc）上微调llama2语言大模型；（3）模型的推断以及微调模型和原始模型的结果比较。

### 一，llama2 简介


### 二，模型微调
为了帮助人们更好地使用 llama 模型，Meta 官方专门提供了一个项目 [llama-recipes](https://github.com/facebookresearch/llama-recipes/). 本文基于这个项目，介绍如何对llama2 模型进行微调，以及在自定义的数据集上进行微调，本文的实验代码开源在 [llama-tutorials](https://github.com/mmdatong/llama2-tutorials)，**您也可以在[美美大同平台](https://mmdatong.com/)一键启动机器，快速上手学习和使用模型。注意，如您在美美大同平台使用 llama2 模型，表示你已经阅读并同意 Meta 关于 llama2 的[协议](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)。**

下面本文介绍如何在 samsum 数据集和自定义的 arithmetic 数据集上微调 llama2 模型。

### 2.1 在samsum 数据集上微调 llama2

#### 2.1.1 samsum 数据集介绍

首先我们使用 Meta 官方提供的 samsum 数据集合进行微调。samsum 数据集可以直接从 huggingface 上下载，也可以直接加载本地已经下载好的samsum 数据。

```python
from datasets import load_dataset

# load from huggingface
samsum_dataset = load_dataset("samsum") 

# or load from local
samsum_dataset = load_dataset("/root/.cache/huggingface/datasets/samsum")

print(samsum_dataset)
```

```
DatasetDict({
    train: Dataset({
        features: ['id', 'dialogue', 'summary'],
        num_rows: 14732
    })
    test: Dataset({
        features: ['id', 'dialogue', 'summary'],
        num_rows: 819
    })
    validation: Dataset({
        features: ['id', 'dialogue', 'summary'],
        num_rows: 818
    })
})

```

如上所示，samsum 数据集的训练，验证和测试数据分别有14732，819和818条数据，每条数据分别有 dialogue 和 summary 两个字段。微调的目标就是让 llama2 模型可以基于 dialogue（对话）进行总结，输出总结的内容。

#### 2.1.2 模型微调
以下两个脚本都可以直接在 samsum 数据上进行微调，在网络不稳定情况下，推荐使用第一个脚本。本部分实验需要 24GB 显存，推荐使用 3090 或者 4090 显卡。实验在 samsum 数据集上训练1个epoch，预计训练时间是 1.5 小时。


```
# you can using following script:
cd /root/workspace  
bash finetune_samsum.sh


# or method 1：
python -m llama_recipes.finetuning \
       	--use_peft \
		--peft_method lora \
		--quantization \
		--use_fp16 \
		--model_name meta-llama/Llama-2-7b-hf \
		--output_dir output_samsum \
		--dataset custom_dataset \
		--custom_dataset.file "dataset.py:get_preprocessed_samsum" \
		--batch_size_training 1 \
		--num_epochs 1 \
		--use_fast_kernels

# or method 2：
python -m llama_recipes.finetuning \
       	--use_peft \
		--peft_method lora \
		--quantization \
		--use_fp16 \
		--model_name meta-llama/Llama-2-7b-hf \
		--output_dir output_samsum \
		--dataset samsum_dataset \
		--batch_size_training 1 \
		--num_epochs 1 \
		--use_fast_kernels
```


### 2.2 在自定义数据集（arithmetic 数据集）上微调 llama2
#### 2.2.1 arithmetic 数据集介绍
为了展示如何在自定义的数据集上进行 llama2 的模型微调，本文借鉴 [GOAT论文](https://arxiv.org/pdf/2305.14201.pdf) 中提到的方法。arithmetic 生成了一批数学计算的文本，让 llama2 模型学会 100 以内的加法算术。

首先我们使用以下的脚本生成这样一批数据，其中训练集，测试集，验证集的数量分别是 9500，500，500.

```
cd /root/workspace/arithmetic_data
bash create_arithmetic_data.sh
```

接下来，我们来看看 arithmetic 数据集的格式。
``` python
# run ipython in /root/workspace
from datasets import load_dataset
arithmetic_dataset = load_dataset("csv", data_files={"train": "arithmetic_data/arithmetic_train.csv"})["train"]

print(arithmetic_dataset)

```


```
Dataset({
    features: ['instruction', 'input', 'output', 'answer'],
    num_rows: 9500
})
```

生成的自定义 arithmetic 训练数据有9500条，每一条有'instruction', 'input', 'output', 'answer'字段，实际训练用到的字段只有 'instruction', 'output'。


#### 2.2.1 arithmetic 数据集处理
以上介绍了自定义 arithmetic 数据的生成方式和数据格式。下面介绍如何用自定义的数据进行模型的微调。

llama-recipes 提供了一个接口，允许用户可以自己设计训练数据的输入格式，在 [dataset.py](https://github.com/mmdatong/llama2-tutorials/blob/master/dataset.py#L42) 中 get_preprocessed_arithmetic 函数展示了如何读取自定义数据，并且转化为 llama2 模型的输入。

在准备好数据处理函数之后，用户可以通过 `--dataset` 和 `--custom_dataset.file` 两个参数，指定模型训练用到的数据集。在 arithmetic 数据集上微调的脚本如下所示：
```shell
bash finetune_arithmetic.sh

# or using below script

python -m llama_recipes.finetuning \
       	--use_peft \
		--peft_method lora \
		--quantization \
		--use_fp16 \
		--model_name meta-llama/Llama-2-7b-hf \
		--dataset custom_dataset \
		--custom_dataset.file "dataset.py:get_preprocessed_arithmetic" \
		--output_dir output_arithmetic \
		--batch_size_training 1 \
		--num_epochs 1 \
		--use_fast_kernels
```


### 三，模型预测和结果比较

经过以上的微调，得到 finetune 的模型，两个任务模型训练的结果分别保存在 `output_samsum` 和`output_arithmetic`


#### 3.1. samsum 任务的比较
[infer_samsum.sh](https://github.com/mmdatong/llama2-tutorials/blob/master/infer_samsum.sh) 脚本中展示如何用原始llama2 模型以及微调后的模型进行预测。以下是预测结果的比较。


| dialogue | summary of llama2 | sumary of finetuned llama2 |
| -------- | ------- | ------- |
| Summarize this dialog: <br>A: Hi Tom, are you busy tomorrow’s afternoon? <br>B: I’m pretty sure I am. What’s up?<br>A: Can you go with me to the animal shelter?.<br>B: What do you want to do?<br>A: I want to get a puppy for my son.<br>B: That will make him so happy.<br>A: Yeah, we’ve discussed it many times. I think he’s ready now.<br>B: That’s good. Raising a dog is a tough issue. Like having a baby ;-) <br>A: I'll get him one of those little dogs.<br>B: One that won't grow up too big;-)<br>A: And eat too much;-))<br>B: Do you know which one he would like?<br>A: Oh, yes, I took him there last Monday. He showed me one that he really liked.<br>B: I bet you had to drag him away.<br>A: He wanted to take it home right away ;-).<br>B: I wonder what he'll name it.<br>A: He said he’d name it after his dead hamster – Lemmy  - he's  a great Motorhead fan :-)))<br>---<br>Summary:<br> |  There are a few other ways to make this dialog more interesting. For example:<br>A: Let me know if you can’t go;-))<br>B: It’s ok. I'm not sure of this yet.<br>A: Ok, so I’ll take your place. Who’s going?<br>B: No, don't forget: you already have dinner with your family ;-))<br>A: Oh, sure, I do<br> | A wants to buy a puppy for his son from an animal shelter. His son already saw a puppy he liked there which he did not buy. |


以上是原始llama2模型和微调后的模型在同一个输入下的结果，可见原始模型只能简单地重复，而微调后的模型能比较好地总结对话的内容。


#### 3.2 arithmetic 任务的比较

[infer_arithmetic.sh](https://github.com/mmdatong/llama2-tutorials/blob/master/infer_arithmetic.sh) 脚本中展示如何用原始llama2 模型以及微调后的模型进行预测。以下是预测结果的比较。


| dialogue | summary of llama2 | sumary of finetuned llama2 |
| -------- | ------- | ------- |
|Calculate the following expression: <br> What is the value obtained from 19-59?<br> ---<br> Answer: | Answer: There are 2 types of people in this world.<br>\strong{First type}: A person who asks question like which expression did you ask?<br>\strong{Second type}: A person who says - Oops! I missed the number!<br>Answer: Well, 19-59 is a subtraction that involves numbers greater than 10. To evaluate, just write out the digits one by one in your head: "one nine, two nine,<br> | Answer: 19 - 59 = -40 |

以上的对比显示，相比于原始模型，微调后的模型可以比较好地进行数学计算了。


基于以上的教程，希望您掌握了llama2的微调方法，可以在自己的数据和任务上使用 llama2 模型。如有问题，欢迎提问和交流。
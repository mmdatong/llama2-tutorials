# llama 2 系列教程（1）—— 在自定义数据集上微调和模型预测

以 chatgpt 和 gpt4 为代表的语言大模型向人们展示了 AI 的强大能力，并且在实际生活中逐渐成为人类的重要助手，在很多场景下极大提升了工作效率，甚至能完全取代很多工作。相比于 chatgpt 和 gpt4，Meta 公司开源的 llama 2 允许用户可以更灵活地使用语言大模型，满足用户自定义的需求，并且可以商用。

本文主要向对语言大模型感兴趣，希望使用语言大模型的朋友提供一个基础的入门版教程，帮助朋友们快速地入门llama模型的使用。本文的主要部分包括：（1） llama2 语言大模型的简介；（2）在 samsum 数据集和自定义数据集（arithmetirc）上微调llama2语言大模型；（3）模型的推断以及微调模型和原始模型的结果比较。

### 一，llama2 简介


### 二，模型微调
为了帮助人们更好地使用 llama 模型，Meta 官方专门提供了一个项目 [llama-recipes](https://github.com/facebookresearch/llama-recipes/). 本文基于这个项目，介绍如何对llama2 模型进行微调，以及在自定义的数据集上进行微调，本文的实验代码开源在 [llama-tutorials](https://github.com/mmdatong/llama2-tutorials)，**用户也可以在[美美大同](https://mmdatong.com/)一键启动机器，快速上手学习和使用模型**。注意，如您在美美大同网站使用 llama2 模型，表示你已经阅读并同意 Meta 关于 llama2 的[协议](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)。

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

``` json
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


```json
Dataset({
    features: ['instruction', 'input', 'output', 'answer'],
    num_rows: 9500
})
```

生成的自定义 arithmetic 训练数据有9500条，每一条有'instruction', 'input', 'output', 'answer'字段，实际训练用到的字段只有 'instruction', 'output'。


#### 2.2.1 arithmetirc 数据集处理
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
		--output_dir output_samsum \
		--batch_size_training 1 \
		--num_epochs 1 \
		--use_fast_kernels


```





### 三，模型预测和结果比较



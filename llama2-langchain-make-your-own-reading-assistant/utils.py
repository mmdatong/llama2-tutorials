from langchain.llms import CTranslate2

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

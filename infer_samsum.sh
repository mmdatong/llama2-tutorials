# infer samsum task using our finetuned llama2
cat prompt_samsum.txt | python inference.py \
	--model_name meta-llama/Llama-2-7b-hf \
	--peft_model $PWD/output_samsum \
	--use_auditnlg 


# infer samsum task using original llama2
cat prompt_samsum.txt | python inference.py \
	--model_name meta-llama/Llama-2-7b-hf \
	--use_auditnlg 

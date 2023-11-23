



# infer arithmetirc task using our finetuned llama2
cat prompt_arithmetic.txt | python inference.py \
	--model_name meta-llama/Llama-2-7b-hf \
	--peft_model $PWD/output_arithmetric \
	--use_auditnlg 


# infer samsum task using original llama2
# cat prompt_arithmetic.txt | python inference.py \
# 	--model_name meta-llama/Llama-2-7b-hf \
# 	--use_auditnlg 
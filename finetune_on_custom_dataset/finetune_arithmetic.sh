# set http proxy in container
source /etc/network_turbo

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

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


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift sft \
  --model Qwen/Qwen3-8B \
  --train_type lora \
  --dataset $DATASET_PATH \
  --torch_dtype bfloat16 \
  --num_train_epochs 6 \
  --per_device_train_batch_size 1  \
  --learning_rate 1e-5 \
  --lora_rank 64 \
  --lora_alpha 32 \
  --deepspeed zero3 \
  --target_modules all-linear \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing False \
  --max_length 8192 \
  --output_dir output/NURBGen_H100_T1_DeepSpeed_8B \
  --save_steps 10000 \
  --logging_steps 10000 \
  --eval_steps 10000 \
  --save_total_limit 2 \
  --warmup_ratio 0.03 \
  --dataloader_num_workers 4 \
  --model_author swift \
  --model_name nurbs-stage0
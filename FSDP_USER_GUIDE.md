# User Guide: Running HuggingFace Llama 2 Training on v4 and v5e with PyTorch/XLA FSDP API


This user guide provides a concise overview of the essential steps required to run HuggingFace (HF) Llama 2 training on both v4 and v5e, with PyTorch/XLA FSDP API. For better training performance, please check out our [SPMD training user guide](https://github.com/pytorch-tpu/transformers/blob/llama2-google-next-training/SPMD_USER_GUIDE.md?plain=1)

## Environment Setup

The environment setup to train with FSDP is the same as training with SPMD, please follow [this guide](https://github.com/pytorch-tpu/transformers/blob/llama2-google-next-training/SPMD_USER_GUIDE.md#environment-setup) to setup the cloud TPU environment. The only difference is to checkout to `llama2-fsdp-training` branch in the HF Transformer installation command.

## Steps to Run HF Llama 2 with PyTorch/XLA FSDP API

```
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} \
--zone ${ZONE} \
--project ${PROJECT} \
--worker=all \
--command='
# Setup envs
export PJRT_DEVICE=TPU;

export PROFILE_EPOCH=0
export PROFILE_STEP=3
export PROFILE_DURATION_MS=20000
export PROFILE_LOGDIR=/tmp/home/

cd transformers;

python -u examples/pytorch/xla_spawn.py \
  --multiple_device True \
  examples/pytorch/language-modeling/run_clm.py \
  --tokenizer_name gpt2 \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --per_device_train_batch_size 24 \
  --num_train_epochs 2 \
  --do_train \
  --output_dir /tmp/output \
  --overwrite_output_dir \
  --config_name ~/config.json \
  --save_strategy no \
  --logging_strategy no \
  --optim adafactor \
  --block_size 1024
'
```

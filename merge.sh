cd /root/verl && python scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --hf_model_path /root/dataDisk/Qwen3-8B \
    --local_dir /root/dataDisk/checkpoints/global_step_48/actor \
    --target_dir /root/dataDisk/DeepWereWolf-Qwen3-8B-Grpo-Agentic5
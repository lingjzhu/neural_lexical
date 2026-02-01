python src/train_splade_causal.py --train_data training_data_v1_final_multi_neg.jsonl --eval_data interpretation/data/interpretation_w1/randomly_selected_test_amazon.jsonl --base_model modernbert-base --backend modernbert_fused_mean --output_dir hrs_checkpoints/modernbert-fused-mean-reg5e-3-scale1 --reg_weight 5e-3 --use_wandb --run_name mobdernbert-fused-mean-reg5e-3-scale1 --lr 1e-4 --batch_size 64 --scale_start 1 --scale_end 1 --scale_total_steps 3000 --reg_total_steps 2000 --reg_start 5e-4 


python src/train_splade_causal.py --train_data training_data_v1_final_multi_neg.jsonl --eval_data interpretation/data/interpretation_w1/randomly_selected_test_amazon.jsonl --base_model modernbert-base --backend modernbert_fused_max --output_dir hrs_checkpoints/modernbert-fused-max-reg5e-3-scale1 --reg_weight 5e-3 --use_wandb --run_name mobdernbert-fused-max-reg5e-3-scale1 --lr 1e-4 --batch_size 64 --scale_start 1 --scale_end 1 --scale_total_steps 3000 --reg_total_steps 2000 --reg_start 5e-4 

python src/train_splade_causal.py --train_data training_data_v1_final_multi_neg.jsonl --eval_data interpretation/data/interpretation_w1/randomly_selected_test_amazon.jsonl --base_model modernbert-base --backend modernbert_fused_mean --output_dir hrs_checkpoints/modernbert-fused-mean-reg5e-3-scale20 --reg_weight 5e-3 --use_wandb --run_name mobdernbert-fused-mean-reg5e-3-scale20 --lr 1e-4 --batch_size 64 --scale_start 20 --scale_end 20 --scale_total_steps 3000 --reg_total_steps 2000 --reg_start 5e-4 


python src/train_splade_causal.py --train_data training_data_v1_final_multi_neg.jsonl --eval_data interpretation/data/interpretation_w1/randomly_selected_test_amazon.jsonl --base_model answerdotai/ModernBERT-large --backend modernbert_fused_mean --output_dir hrs_checkpoints/modernbert-large-fused-mean-reg5e-4-scale1 --reg_weight 5e-4 --use_wandb --run_name mobdernbert-large-fused-mean-reg5e-3-scale1 --lr 5e-5 --batch_size 64 --scale_start 1 --scale_end 1 --scale_total_steps 5000 --reg_total_steps 2000 --reg_start 5e-5 


#python src/train_splade_causal.py --train_data training_data_v1_final_multi_neg.jsonl --eval_data interpretation/data/interpretation_w1/randomly_selected_test_amazon.jsonl --base_model Qwen/Qwen3-0.6B --backend qwen3_fused_mean --output_dir hrs_checkpoints/modernbert-fused-mean-reg5e-3-scale20 --reg_weight 5e-3 --use_wandb --run_name mobdernbert-fused-mean-reg5e-3-scale20 --lr 1e-4 --batch_size 64 --scale_start 20 --scale_end 20 --scale_total_steps 3000 --reg_total_steps 2000 --reg_start 5e-4 

python src/evaluate_hrs.py --parent_dir hrs_release_11-22-24  --model_name hrs_checkpoints/modernbert-large-fused-mean-reg5e-4-scale1/checkpoint-6326 --output_file hrs_eval/modernbert-large-fused-mean-reg5e-4-scale1-1024.jsonl --k 1024 --preds_dir hrs_preds/modernbert-large-fused-mean-reg5e-4-scale1-1024

python src/evaluate_hrs.py --parent_dir hrs_release_11-22-24  --model_name hrs_checkpoints/modernbert-large-fused-mean-reg5e-4-scale1/checkpoint-6326 --output_file hrs_eval/modernbert-large-fused-mean-reg5e-4-scale1-512.jsonl --k 512

python src/evaluate_hrs.py --parent_dir hrs_release_11-22-24  --model_name hrs_checkpoints/modernbert-large-fused-mean-reg5e-4-scale1/checkpoint-6326 --output_file hrs_eval/modernbert-large-fused-mean-reg5e-4-scale1-128.jsonl --k 128

python src/evaluate_hrs.py --parent_dir hrs_release_11-22-24  --model_name hrs_checkpoints/modernbert-large-fused-mean-reg5e-4-scale1/checkpoint-6326 --output_file hrs_eval/modernbert-large-fused-mean-reg5e-4-scale1-64.jsonl --k 64 --pred_dir hrs_preds/modernbert-large-fused-mean-reg5e-4-scale1

python src/evaluate_hrs.py --parent_dir hrs_release_11-22-24  --model_name hrs_checkpoints/modernbert-fused-mean-reg5e-3-scale1/checkpoint-6326 --output_file hrs_eval/modernbert-fused-mean-reg5e-3-scale1-128 --k 128

python src/evaluate_hrs.py --parent_dir hrs_release_11-22-24  --model_name hrs_checkpoints/modernbert-fused-mean-reg5e-3-scale1/checkpoint-6326 --output_file hrs_eval/modernbert-fused-mean-reg5e-3-scale1-128 --k 256

python src/evaluate_hrs.py --parent_dir hrs_release_11-22-24  --model_name hrs_checkpoints/modernbert-fused-mean-reg5e-3-scale1/checkpoint-6326 --output_file hrs_eval/modernbert-fused-mean-reg5e-3-scale1-128 --k 512


python train_dense_opt.py --train_data training_data_v1_final_multi_neg.jsonl --eval_data interpretation/data/interpretation_w1/randomly_selected_test_amazon.jsonl --base_model answerdotai/ModernBERT-large --backend modernbert --output_dir hrs_checkpoints/modernbert-large-dense --reg_weight 0 --use_wandb --run_name mobdernbert-large-dense --lr 5e-4 --batch_size 64 --k 1024




python src/train_splade_causal.py --train_data training_data_v1_hard_negatives.jsonl --eval_data interpretation/data/interpretation_w1/randomly_selected_test_amazon.jsonl --base_model modernbert-base --backend modernbert_fused_mean --output_dir hrs_checkpoints/modernbert-fused-mean-reg5e-3-scale1-hn --reg_weight 5e-3 --use_wandb --run_name mobdernbert-fused-mean-reg5e-3-scale1-hn --lr 1e-4 --batch_size  32 --scale_start 1 --scale_end 1 --scale_total_steps 3000 --reg_total_steps 2000 --reg_start 5e-4 

python src/train_splade_causal.py --train_data training_data_v1_hard_negatives.jsonl --eval_data interpretation/data/interpretation_w1/randomly_selected_test_amazon.jsonl --base_model answerdotai/ModernBERT-large --backend modernbert_fused_mean --output_dir hrs_checkpoints/modernbert-large-fused-mean-reg5e-4-scale1-hn --reg_weight 5e-4 --use_wandb --run_name mobdernbert-large-fused-mean-reg5e-3-scale1-hn --lr 5e-5 --batch_size 16 --scale_start 1 --scale_end 1 --scale_total_steps 5000 --reg_total_steps 2000 --reg_start 5e-5 




python  train_sparse_colbert.py --train_data training_data_v1_final_dedup.jsonl --eval_data amazon_triplets.jsonl --base_model modernbert-base --backend modernbert_sparse_colbert --output_dir hrs_checkpoints/modernbert-sparse_colbert-fp8 --use_wandb --run_name modernbert-sparse_colbert-fp8 --lr 5e-5 --batch_size 32 --k 1024

python  train_sparse_colbert.py --train_data training_data_v1_final_dedup.jsonl --eval_data amazon_triplets.jsonl --base_model hrs_checkpoints/modernbert-fused-mean-reg5e-3-scale1/checkpoint-6326 --backend modernbert_sparse_colbert --output_dir hrs_checkpoints/modernbert-sparse_colbert-k32-warmstart --use_wandb --run_name modernbert-sparse_colbert-k32-warmstart --lr 1e-5 --batch_size 64 --k 32
python  train_sparse_colbert.py --train_data training_data_v1_final_dedup.jsonl --eval_data amazon_triplets.jsonl --base_model hrs_checkpoints/modernbert-fused-mean-reg5e-3-scale1/checkpoint-6326 --backend modernbert_sparse_colbert --output_dir hrs_checkpoints/modernbert-sparse_colbert-k64-warmstart --use_wandb --run_name modernbert-sparse_colbert-k64-warmstart --lr 1e-5 --batch_size 64 --k 64
python measure_colbert.py --train_data training_data_v1_final_dedup.jsonl --eval_data amazon_triplets.jsonl



python src/evaluate_hrs.py --parent_dir hrs_release_11-22-24  --model_name hrs_checkpoints/modernbert-large-fused-mean-reg5e-4-scale1/checkpoint-6326 --output_file hrs_eval/modernbert-large-fused-mean-reg5e-4-scale1-1024.jsonl --k 1024 --preds_dir hrs_preds/modernbert-large-fused-mean-reg5e-4-scale1-1024

python src/evaluate_hrs.py --parent_dir hrs_release_11-22-24  --model_name hrs_checkpoints/modernbert-fused-mean-reg5e-3-scale1/checkpoint-6326 --output_file hrs_eval/modernbert-fused-mean-reg5e-3-scale1-1024.jsonl --k 1024 --preds_dir hrs_preds/modernbert-fused-mean-reg5e-3-scale1-1024

python src/evaluate_hrs.py --parent_dir hrs_release_11-22-24  --model_name hrs_checkpoints/qwen3-0.6B-mean-reg5e-3-scale1/checkpoint-6326 --output_file hrs_eval/qwen3-0.6B-mean-reg5e-3-scale1-1024.jsonl --k 1024 --preds_dir hrs_preds/qwen3-0.6B-mean-reg5e-3-scale1-1024 --backend qwen3_fused_mean

python src/rerank_sparse.py hrs_preds/modernbert-fused-mean-reg5e-3-scale1-1024 --model-name hrs_checkpoints/modernbert-fused-mean-reg5e-3-scale1/checkpoint-6326 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name modernbert-fused-mean-reg5e-3-scale1-zeroshot-k32 --k 32

python src/rerank_sparse.py hrs_preds/modernbert-fused-mean-reg5e-3-scale1-1024 --model-name hrs_checkpoints/modernbert-fused-mean-reg5e-3-scale1/checkpoint-6326 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name modernbert-fused-mean-reg5e-3-scale1-zeroshot-k64 --k 64

python src/rerank_sparse.py hrs_preds/modernbert-fused-mean-reg5e-3-scale1-1024 --model-name hrs_checkpoints/modernbert-fused-mean-reg5e-3-scale1/checkpoint-6326 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name modernbert-fused-mean-reg5e-3-scale1-zeroshot-k128 --k 128

python src/rerank_sparse.py hrs_preds/modernbert-fused-mean-reg5e-3-scale1-1024 --model-name hrs_checkpoints/modernbert-fused-mean-reg5e-3-scale1/checkpoint-6326 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name modernbert-fused-mean-reg5e-3-scale1-zeroshot-k256 --k 256


python src/rerank_sparse.py hrs_preds/modernbert-fused-mean-reg5e-3-scale1-1024 --model-name hrs_checkpoints/modernbert-fused-mean-reg5e-3-scale1/checkpoint-6326 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name modernbert-fused-mean-reg5e-3-scale1-zeroshot-k512 --k 512

python src/rerank_sparse.py hrs_preds/modernbert-fused-mean-reg5e-3-scale1-1024 --model-name hrs_checkpoints/modernbert-fused-mean-reg5e-3-scale1/checkpoint-6326 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name modernbert-fused-mean-reg5e-3-scale1-zeroshot-k1024 --k 1024


python src/rerank_sparse.py hrs_preds/modernbert-large-fused-mean-reg5e-4-scale1-1024 --model-name hrs_checkpoints/modernbert-large-fused-mean-reg5e-4-scale1/checkpoint-6326 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name modernbert-large-fused-mean-reg5e-4-scale1-zeroshot-k32 --k 32

python src/rerank_sparse.py hrs_preds/modernbert-large-fused-mean-reg5e-4-scale1-1024 --model-name hrs_checkpoints/modernbert-large-fused-mean-reg5e-4-scale1/checkpoint-6326 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name modernbert-large-fused-mean-reg5e-4-scale1-zeroshot-k64 --k 64

python src/rerank_sparse.py hrs_preds/modernbert-large-fused-mean-reg5e-4-scale1-1024 --model-name hrs_checkpoints/modernbert-large-fused-mean-reg5e-4-scale1/checkpoint-6326 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name mmodernbert-large-fused-mean-reg5e-4-scale1-zeroshot-k128 --k 128

python src/rerank_sparse.py hrs_preds/modernbert-large-fused-mean-reg5e-4-scale1-1024 --model-name hrs_checkpoints/modernbert-large-fused-mean-reg5e-4-scale1/checkpoint-6326 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name modernbert-large-fused-mean-reg5e-4-scale1-zeroshot-k256 --k 256 --act relu



python src/rerank_sparse.py hrs_preds/qwen3-0.6B-mean-reg5e-3-scale1-1024 --model-name hrs_checkpoints/qwen3-0.6B-mean-reg5e-3-scale1/checkpoint-6326 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name qwen3-0.6B-mean-reg5e-3-scale1-zeroshot-k32 --k 32 --backend qwen3_sparse_colbert

python src/rerank_sparse.py hrs_preds/qwen3-0.6B-mean-reg5e-3-scale1-1024 --model-name hrs_checkpoints/qwen3-0.6B-mean-reg5e-3-scale1/checkpoint-6326 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name qwen3-0.6B-mean-reg5e-3-scale1-zeroshot-k64 --k 64 --backend qwen3_sparse_colbert

python src/rerank_sparse.py hrs_preds/qwen3-0.6B-mean-reg5e-3-scale1-1024 --model-name hrs_checkpoints/qwen3-0.6B-mean-reg5e-3-scale1/checkpoint-6326 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name qwen3-0.6B-mean-reg5e-3-scale1-zeroshot-k128 --k 128 --backend qwen3_sparse_colbert

python src/rerank_sparse.py hrs_preds/qwen3-0.6B-mean-reg5e-3-scale1-1024 --model-name hrs_checkpoints/qwen3-0.6B-mean-reg5e-3-scale1/checkpoint-6326 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name qwen3-0.6B-mean-reg5e-3-scale1-zeroshot-k256 --k 256 --backend qwen3_sparse_colbert

python src/rerank_sparse.py hrs_preds/qwen3-0.6B-mean-reg5e-3-scale1-1024 --model-name hrs_checkpoints/qwen3-0.6B-mean-reg5e-3-scale1/checkpoint-6326 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name qwen3-0.6B-mean-reg5e-3-scale1-zeroshot-k512 --k 512 --backend qwen3_sparse_colbert

python src/rerank_sparse.py hrs_preds/qwen3-0.6B-mean-reg5e-3-scale1-1024 --model-name hrs_checkpoints/qwen3-0.6B-mean-reg5e-3-scale1/checkpoint-6326 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name qwen3-0.6B-mean-reg5e-3-scale1-zeroshot-k1024 --k 1024 --backend qwen3_sparse_colbert

python src/rerank_sparse.py hrs_preds/modernbert-fused-mean-reg5e-3-scale1-1024 --model-name hrs_checkpoints/modernbert-sparse_colbert-fp8/checkpoint-12652 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name modernbert-sparse_colbert-fp8-k1024 --k 1024

python src/rerank_sparse.py hrs_preds/modernbert-fused-mean-reg5e-3-scale1-1024 --model-name hrs_checkpoints/modernbert-sparse_colbert-fp8/checkpoint-12652 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name modernbert-sparse_colbert-fp8-k128 --k 128

python src/rerank_sparse.py hrs_preds/modernbert-fused-mean-reg5e-3-scale1-1024 --model-name hrs_checkpoints/modernbert-sparse_colbert-fp8/checkpoint-12652 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name modernbert-sparse_colbert-fp8-k64 --k 64

python src/rerank_sparse.py hrs_preds/modernbert-fused-mean-reg5e-3-scale1-1024 --model-name hrs_checkpoints/modernbert-sparse_colbert-fp8/checkpoint-12652 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name modernbert-sparse_colbert-fp8-k32 --k 32

python src/rerank_sparse.py hrs_preds/modernbert-fused-mean-reg5e-3-scale1-1024 --model-name hrs_checkpoints/modernbert-sparse_colbert-k64-warmstart/checkpoint-6326 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name modernbert-sparse_colbert-k64-warmstart --k 64

python src/rerank_sparse.py hrs_preds/modernbert-fused-mean-reg5e-3-scale1-1024 --model-name hrs_checkpoints/modernbert-sparse_colbert-k32-warmstart/checkpoint-6326 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name modernbert-sparse_colbert-k32-warmstart --k 32


python src/rerank_sparse.py hrs_preds/modernbert-large-fused-mean-reg5e-4-scale1-1024 --model-name hrs_checkpoints/modernbert-large-fused-mean-reg5e-3-scale1/checkpoint-6326 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name modernbert-large-fused-mean-reg5e-3-scale1-zeroshot-k32 --k 32

python src/rerank_sparse.py hrs_preds/modernbert-large-fused-mean-reg5e-4-scale1-1024 --model-name hrs_checkpoints/modernbert-large-fused-mean-reg5e-3-scale1/checkpoint-6326 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name modernbert-large-fused-mean-reg5e-3-scale1-zeroshot-k64 --k 64

python src/rerank_sparse.py hrs_preds/modernbert-large-fused-mean-reg5e-4-scale1-1024 --model-name hrs_checkpoints/modernbert-large-fused-mean-reg5e-3-scale1/checkpoint-6326 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name modernbert-large-fused-mean-reg5e-3-scale1-zeroshot-k128 --k 128

python src/rerank_sparse.py hrs_preds/modernbert-large-fused-mean-reg5e-4-scale1-1024 --model-name hrs_checkpoints/modernbert-large-fused-mean-reg5e-3-scale1/checkpoint-6326 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name modernbert-large-fused-mean-reg5e-3-scale1-zeroshot-k256 --k 256 

python src/rerank_sparse.py hrs_preds/modernbert-large-fused-mean-reg5e-4-scale1-1024 --model-name hrs_checkpoints/modernbert-large-fused-mean-reg5e-3-scale1/checkpoint-6326 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name modernbert-large-fused-mean-reg5e-3-scale1-zeroshot-k512 --k 512

python src/rerank_sparse.py hrs_preds/modernbert-large-fused-mean-reg5e-4-scale1-1024 --model-name hrs_checkpoints/modernbert-large-fused-mean-reg5e-3-scale1/checkpoint-6326 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name modernbert-large-fused-mean-reg5e-3-scale1-zeroshot-k1024 --k 1024 


python src/rerank_sparse.py hrs_preds/modernbert-large-fused-mean-reg5e-4-scale1-1024 --model-name hrs_checkpoints/modernbert-large-sparse_colbert-fp8-k1024-t1/checkpoint-12652 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name modernbert-large-sparse_colbert-fp8-k1024-t1-k1024 --k 1024

python src/rerank_sparse.py hrs_preds/modernbert-large-fused-mean-reg5e-4-scale1-1024 --model-name hrs_checkpoints/modernbert-large-sparse_colbert-fp8-k1024-t1/checkpoint-12652 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name modernbert-large-sparse_colbert-fp8-k1024-t1-k512 --k 512

python src/rerank_sparse.py hrs_preds/modernbert-large-fused-mean-reg5e-4-scale1-1024 --model-name hrs_checkpoints/modernbert-large-sparse_colbert-fp8-k1024-t1/checkpoint-12652 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name modernbert-large-sparse_colbert-fp8-k1024-t1-k256 --k 256

python src/rerank_sparse.py hrs_preds/modernbert-large-fused-mean-reg5e-4-scale1-1024 --model-name hrs_checkpoints/modernbert-large-sparse_colbert-fp8-k1024-t1/checkpoint-12652 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name modernbert-large-sparse_colbert-fp8-k1024-t1-k128 --k 128

python evaluate_hrs_spladev3.py --parent_dir hrs_release_11-22-24  --model_name hrs_checkpoints --output_file hrs_eval/spladev3-1024.jsonl --k 1024 
python evaluate_hrs_bm25.py --parent_dir hrs_release_11-22-24 --output_file hrs_eval/bm25.jsonl 
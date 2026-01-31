

#python  train_sparse_colbert.py --train_data bluesky/train_2epoch_5posts.jsonl --eval_data bluesky/20251114_dev_pairs_5post.jsonl --base_model modernbert-base --backend modernbert_sparse_colbert --output_dir checkpoints/modernbert-sparse_colbert-relu-32 --use_wandb --run_name modernbert-sparse_colbert-relu-32 --lr 5e-4 --batch_size 32 

#python  train_sparse_colbert.py --train_data bluesky/train_2epoch_5posts.jsonl --eval_data bluesky/20251114_dev_pairs_5post.jsonl --base_model modernbert-base --backend modernbert_sparse_colbert --output_dir checkpoints/modernbert-sparse_colbert-relu --use_wandb --run_name modernbert-sparse_colbert-relu --lr 5e-4 --batch_size 64 

#python train_splade_causal.py --train_data bluesky/train_2epoch_5posts.jsonl --eval_data bluesky/20251114_dev_pairs_5post.jsonl --base_model Qwen/Qwen3-0.6B --backend qwen3_fused --output_dir checkpoints/qwen3-0.6b-fused-16-256 --reg_weight 5e-5 --use_wandb --run_name qwen3-0.6b-fused-16-256 --lr 5e-5 --batch_size 64 --k 16,32,64,128,256,151936


#python train_splade_causal.py --train_data training_triplets_multi_neg.jsonl --eval_data interpretation/data/interpretation_w1/randomly_selected_test_amazon.jsonl --base_model Qwen/Qwen3-0.6B --backend qwen3_fused --output_dir checkpoints/qwen3-0.6b-fused --reg_weight 5e-5 --use_wandb --run_name qwen3-0.6b-fused --lr 5e-4 --batch_size 64 


#python train_splade_causal.py --train_data bluesky/train_2epoch_5posts.jsonl --eval_data bluesky/20251114_dev_pairs_5post.jsonl --base_model modernbert-base --backend modernbert_fused_mean --output_dir checkpoints/modernbert-fused-mean-reg5e-4 --reg_weight 5e-4 --use_wandb --run_name mobdernbert-fused-mean-reg5e-4 --lr 1e-4 --batch_size 64

#python train_splade_causal.py --train_data bluesky/train_2epoch_5posts.jsonl --eval_data bluesky/20251114_dev_pairs_5post.jsonl --base_model modernbert-base --backend modernbert_fused_mean --output_dir checkpoints/modernbert-fused-mean-reg5e-3 --reg_weight 5e-3 --use_wandb --run_name mobdernbert-fused-mean-reg5e-3 --lr 1e-4 --batch_size 64

#python train_splade_causal.py --train_data bluesky/train_2epoch_5posts.jsonl --eval_data bluesky/20251114_dev_pairs_5post.jsonl --base_model modernbert-base --backend modernbert_fused_mean --output_dir checkpoints/modernbert-fused-mean-reg5e-3-k16-1024 --reg_weight 5e-3 --use_wandb --run_name mobdernbert-fused-mean-reg5e-3-k16-1024 --lr 1e-4 --batch_size 64 --k 16,32,64,128,256,512,1024

#python train_splade_causal.py --train_data bluesky/train_2epoch_5posts.jsonl --eval_data bluesky/20251114_dev_pairs_5post.jsonl --base_model modernbert-base --backend modernbert_fused_mean --output_dir checkpoints/modernbert-fused-mean-reg5e-2-scale1-20 --reg_weight 5e-2 --use_wandb --run_name mobdernbert-fused-mean-reg5e-2-scale1-20 --lr 1e-4 --batch_size 64 --scale_start 1 --scale_end 20 --reg_total_steps 4000 --reg_start 5e-4

#python train_splade_causal.py --train_data bluesky/train_2epoch_5posts.jsonl --eval_data bluesky/20251114_dev_pairs_5post.jsonl --base_model modernbert-base --backend modernbert_fused_mean --output_dir checkpoints/modernbert-fused-mean-reg5e-3-relu --reg_weight 5e-3 --use_wandb --run_name mobdernbert-fused-mean-reg5e-3-relu --lr 1e-4 --batch_size 64 --activation relu

#python train_splade_causal.py --train_data bluesky/train_2epoch_5posts.jsonl --eval_data bluesky/20251114_dev_pairs_5post.jsonl --base_model modernbert-base --backend modernbert_fused_mean --output_dir checkpoints/modernbert-fused-mean-reg5e-3-scale1-50 --reg_weight 5e-3 --use_wandb --run_name mobdernbert-fused-mean-reg5e-3-scale1-50 --lr 1e-4 --batch_size 64 --scale_start 1 --scale_end 50 

#python train_splade_causal.py --train_data bluesky/train_2epoch_5posts.jsonl --eval_data bluesky/20251114_dev_pairs_5post.jsonl --base_model modernbert-base --backend modernbert_fused_mean --output_dir checkpoints/modernbert-fused-mean-reg5e-2 --reg_weight 5e-2 --use_wandb --run_name mobdernbert-fused-mean-reg5e-2 --lr 1e-4 --batch_size 64

#python train_splade_causal.py --train_data bluesky/train_2epoch_5posts.jsonl --eval_data bluesky/20251114_dev_pairs_5post.jsonl --base_model modernbert-base --backend modernbert_fused_mean --output_dir checkpoints/modernbert-fused-mean-reg5e-3-scale20 --reg_weight 5e-3 --use_wandb --run_name mobdernbert-fused-mean-reg5e-3-scale20 --lr 1e-4 --batch_size 64 --scale 20

#python train_splade_causal.py --train_data bluesky/train_2epoch_5posts.jsonl --eval_data bluesky/20251114_dev_pairs_5post.jsonl --base_model modernbert-base --backend modernbert_fused_mean --output_dir checkpoints/modernbert-fused-mean-reg5e-3-relu --reg_weight 5e-3 --use_wandb --run_name mobdernbert-fused-mean-reg5e-3-relu --lr 1e-4 --batch_size 64 --activation relu

#python train_splade_causal.py --train_data training_data_v1_final_multi_neg.jsonl --eval_data interpretation/data/interpretation_w1/randomly_selected_test_amazon.jsonl --base_model modernbert-base --backend modernbert_fused_mean --output_dir hrs_checkpoints/modernbert-fused-mean-reg5e-3-scale1-20 --reg_weight 5e-3 --use_wandb --run_name mobdernbert-fused-mean-reg5e-3-scale1-20 --lr 1e-4 --batch_size 64 --scale_start 1 --scale_end 20 --scale_total_steps 3000 --reg_total_steps 2000 --reg_start 5e-4 

#python evaluate_hrs.py --parent_dir hrs_release_11-22-24/seed-0/HRS2_english_medium/TA1  --model_name hrs_checkpoints/modernbert-fused-mean-reg5e-3-scale1-20/checkpoint-6326


#python evaluate_hrs.py --parent_dir hrs_release_11-22-24  --model_name hrs_checkpoints/modernbert-fused-mean-reg5e-3-scale20/checkpoint-6326 --output_file hrs_eval/modernbert-fused-mean-reg5e-3-scale20.jsonl
#python evaluate_hrs.py --parent_dir hrs_release_11-22-24  --model_name hrs_checkpoints/modernbert-large-fused-mean-reg5e-4-scale1/checkpoint-6326 --output_file hrs_eval/modernbert-large-fused-mean-reg5e-4-scale1-16384.jsonl --k 16384


#ython train_dense.py   --train_data training_data_v1_final_multi_neg.jsonl --eval_data interpretation/data/interpretation_w1/randomly_selected_test_amazon.jsonl --base_model answerdotai/ModernBERT-large   --output_dir hrs_checkpoints/modernbert-large-dense-2e   --run_name modernbert-large-dense-2e    --epochs 2   --batch_size 64   --lr 3e-4   --use_wandb

#python train_dense.py   --train_data training_data_v1_final_multi_neg.jsonl --eval_data interpretation/data/interpretation_w1/randomly_selected_test_amazon.jsonl --base_model modernbert-base   --output_dir hrs_checkpoints/modernbert-base-dense-2e   --run_name modernbert-base-dense-2e     --epochs 2   --batch_size 64   --lr 3e-4   --use_wandb

#python evaluate_hrs_dense.py --parent_dir hrs_release_11-22-24  --model_name hrs_checkpoints/modernbert-base-dense-2e/checkpoint-12652 --output_file hrs_eval/modernbert-base-dense-2e.jsonl

#python evaluate_hrs_dense.py --parent_dir hrs_release_11-22-24  --model_name hrs_checkpoints/modernbert-large-dense-2e/checkpoint-12652 --output_file hrs_eval/modernbert-large-dense-2e.jsonl

#cd ../pylate/
#python train_colbert.py --run_name colbert-modernbert-base-maxsim --similarity_fn MaxSim 

python train_colbert.py --run_name colbert-modernbert-base-maskedmean-nor-t0.05 --similarity_fn MaskedMean --model_name ../hiatus/modernbert-base --temp 0.05
#python train_colbert.py --run_name colbert-modernbert-base-maskedmean-og-t0.02 --similarity_fn MaskedMean --model_name ../hiatus/modernbert-base --temp 0.02
python train_colbert.py --run_name colbert-modernbert-base-maskedmean-nor --similarity_fn MaskedMean --model_name ../hiatus/modernbert-base --temp 1.0

python rerank_all.py ../hiatus/hrs_preds/modernbert-base-dense --model-name checkpoints/colbert-modernbert-base-maskedmean-nor-t0.05/checkpoint-12652 --similarity-fn MaskedMean --top-k 100 --wandb-project hrs-rerank --run-name colbert-modernbert-base-maskedmean-nor-t0.05
python rerank_all.py ../hiatus/hrs_preds/modernbert-base-dense --model-name checkpoints/colbert-modernbert-base-maskedmean-nor/checkpoint-12652 --similarity-fn MaskedMean --top-k 100 --wandb-project hrs-rerank --run-name colbert-modernbert-base-maskedmean-nor
#python rerank_all.py ../hiatus/hrs_preds/modernbert-base-dense --model-name checkpoints/colbert-modernbert-base-maskedmean-og-t0.02/checkpoint-12652 --similarity-fn MaskedMean --top-k 100 --wandb-project hrs-rerank --run-name colbert-modernbert-base-maskedmean-og-t0.02

#python train_colbert.py --run_name colbert-modernbert-large-base-maskedmean-og --similarity_fn MaskedMean --model_name answerdotai/ModernBERT-large  
python train_colbert.py --run_name colbert-modernbert-base-maskedmean-bs --similarity_fn MaskedMean --model_name ../hiatus/modernbert-base

python train_colbert.py --run_name colbert-modernbert-large-maxsim-triton-t1 --similarity_fn MaxSim --model_name answerdotai/ModernBERT-large --temp 1
python rerank_all.py ../hiatus/hrs_preds/modernbert-large-dense --model-name checkpoints/colbert-modernbert-large-maxsim-triton-t1/checkpoint-6326 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name colbert-modernbert-large-maxsim-triton-t1

python train_colbert.py --run_name colbert-modernbert-base-maxsim-cache-t1-swap --similarity_fn MaxSim --model_name ../hiatus/modernbert-base --temp 1.0
python rerank_all.py ../hiatus/hrs_preds/modernbert-base-dense --model-name checkpoints/colbert-modernbert-base-maxsim-cache-t1-swap/checkpoint-6326 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name colbert-modernbert-base-maxsim-cache-t1-swap

python train_colbert.py --run_name colbert-modernbert-base-maxsim-triton-nor-t0.05 --similarity_fn MaxSim --model_name ../hiatus/modernbert-base --temp 0.05
python rerank_all.py ../hiatus/hrs_preds/modernbert-base-dense --model-name checkpoints/colbert-modernbert-base-maxsim-triton-nor-t0.05/checkpoint-6326 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name colbert-modernbert-base-maxsim-triton-nor-t0.05

python train_colbert.py --run_name colbert-modernbert-large-maxsim-triton-nor-t0.05 --similarity_fn MaxSim --model_name answerdotai/ModernBERT-large --temp 0.05
python rerank_all.py ../hiatus/hrs_preds/modernbert-large-dense --model-name checkpoints/colbert-modernbert-large-maxsim-triton-nor-t0.05/checkpoint-6326 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name colbert-modernbert-large-maxsim-triton-nor-t0.05

python train_colbert.py --run_name colbert-modernbert-base-maxsim-triton-nor-t0.02 --similarity_fn MaxSim --model_name ../hiatus/modernbert-base --temp 0.02
python rerank_all.py ../hiatus/hrs_preds/modernbert-base-dense --model-name checkpoints/colbert-modernbert-base-maxsim-triton-nor-t0.02/checkpoint-6326 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name colbert-modernbert-base-maxsim-triton-nor-t0.02

python train_colbert.py --run_name colbert-modernbert-base-maxsim-cache-t2-swap --similarity_fn MaxSim --model_name ../hiatus/modernbert-base --temp 2.0
python rerank_all.py ../hiatus/hrs_preds/modernbert-base-dense --model-name checkpoints/colbert-modernbert-base-maxsim-cache-t2-swap/checkpoint-6326 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name colbert-modernbert-base-maxsim-cache-t2-swap


python train_colbert.py --run_name colbert-modernbert-base-maxsim-cache-t5-swap --similarity_fn MaxSim --model_name ../hiatus/modernbert-base --temp 5.0
python rerank_all.py ../hiatus/hrs_preds/modernbert-base-dense --model-name checkpoints/colbert-modernbert-base-maxsim-cache-t5-swap/checkpoint-6326 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name colbert-modernbert-base-maxsim-cache-t5-swap

python train_colbert.py --run_name colbert-modernbert-base-maskedmean-bs-t0.05 --similarity_fn MaskedMean --model_name ../hiatus/modernbert-base --temp 0.05
python train_colbert.py --run_name colbert-modernbert-base-maxsim-bs-t0.05 --similarity_fn MaxSim --model_name ../hiatus/modernbert-base --temp 0.05


python rerank_all.py ../hiatus/hrs_preds/modernbert-base-dense --model-name checkpoints/colbert-modernbert-base-maskedmean-og-t0.05/checkpoint-12652 --similarity-fn MaskedMean --top-k 100 --wandb-project hrs-rerank --run-name colbert-modernbert-base-maskedmean-og-t0.05
python rerank_all.py ../hiatus/hrs_preds/modernbert-base-dense --model-name checkpoints/colbert-modernbert-base-maskedmean-og/checkpoint-12652 --similarity-fn MaskedMean --top-k 100 --wandb-project hrs-rerank --run-name colbert-modernbert-base-maskedmean-og
python rerank_all.py ../hiatus/hrs_preds/modernbert-base-dense --model-name checkpoints/colbert-modernbert-base-maskedmean-og-t0.02/checkpoint-12652 --similarity-fn MaskedMean --top-k 100 --wandb-project hrs-rerank --run-name colbert-modernbert-base-maskedmean-og-t0.02

python rerank_all.py ../hiatus/hrs_preds/modernbert-base-dense --model-name checkpoints/colbert-modernbert-base-maxsim-og/checkpoint-12652 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name colbert-modernbert-base-maxsim-og
python rerank_all.py ../hiatus/hrs_preds/modernbert-base-dense --model-name checkpoints/colbert-modernbert-base-maxsim-og-t0.02/checkpoint-12652 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name colbert-modernbert-base-maxsim-og-t0.02
python rerank_all.py ../hiatus/hrs_preds/modernbert-base-dense --model-name checkpoints/colbert-modernbert-base-maxsim-og-t0.05/checkpoint-12652 --similarity-fn MaxSim --top-k 100 --wandb-project hrs-rerank --run-name colbert-modernbert-base-maxsim-og-t0.05



#python mine_hard_negatives.py --model-path hrs_checkpoints/modernbert-large-dense/checkpoint-6326 --dataset training_data_v1_final_dedup.jsonl --anchor-column query --positive-column positive --output-path training_data_v1_anchor_hard_negatives.jsonl

#python mine_hard_negatives.py --model-path hrs_checkpoints/modernbert-large-dense/checkpoint-6326 --dataset training_data_v1_final_dedup.jsonl --anchor-column positive --positive-column query --output-path training_data_v1_positive_hard_negatives.jsonl

python train_dense.py   --train_data training_data_v1_hard_negatives.jsonl  --eval_data interpretation/data/interpretation_w1/randomly_selected_test_amazon.jsonl --base_model answerdotai/ModernBERT-large   --output_dir hrs_checkpoints/modernbert-large-dense-hn   --run_name modernbert-large-dense-hn    --epochs 1   --batch_size 16   --lr 3e-4   --use_wandb

python train_dense.py   --train_data training_data_v1_hard_negatives.jsonl  --eval_data interpretation/data/interpretation_w1/randomly_selected_test_amazon.jsonl --base_model modernbert-base   --output_dir hrs_checkpoints/modernbert-base-dense-hn   --run_name modernbert-base-dense-hn     --epochs 1   --batch_size 16   --lr 3e-4   --use_wandb


python evaluate_hrs_dense.py --parent_dir hrs_release_11-22-24  --model_name hrs_checkpoints/modernbert-base-dense-hn/checkpoint-12652 --output_file hrs_eval/modernbert-base-dense-hn.jsonl

python evaluate_hrs_dense.py --parent_dir hrs_release_11-22-24  --model_name hrs_checkpoints/modernbert-large-dense-hn/checkpoint-12652 --output_file hrs_eval/modernbert-large-dense-hn.jsonl


python evaluate_hrs_dense.py --parent_dir hrs_release_11-22-24  --model_name hrs_checkpoints/modernbert-large-dense/checkpoint-6326 --output_file hrs_eval/modernbert-large-dense.jsonl
python evaluate_hrs_dense.py --parent_dir hrs_release_11-22-24  --model_name hrs_checkpoints/modernbert-base-dense/checkpoint-6326 --output_file hrs_eval/modernbert-base-dense.jsonl
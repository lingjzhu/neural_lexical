import sys
sys.path.append('/home/slimelab/Projects/neural_lexical/exp2')
import torch
from distill_clustered_colbert import CachedDistillationLoss, SparseSelfMultipleNegativesRankingLoss, memory_efficient_colbert_similarity
from sentence_transformers import SparseEncoderTrainer, SparseEncoderTrainingArguments

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')
        self.model_card_data = type('obj', (object,), {'model_id': '', 'set_model_id': lambda x: None, 'set_best_model_step': lambda x: None})()
    
    def tokenize(self, texts):
        return {"input_ids": [[1]], "attention_mask": [[1]]}


model = MockModel()
loss = CachedDistillationLoss(
    model=model,
    sparse_loss=SparseSelfMultipleNegativesRankingLoss(
        model=model, scale=1.0, similarity_fct=memory_efficient_colbert_similarity
    ),
    dense_loss=None,
    query_regularizer_weight=0.0,
    document_regularizer_weight=0.0,
    k=None,
    aux_weight=0.0,
    scale_start=1.0,
    scale_end=1.0,
    total_steps=10,
    reg_start=0.0,
    reg_total_steps=10
)

import datasets
dummy_dataset = datasets.Dataset.from_dict({'anchor': ['a'], 'positive': ['b']})

args = SparseEncoderTrainingArguments(output_dir='./tmp_test')
trainer = SparseEncoderTrainer(model=model, args=args, loss=loss, train_dataset=dummy_dataset, eval_dataset=dummy_dataset)

loss_fn = getattr(trainer, 'loss', None)
if isinstance(loss_fn, dict):
    loss_fn = list(loss_fn.values())[0] if loss_fn else None

print('Trainer loss type:', type(loss_fn))
print('Has attr last_mnrl_loss:', hasattr(loss_fn, 'last_mnrl_loss'))

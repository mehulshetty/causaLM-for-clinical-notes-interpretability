import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class BertCFModel(BertPreTrainedModel):
    def __init__(self, config, lambda_=1.0):
        super(BertCFModel, self).__init__(config)
        self.lambda_ = lambda_
        self.bert = BertModel(config)
        self.mlm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.tc_head = nn.Linear(config.hidden_size, 2)  # Binary classification for chest_pain
        self.cc_head = nn.Linear(config.hidden_size, 2)  # Example: Binary classification for a controlled concept
        self.init_weights()
    
    def forward(self, input_ids, attention_mask=None, labels=None, chest_pain=None, controlled_concept=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
        pooled_output = outputs.pooler_output        # (batch_size, hidden_size)
        
        # MLM Task
        prediction_scores = self.mlm_head(sequence_output)
        mlm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            mlm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        
        # Treated Concept (TC) Task
        # Apply Gradient Reversal Layer
        grl = GradientReversalLayer.apply
        reversed_pooled = grl(pooled_output, self.lambda_)
        tc_logits = self.tc_head(reversed_pooled)
        tc_loss = None
        if chest_pain is not None:
            loss_fct = nn.CrossEntropyLoss()
            tc_loss = loss_fct(tc_logits, chest_pain.view(-1))
        
        # Controlled Concept (CC) Task
        cc_logits = self.cc_head(pooled_output)
        cc_loss = None
        if controlled_concept is not None:
            loss_fct = nn.CrossEntropyLoss()
            cc_loss = loss_fct(cc_logits, controlled_concept.view(-1))
        
        # Combine Losses
        total_loss = 0
        if mlm_loss is not None:
            total_loss += mlm_loss
        if cc_loss is not None:
            total_loss += cc_loss
        if tc_loss is not None:
            total_loss += tc_loss
        
        return total_loss, mlm_loss, cc_loss, tc_loss

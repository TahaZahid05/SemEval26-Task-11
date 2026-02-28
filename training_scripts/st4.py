

import subprocess
import sys

def install_if_missing(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_if_missing('transformers')
install_if_missing('sentencepiece')
install_if_missing('scikit_learn')

import re
import json
import torch
import torch.nn as nn
import time
import random
import numpy as np
import pandas as pd
import os
import shutil
import math
import gc
from collections import Counter, defaultdict

from torch.utils.data import DataLoader, Dataset, Sampler
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from azureml.core import Run
try:
                                                             
    run = Run.get_context()
    if run.id.startswith('OfflineRun'):                       
        run = None
except Exception:
    run = None

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import argparse

parser = argparse.ArgumentParser()
                                                                                
parser.add_argument('--data_file', type=str, help='Path to main JSON file')
parser.add_argument('--gold_file', type=str, help='Path to gold JSON file')
args, unknown = parser.parse_known_args()

MODEL_NAME = "microsoft/mdeberta-v3-base"
BATCH_SIZE = 72                                                          
EVAL_BATCH_SIZE = 72
ACCUMULATION_STEPS = 1                                        
EPOCHS = 3                                                                        
LEARNING_RATE = 5e-06                                        
WARMUP_RATIO = 0.06                      
WEIGHT_DECAY = 0.01                      
NUM_WORKERS = 4
FP16 = True
DRO_WARMUP_EPOCHS = 1 
DRO_TAU = 0.5 

MAX_LEN = 265
BUFFER_SIZE = 16
PATIENCE = 2
MIN_DELTA = 0.001

OUTPUT_DIR = "./outputs" 
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "trained_model")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

JSON_FILE_PATH = args.data_file if args.data_file else "all_languages_merged.json"
GOLD_FILE_PATH = args.gold_file if args.gold_file else "gold_changes.json"

class PreparingInput(Dataset):
    def __init__(self, data, tokenizer, max_len=512, buffer_size=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.buffer_size = buffer_size
        self.sep_token = tokenizer.sep_token
        self.sep_id = tokenizer.sep_token_id
        self.lang_map = {
            'en': 0, 'de': 1, 'es': 2, 'fr': 3, 'it': 4, 'nl': 5, 
            'pt': 6, 'ru': 7, 'zh': 8, 'sw': 9, 'bn': 10, 'te': 11
        }

        self.split_pattern = r'[\.\u3002\u0964]+'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        raw_text = item["syllogism"]

        sentences = [s.strip() for s in re.split(self.split_pattern, raw_text) if s.strip()]

        if not sentences:
             sentences = ["Empty"]

        conclusion = sentences[-1]
        premises = sentences[:-1]

        premises_text = self.sep_token.join(premises)

        encoding = self.tokenizer(
            conclusion,
            premises_text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        premise_spans = torch.zeros(self.buffer_size, 2, dtype=torch.long)
        premise_mask = torch.zeros(self.buffer_size, dtype=torch.float)

        all_sep_indices = (input_ids == self.sep_id).nonzero(as_tuple=True)[0]

        if len(all_sep_indices) > 1:
            sep_positions = all_sep_indices.tolist()
            segment_starts = sep_positions[:-1]
            segment_ends = sep_positions[1:]

            num_premises = len(segment_starts)
            limit = min(num_premises, self.buffer_size)

            for i in range(limit):
                start_pos = segment_starts[i] + 1
                end_pos = segment_ends[i] - 1

                if end_pos >= start_pos:
                    premise_spans[i, 0] = start_pos
                    premise_spans[i, 1] = end_pos
                    premise_mask[i] = 1.0

        if 'validity' in item:
            is_valid = item['validity']
            validity_label = torch.tensor(1.0 if is_valid else 0.0, dtype=torch.float)

            selection_label = torch.zeros(self.buffer_size, dtype=torch.float)
            if is_valid:
                for p_idx in item.get('premises', []):
                    if p_idx < self.buffer_size:
                        selection_label[p_idx] = 1.0
        else:
            validity_label = torch.tensor(0.0, dtype=torch.float)
            selection_label = torch.zeros(self.buffer_size, dtype=torch.float)

        if 'plausibility' in item:
            pl = item['plausibility']
        
            if isinstance(pl, bool):
                plausibility_label = torch.tensor(2 if pl else 0, dtype=torch.long)
        
            elif isinstance(pl, str):
                pl = pl.strip().lower()
                if pl in ("plausible", "true", "yes"):
                    plausibility_label = torch.tensor(2, dtype=torch.long)
                elif pl in ("implausible", "false", "no"):
                    plausibility_label = torch.tensor(0, dtype=torch.long)
                else:
                    plausibility_label = torch.tensor(1, dtype=torch.long)           
        
            else:
                plausibility_label = torch.tensor(1, dtype=torch.long)                    
        else:
            plausibility_label = torch.tensor(1, dtype=torch.long)                     

        lang_id = self.lang_map.get(item.get('language', 'en'), 0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "premise_spans": premise_spans,
            "premise_mask": premise_mask,
            "validity_label": validity_label,
            "selection_label": selection_label,
            "plausibility_label": plausibility_label,
            "language_label": torch.tensor(lang_id, dtype = torch.long)
        }

class VariableSyllogismModel(nn.Module):
    def __init__(self, model_name, dropout_rate=0.1, max_len=512):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.gradient_checkpointing_enable()
        
        hidden_size = self.backbone.config.hidden_size
        self.max_len = max_len

        self.validity_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )

        self.selection_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask, premise_spans, premise_mask=None,
        validity_labels=None, selection_labels=None):

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        cls_vectors = last_hidden_state[:, 0, :]
        validity_logits = self.validity_head(cls_vectors).squeeze(-1)

        batch_size, buffer_size, hidden_size = input_ids.size(0), premise_spans.size(1), last_hidden_state.size(-1)

        max_seq_len = last_hidden_state.size(1)

        positions = torch.arange(max_seq_len, device=input_ids.device).unsqueeze(0).unsqueeze(0)
        positions = positions.expand(batch_size, buffer_size, -1)

        starts = premise_spans[:, :, 0].unsqueeze(-1)
        ends = premise_spans[:, :, 1].unsqueeze(-1)

        in_span = (positions >= starts) & (positions <= ends)

        valid_premise = premise_mask.unsqueeze(-1).bool()

        token_mask = in_span & valid_premise

        expanded_hidden = last_hidden_state.unsqueeze(1).expand(-1, buffer_size, -1, -1)

        masked_hidden = expanded_hidden * token_mask.unsqueeze(-1).float()
        summed = masked_hidden.sum(dim=2)

        counts = token_mask.sum(dim=2, keepdim=True).clamp(min=1)

        premise_vectors = summed / counts

        selection_logits = self.selection_head(premise_vectors).squeeze(-1)

        return {
            'validity_logits': validity_logits,
            'selection_logits': selection_logits
        }

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

def save_model_for_inference(model, tokenizer, save_dir, config_dict):

    os.makedirs(save_dir, exist_ok = True)

    torch.save(model.state_dict(), os.path.join(save_dir, 'pytorch_model.bin'))

    config = {
        'model_name': config_dict['model_name'],
        'max_len': config_dict['max_len'],
        'buffer_size': config_dict['buffer_size'],
        'dropout_rate': 0.1
    }
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    tokenizer.save_pretrained(save_dir)
    print(f"Full Model (Backbone + Heads) saved to '{save_dir}/'")

def load_model_for_inference(save_dir, device='cpu'):
    with open(os.path.join(save_dir, 'config.json'), 'r') as f:
        config = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(save_dir)

    model = VariableSyllogismModel(
        model_name=config['model_name'],
        dropout_rate=config['dropout_rate'],
        max_len=config['max_len']
    )

    model_path = os.path.join(save_dir, 'pytorch_model.bin')
    state_dict = torch.load(model_path, map_location=device)

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model, tokenizer, config

def get_group_ids(validity_labels, plausibility_labels, lang_ids, num_langs = 12):
    """
    Maps each sample to one of 72 groups (2 Validities x 3 Plausibilities x 12 Languages).
    
    validity_labels: 0, 1
    plausibility_labels: 0 (implausible), 1 (neutral), 2 (plausible)
    lang_ids: 0..11
    """
                                                           
    plaus_idx = plausibility_labels.clone()
    plaus_idx[plaus_idx == 0] = 0
    plaus_idx[plaus_idx == 1] = 1
    plaus_idx[plaus_idx == 2] = 2

    group_ids = (validity_labels.long() * (3 * num_langs)) +\
                (plaus_idx.long() * num_langs) +\
                lang_ids.long()
                
    return group_ids

def compute_semeval_metrics(predictions, ground_truth):
    gt_map = {item['id']: item for item in ground_truth}

    total_precision, total_recall, valid_f1_count = 0.0, 0.0, 0

    for pred in predictions:
        item_id = pred['id']
        if item_id not in gt_map: 
            continue
            
        gt_item = gt_map[item_id]

        if not gt_item.get('validity', False):
            continue

        pred_set = set(pred.get('premises_pred', []))
        true_set = set(gt_item.get('premises', []))
        
        tp = len(true_set.intersection(pred_set))
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        total_precision += precision
        total_recall += recall
        valid_f1_count += 1

    f1_premises = 0.0
    if valid_f1_count > 0:
        macro_prec = total_precision / valid_f1_count
        macro_rec = total_recall / valid_f1_count
        if (macro_prec + macro_rec) > 0:
                                            
            f1_premises = 100 * (2 * (macro_prec * macro_rec) / (macro_prec + macro_rec))

    correct_validity = 0
    total_validity = 0
    subgroups = {
        (True, True): [0, 0], (True, False): [0, 0],
        (False, True): [0, 0], (False, False): [0, 0]
    }

    for pred in predictions:
        item_id = pred['id']
        if item_id not in gt_map: 
            continue
            
        gt_item = gt_map[item_id]
        p_val = pred['validity_pred']
        t_val = gt_item['validity']
        
        is_correct = (p_val == t_val)
        total_validity += 1
        if is_correct: 
            correct_validity += 1

        pl = gt_item.get('plausibility', None)
        if isinstance(pl, str):
            pl = pl.strip().lower()
            if pl in ("neutral", "neither", "uncertain"):
                continue
            pl = True if pl in ("plausible", "true", "yes") else False
        
        if isinstance(pl, bool):
            key = (t_val, pl)
            subgroups[key][1] += 1                           
            if is_correct:
                subgroups[key][0] += 1                             

    accuracy = (correct_validity / total_validity * 100) if total_validity > 0 else 0.0

    def get_acc(v, p):
        corr, tot = subgroups[(v, p)]
        return (corr / tot * 100) if tot > 0 else 0.0

    c_intra = (abs(get_acc(True, True) - get_acc(True, False)) +
               abs(get_acc(False, True) - get_acc(False, False))) / 2.0
    c_inter = (abs(get_acc(True, True) - get_acc(False, True)) +
               abs(get_acc(True, False) - get_acc(False, False))) / 2.0

    tot_content_effect = max(0.0, (c_intra + c_inter) / 2.0)
    log_penalty = math.log(1 + tot_content_effect)
    
    overall_performance = (accuracy + f1_premises) / 2.0
    ranking_score = overall_performance / (1 + log_penalty)

    return {
        "validity_accuracy": round(accuracy, 4),
        "premise_f1": round(f1_premises, 4),
        "overall_performance": round(overall_performance, 4),
        "total_bias": round(tot_content_effect, 4),
        "ranking_score": round(ranking_score, 4)
    }

from tqdm.auto import tqdm

def train_engine(model, train_loader, val_loader, epochs, lr, device, 
                 patience=3, min_delta=0.001, save_dir=MODEL_SAVE_DIR, 
                 bias_lambda=1.0, fp16=True):

    torch.cuda.empty_cache()
    gc.collect()
                                                  
    backbone_params = list(model.backbone.named_parameters())
    head_params = list(model.validity_head.named_parameters()) + list(model.selection_head.named_parameters())
    
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
                                         
        {
            "params": [p for n, p in backbone_params if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY, 
            "lr": lr
        },
        {
            "params": [p for n, p in backbone_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0, 
            "lr": lr
        },
                                      
        {
            "params": [p for n, p in head_params if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY, 
            "lr": lr * 10              
        },
        {
            "params": [p for n, p in head_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0, 
            "lr": lr * 10
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters)
    num_update_steps_per_epoch = len(train_loader) // ACCUMULATION_STEPS
    if len(train_loader) % ACCUMULATION_STEPS != 0:
        num_update_steps_per_epoch += 1
        
    total_training_steps = num_update_steps_per_epoch * epochs
    num_warmup_steps = int(total_training_steps * WARMUP_RATIO)
    
    print(f"Total Steps: {total_training_steps} | Warmup Steps: {num_warmup_steps} (Ratio: {WARMUP_RATIO})")
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=total_training_steps
    )
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, mode='max')

    scaler = GradScaler()

    print(f"Training on {device}")
    val_ground_truth = val_loader.dataset.data
    best_ranking_score = 0.0

    MAX_LAMBDA = bias_lambda

    num_languages = 12
    num_groups = 72                                                 

    group_weights = torch.ones(num_groups, device=device) / num_groups

    group_lr_lst = [0, 0.05, 0.08, 0.12, 0.16, 0.2]

    for epoch in range(epochs):
        if epoch < 1:
            current_lambda = 0.0                       
        elif epoch == 1:
            current_lambda = 0.5      
        elif epoch == 2:
            current_lambda = 1.0
        else:
            current_lambda = MAX_LAMBDA

        group_lr = group_lr_lst[epoch]
        
        start_time = time.time()

        model.train()
        total_train_loss = 0
        total_train_bias = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            premise_spans = batch['premise_spans'].to(device)
            premise_mask = batch['premise_mask'].to(device)
            validity_labels = batch['validity_label'].to(device)
            selection_labels = batch['selection_label'].to(device)
            plausibility_labels = batch['plausibility_label'].to(device)
            language_labels = batch['language_label'].to(device)

            if step % ACCUMULATION_STEPS == 0:
                 pass                                  

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16):
                outputs = model(input_ids, mask, premise_spans, premise_mask,
                                validity_labels=validity_labels, selection_labels=selection_labels)
                                            
                loss_fct = nn.BCEWithLogitsLoss(reduction='none')

                raw_val_loss = loss_fct(outputs['validity_logits'], validity_labels)
                                                
                group_ids = get_group_ids(validity_labels, plausibility_labels, language_labels)
                group_losses = torch.zeros(num_groups, device=device)
                
                for g in range(num_groups):
                    mask = group_ids == g
                    if mask.any():
                        group_losses[g] = raw_val_loss[mask].mean()

                if epoch >= DRO_WARMUP_EPOCHS and ((step + 1) % ACCUMULATION_STEPS == 0):
                    with torch.no_grad():
                        group_weights *= torch.exp(group_lr * group_losses.detach())
                        group_weights = torch.clamp(group_weights, min=0.05)                  
                        group_weights /= group_weights.sum()

                sel_loss_raw = loss_fct(outputs['selection_logits'], selection_labels)
                masked_sel_loss = sel_loss_raw * premise_mask
                sel_loss = masked_sel_loss.sum() / (premise_mask.sum() + 1e-9)

                val_loss = torch.sum(group_weights * group_losses)

                bias_penalty = torch.tensor(0.0, device=device)

                logits = outputs['validity_logits']
                confidences = torch.where(validity_labels == 1, logits, -logits)

                intra_diffs = []
                valid_plaus_mask = plausibility_labels != 1
                unique_plaus = torch.unique(plausibility_labels[valid_plaus_mask])

                for p in unique_plaus:
                                                                                     
                    mask_p = (plausibility_labels == p) & valid_plaus_mask
                    lbls_p = validity_labels[mask_p]
                    confs_p = confidences[mask_p]

                    if (lbls_p == 1).any() and (lbls_p == 0).any():
                        mean_conf_valid = confs_p[lbls_p == 1].mean()
                        mean_conf_invalid = confs_p[lbls_p == 0].mean()
                        intra_diffs.append(torch.abs(mean_conf_valid - mean_conf_invalid))
                
                intra_loss = torch.stack(intra_diffs).mean() if intra_diffs else torch.tensor(0.0, device=device)

                cross_diffs = []
                unique_val = torch.unique(validity_labels)
                
                for v in unique_val:
                                                                             
                    mask_v = (validity_labels == v)
                    mask_v = (validity_labels == v) & (plausibility_labels != 1)
                    lbls_plaus = plausibility_labels[mask_v]

                    confs_v = confidences[mask_v]

                    valid_mask = (plausibility_labels == 0) | (plausibility_labels == 2)
                    mask_v = (validity_labels == v) & valid_mask
                    
                    lbls_plaus = plausibility_labels[mask_v]
                    confs_v = confidences[mask_v]
                    
                    if (lbls_plaus == 2).any() and (lbls_plaus == 0).any():
                        mean_conf_plaus = confs_v[lbls_plaus == 2].mean()
                        mean_conf_implaus = confs_v[lbls_plaus == 0].mean()
                        cross_diffs.append(torch.abs(mean_conf_plaus - mean_conf_implaus))

                cross_loss = torch.stack(cross_diffs).mean() if cross_diffs else torch.tensor(0.0, device=device)

                if intra_diffs and cross_diffs:
                    bias_penalty = (intra_loss + cross_loss) / 2.0
                else:
                    bias_penalty = torch.tensor(0.0, device=device)

                loss = val_loss + (1.0 * sel_loss) + (current_lambda * bias_penalty)

                loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()
                               
            if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            total_train_loss += loss.item() * ACCUMULATION_STEPS
            total_train_bias += bias_penalty.item()
            progress_bar.set_postfix({
                'loss': f"{loss.item() * ACCUMULATION_STEPS:.4f}", 
                'bias_pen': f"{bias_penalty.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_bias = total_train_bias / len(train_loader)

        torch.cuda.empty_cache()
        gc.collect()

        model.eval()
        total_val_loss = 0
        total_val_bias = 0
        val_predictions = []
        global_idx = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                premise_spans = batch['premise_spans'].to(device)
                premise_mask = batch['premise_mask'].to(device)
                validity_labels = batch['validity_label'].to(device)
                selection_labels = batch['selection_label'].to(device)
                plausibility_labels = batch['plausibility_label'].to(device)
                language_labels = batch['language_label'].to(device)

                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16):
                    outputs = model(input_ids, mask, premise_spans, premise_mask)

                    loss_fct = nn.BCEWithLogitsLoss(reduction='none')

                    raw_val_loss = loss_fct(outputs['validity_logits'], validity_labels)
                    group_ids = get_group_ids(validity_labels, plausibility_labels, language_labels)
                    group_losses = torch.zeros(num_groups, device=device)
                    
                    for g in range(num_groups):
                        mask = group_ids == g
                        if mask.any():
                            group_losses[g] = raw_val_loss[mask].mean()

                    s_loss_raw = loss_fct(outputs['selection_logits'], selection_labels)
                    s_loss = (s_loss_raw * premise_mask).sum() / (premise_mask.sum() + 1e-9)

                    bias_penalty = torch.tensor(0.0, device=device)

                    bias_penalty = torch.tensor(0.0, device=device)
                    logits = outputs['validity_logits']
                    confidences = torch.where(validity_labels == 1, logits, -logits)

                    intra_diffs = []
                    valid_plaus_mask = plausibility_labels != 1                   
                    unique_plaus = torch.unique(plausibility_labels[valid_plaus_mask])

                    for p in unique_plaus:
                        mask_p = (plausibility_labels == p) & valid_plaus_mask
                        lbls_p = validity_labels[mask_p]
                        confs_p = confidences[mask_p]
                        if (lbls_p == 1).any() and (lbls_p == 0).any():
                            intra_diffs.append(torch.abs(confs_p[lbls_p==1].mean() - confs_p[lbls_p==0].mean()))
                    intra_loss = torch.stack(intra_diffs).mean() if intra_diffs else torch.tensor(0.0, device=device)

                    cross_diffs = []
                    unique_val = torch.unique(validity_labels)
                    
                    for v in unique_val:
                        valid_mask = (plausibility_labels == 0) | (plausibility_labels == 2)
                        mask_v = (validity_labels == v) & valid_mask
                    
                        lbls_plaus = plausibility_labels[mask_v]
                        confs_v = confidences[mask_v]
                    
                        if (lbls_plaus == 2).any() and (lbls_plaus == 0).any():
                            mean_conf_plaus = confs_v[lbls_plaus == 2].mean()
                            mean_conf_implaus = confs_v[lbls_plaus == 0].mean()
                            cross_diffs.append(torch.abs(mean_conf_plaus - mean_conf_implaus))
                    
                    cross_loss = torch.stack(cross_diffs).mean() if cross_diffs else torch.tensor(0.0, device=device)

                    if intra_diffs and cross_diffs:
                        bias_penalty = (intra_loss + cross_loss) / 2.0
                    else:
                        bias_penalty = torch.tensor(0.0, device=device)

                    v_loss = torch.sum(group_weights * group_losses)
                    batch_val_loss = v_loss + (1.0 * s_loss) + (current_lambda * bias_penalty)
                    
                    total_val_loss += batch_val_loss.item()
                    total_val_bias += bias_penalty.item()

                val_probs = torch.sigmoid(outputs['validity_logits'])
                sel_probs = torch.sigmoid(outputs['selection_logits'])

                for i in range(len(val_probs)):
                    current_id = val_ground_truth[global_idx]['id']
                    global_idx += 1
                    p_mask = premise_mask[i].bool()
                    real_s_probs = sel_probs[i][p_mask]

                    result = {'id': current_id}
                    result['validity_pred'] = (val_probs[i].item() >= 0.5)

                    if result['validity_pred']:
                        k = min(2, len(real_s_probs))
                        if k > 0:
                            top_indices = torch.argsort(real_s_probs, descending=False)[-k:]
                            result['premises_pred'] = sorted(top_indices.tolist())
                        else:
                            result['premises_pred'] = []
                    else:
                        result['premises_pred'] = []

                    val_predictions.append(result)

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_bias = total_val_bias / len(val_loader)

        metrics = compute_semeval_metrics(val_predictions, val_ground_truth)
        current_ranking_score = metrics['ranking_score']

        epoch_time = time.time() - start_time

        print(f"\nEpoch {epoch + 1}/{epochs} Completed in {epoch_time:.0f}s")

        print(f"   Loss  (Total): Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        print(f"   Bias Penalty : Train: {avg_train_bias:.4f} | Val: {avg_val_bias:.4f}")

        print(f"   Val Acc: {metrics['validity_accuracy']:.2f}% | Premise F1: {metrics['premise_f1']:.2f}%")
        print(f"   Metric Bias: {metrics['total_bias']:.4f} (Calculated by SemEval script)")
        print(f"   Ranking Score: {metrics['ranking_score']:.4f}")

        if 'run' in globals() and run is not None:
            run.log("train_loss", avg_train_loss)
            run.log("val_loss", avg_val_loss)
            run.log("ranking_score", current_ranking_score)

        if current_ranking_score > best_ranking_score:
            best_ranking_score = current_ranking_score
            save_model_for_inference(model, train_loader.dataset.tokenizer, save_dir,
                                     {'model_name': MODEL_NAME, 'max_len': MAX_LEN, 'buffer_size': BUFFER_SIZE})
            print("   Checkpoint Saved!")

        if early_stopping(current_ranking_score):
            print(f"\nEarly Stopping triggered.")
            break

        torch.cuda.empty_cache()
        gc.collect()

    return model

def predict(model, dataloader, device):
    model.eval()
    predictions = []
    raw_data = dataloader.dataset.data
    global_idx = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            premise_spans = batch['premise_spans'].to(device)
            premise_mask = batch['premise_mask'].to(device)

            with autocast():
                outputs = model(input_ids, mask, premise_spans, premise_mask)

            val_probs = torch.sigmoid(outputs['validity_logits'])
            sel_probs = torch.sigmoid(outputs['selection_logits'])

            for i in range(len(val_probs)):
                current_id = raw_data[global_idx]['id']
                global_idx += 1
                p_mask = premise_mask[i].bool()
                real_s_probs = sel_probs[i][p_mask]

                result = {'id': current_id}
                result['validity_pred'] = (val_probs[i].item() >= 0.5)

                if result['validity_pred']:
                    k = min(2, len(real_s_probs))
                    if k > 0:
                        top_indices = torch.argsort(real_s_probs, descending=False)[-k:]
                        result['premises_pred'] = sorted(top_indices.tolist())
                    else:
                        result['premises_pred'] = []
                else:
                    result['premises_pred'] = []

                predictions.append(result)
    return predictions

def compute_group_id(item):
    validity = int(item['validity'])
    pl = item.get('plausibility', 1)
    if isinstance(pl, bool):
        pl = 2 if pl else 0
    elif isinstance(pl, str):
        pl = 1

    lang_id = int(item.get('lang_id', 0))

    group_id = (validity * (3 * 12)) + (pl * 12) + lang_id
    return group_id

class StratifiedBatchSampler(Sampler):
    def __init__(self, group_ids, batch_size, num_groups=6):
        self.group_ids = np.array(group_ids)
        self.batch_size = batch_size
        self.num_groups = num_groups

        self.groups = defaultdict(list)
        for idx, g in enumerate(self.group_ids):
            self.groups[g].append(idx)
        
        self.group_keys = list(self.groups.keys())

        self.samples_per_group = max(1, batch_size // num_groups)

        min_group_size = min(len(v) for v in self.groups.values())
        self.num_batches = min_group_size // self.samples_per_group

    def __iter__(self):
                                                        
        shuffled_groups = {}
        for g in self.group_keys:
            shuffled = np.array(self.groups[g]).copy()
            np.random.shuffle(shuffled)
            shuffled_groups[g] = shuffled

        for i in range(self.num_batches):
            batch = []

            for g in self.group_keys:
                start = i * self.samples_per_group
                end = (i + 1) * self.samples_per_group
                batch.extend(shuffled_groups[g][start:end].tolist())

            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches

if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    try:
                                                                  
        with open(JSON_FILE_PATH, 'r') as f:
            full_data = json.load(f)
        print(f"Loaded {len(full_data)} samples.")
    except Exception as e:
        print(f"Error loading data from {JSON_FILE_PATH}: {e}")
        exit()

    with open(GOLD_FILE_PATH, 'r') as f:
        gold_data = json.load(f)

    if full_data:
        random.seed(42)
        gold_stratify = [f"{x['validity']}_{x.get('plausibility', 'neutral')}" for x in gold_data]
    
        train_gold, temp_gold, _, temp_gold_labels = train_test_split(
            gold_data, gold_stratify, test_size=0.20, random_state=42, stratify=gold_stratify
        )
        val_gold, test_gold = train_test_split(
            temp_gold, test_size=0.5, random_state=42, stratify=temp_gold_labels
        )

        aug_stratify = [f"{x['validity']}_{x.get('plausibility', 'neutral')}" for x in full_data]
        
        train_aug, temp_aug, _, temp_aug_labels = train_test_split(
            full_data, aug_stratify, test_size=0.20, random_state=42, stratify=aug_stratify
        )
        val_aug, test_aug = train_test_split(
            temp_aug, test_size=0.5, random_state=42, stratify=temp_aug_labels
        )

        train_data = train_gold + train_aug
        val_data = val_gold + val_aug
        test_data = test_gold + test_aug

        random.shuffle(train_data)                                         
        random.shuffle(val_data)
        random.shuffle(test_data)

        group_ids = [compute_group_id(x) for x in train_data]

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        train_ds = PreparingInput(train_data, tokenizer, max_len=MAX_LEN, buffer_size=BUFFER_SIZE)
        val_ds = PreparingInput(val_data, tokenizer, max_len=MAX_LEN, buffer_size=BUFFER_SIZE)
        test_ds = PreparingInput(test_data, tokenizer, max_len=MAX_LEN, buffer_size=BUFFER_SIZE)

        sampler = StratifiedBatchSampler(group_ids, BATCH_SIZE)

        train_loader = DataLoader(
            train_ds,
            batch_sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        val_loader = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        model = VariableSyllogismModel(MODEL_NAME, max_len=MAX_LEN).to(device)

        trained_model = train_engine(
            model, train_loader, val_loader,
            epochs=EPOCHS, lr=LEARNING_RATE, device=device,
            patience=PATIENCE, min_delta=MIN_DELTA, save_dir=MODEL_SAVE_DIR,
            bias_lambda=2.0,
            fp16=FP16
        )

        print("\nLoading best model checkpoint for test predictions...")
        best_model, _, _ = load_model_for_inference(MODEL_SAVE_DIR, device=device)
        
        print("\nPredicting on Test Set...")
        test_predictions = predict(best_model, test_loader, device)
        metrics = compute_semeval_metrics(test_predictions, test_data)

        if 'run' in globals() and run is not None:
            run.log("Final_Test_Accuracy", metrics['validity_accuracy'])
            run.log("Final_Ranking_Score", metrics['ranking_score'])
            run.log("Final_Bias", metrics['total_bias'])

        print("\n" + "="*40)
        print("FINAL TEST RESULTS")
        print("="*40)
        print(f"Validity Acc: {metrics['validity_accuracy']:.2f}%")
        print(f"Ranking Score: {metrics['ranking_score']:.4f}")

        pred_path = os.path.join(OUTPUT_DIR, "train_predictions.json")
        with open(pred_path, "w") as f:
            json.dump(test_predictions, f, indent=4)

        print(f"\nResults synced to Azure Storage via {OUTPUT_DIR}")
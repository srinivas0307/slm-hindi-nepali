# %% [markdown]
# # Finetune Nepali Model on Reasoning Data

# %%
# UPDATE THESE PATHS for Kaggle
TRAIN_DATA_PATH = "/kaggle/input/reasoning-data-nepali/train_5k.jsonl"
VAL_DATA_PATH = "/kaggle/input/reasoning-data-nepali/val.jsonl"
PRETRAINED_CHECKPOINT = "/kaggle/input/nepali-pretrained-model/best_model.pt"
TOKENIZER_PATH = "/kaggle/input/nepali-tokenizer/tokenizer.model"
LM_TEST_BIN = "/kaggle/input/notebooks/srinivasu2005/tokenize-nepali/nepali_tokenized/test.bin"
OUTPUT_DIR = "/kaggle/working/finetuned_model_nepali"
BASELINE_LM_PPL = 24.01  # Phase 2 pretrained Nepali test PPL

# %%
from dataclasses import dataclass
import os, sys, math, time, json, random, logging, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

# Set up logging
handlers = [logging.StreamHandler(sys.stdout)]
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    handlers.append(logging.FileHandler(os.path.join(OUTPUT_DIR, 'finetune.log')))
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=handlers
)
logger = logging.getLogger(__name__)

# Set seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)

# %%
@dataclass
class TransformerConfig:
    vocab_size: int = 4000
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1184
    max_seq_len: int = 512
    dropout: float = 0.1
    tie_weights: bool = True
    rope_base: int = 10000
    pad_id: int = 0; bos_id: int = 2; eos_id: int = 3
    @property
    def d_k(self): return self.d_model // self.n_heads

def precompute_rope_frequencies(d_k, max_seq_len, base=10000, device=None):
    freq = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)
    theta = 1.0 / (base ** (freq / d_k))
    pos = torch.arange(0, max_seq_len, dtype=torch.float32, device=device)
    angles = torch.outer(pos, theta)
    return torch.cos(angles), torch.sin(angles)

def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c = cos.unsqueeze(0).unsqueeze(0); s = sin.unsqueeze(0).unsqueeze(0)
    return torch.stack([x1*c - x2*s, x1*s + x2*c], dim=-1).flatten(-2)

class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads, self.d_k, self.d_model = cfg.n_heads, cfg.d_k, cfg.d_model
        self.W_Q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_K = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_V = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_O = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        cos, sin = precompute_rope_frequencies(cfg.d_k, cfg.max_seq_len, cfg.rope_base)
        self.register_buffer("cos_table", cos); self.register_buffer("sin_table", sin)
        self.register_buffer("causal_mask", torch.triu(torch.ones(cfg.max_seq_len, cfg.max_seq_len), diagonal=1).bool())
    def forward(self, x, return_attention=False):
        B, T, _ = x.shape
        Q = self.W_Q(x).view(B,T,self.n_heads,self.d_k).transpose(1,2)
        K = self.W_K(x).view(B,T,self.n_heads,self.d_k).transpose(1,2)
        V = self.W_V(x).view(B,T,self.n_heads,self.d_k).transpose(1,2)
        c, s = self.cos_table[:T], self.sin_table[:T]
        Q, K = apply_rope(Q,c,s), apply_rope(K,c,s)
        sc = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
        sc = sc.masked_fill(self.causal_mask[:T,:T].unsqueeze(0).unsqueeze(0), float('-inf'))
        aw = torch.softmax(sc, dim=-1)
        ctx = torch.matmul(self.attn_dropout(aw), V).transpose(1,2).contiguous().view(B,T,self.d_model)
        return (self.resid_dropout(self.W_O(ctx)), aw) if return_attention else (self.resid_dropout(self.W_O(ctx)), None)

class SwiGLUFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.W_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.W_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.W_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
    def forward(self, x): return self.dropout(self.W_down(F.silu(self.W_gate(x)) * self.W_up(x)))

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model); self.attn = MultiHeadCausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model); self.ffn = SwiGLUFeedForward(cfg)
    def forward(self, x, return_attention=False):
        ao, aw = self.attn(self.ln1(x), return_attention); x = x + ao; x = x + self.ffn(self.ln2(x)); return x, aw

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.emb_dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_ln = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights: self.lm_head.weight = self.embedding.weight
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None: torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, nn.LayerNorm):
                torch.nn.init.ones_(m.weight); torch.nn.init.zeros_(m.bias)
    def forward(self, input_ids, targets=None, return_attention=False):
        x = self.emb_dropout(self.embedding(input_ids))
        all_attn = [] if return_attention else None
        for block in self.blocks:
            x, aw = block(x, return_attention)
            if return_attention and aw is not None: all_attn.append(aw)
        logits = self.lm_head(self.final_ln(x))
        result = {"logits": logits}
        if targets is not None:
            result["loss"] = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1), ignore_index=self.config.pad_id)
        if return_attention: result["attention_weights"] = all_attn
        return result

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=128, temperature=0.0, top_k=0, repetition_penalty=1.0):
        self.eval()
        for _ in range(max_new_tokens):
            if input_ids.size(1) > self.config.max_seq_len:
                input_ids_cond = input_ids[:, -self.config.max_seq_len:]
            else:
                input_ids_cond = input_ids
            
            logits = self(input_ids_cond)["logits"][:, -1, :]
            
            if repetition_penalty != 1.0:
                for i in range(input_ids.size(0)):
                    for token in set(input_ids[i].tolist()):
                        if logits[i, token] < 0:
                            logits[i, token] *= repetition_penalty
                        else:
                            logits[i, token] /= repetition_penalty
                            
            if temperature == 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if (next_token == self.config.eos_id).all():
                break
                
        return input_ids

# %% [markdown]
# ## Dataset

# %%
class ReasoningDataset(Dataset):
    def __init__(self, data_path, sp_model, max_seq_len=512):
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        self.sp = sp_model
        self.max_seq_len = max_seq_len
        self.pad_id = 0
        self.bos_id = 2
        self.eos_id = 3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = f"{item['input']} उत्तर: "
        target_text = f"{item['target']}"
        
        prompt_ids = self.sp.encode(prompt)
        target_ids = self.sp.encode(target_text)
        
        # Format: [BOS] prompt answer [EOS]
        full_seq = [self.bos_id] + prompt_ids + target_ids + [self.eos_id]
        
        # Labels: ignore prompt parts
        labels = [self.pad_id] * (1 + len(prompt_ids)) + target_ids + [self.eos_id]
        
        if len(full_seq) > self.max_seq_len:
            full_seq = full_seq[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
        
        # Pad
        pad_len = self.max_seq_len - len(full_seq)
        if pad_len > 0:
            full_seq = full_seq + [self.pad_id] * pad_len
            labels = labels + [self.pad_id] * pad_len
            
        x = torch.tensor(full_seq[:-1], dtype=torch.long)
        y = torch.tensor(labels[1:], dtype=torch.long)
        
        return x, y

# %% [markdown]
# ## Finetuning

# %%
class TokenizedDataset(Dataset):
    """Memory-mapped dataset for evaluating language modeling perplexity on pretraining test set."""
    def __init__(self, bin_path, seq_len=512):
        self.seq_len = seq_len
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.n_samples = max(0, (len(self.data) - 1) // seq_len)
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        start = idx * self.seq_len
        c = self.data[start:start+self.seq_len+1].astype(np.int64)
        return torch.from_numpy(c[:-1]), torch.from_numpy(c[1:])

# %%
def evaluate_lm_ppl(model, lm_loader, device, max_batches=50):
    """Computes perplexity on language modeling test set to monitor catastrophic forgetting."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for idx, (x, y) in enumerate(lm_loader):
            if max_batches and idx >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=device.type=='cuda'):
                res = model(x, targets=y)
            non_pad = (y != 0).sum().item()
            total_loss += res["loss"].item() * non_pad
            total_tokens += non_pad
    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    return avg_loss, ppl

# %%
def create_optimizer(model, weight_decay, learning_rate):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
    return optimizer

def get_lr(it, max_it, peak_lr, min_lr, warmup_steps=200):
    if it < warmup_steps:
        return peak_lr * (it + 1) / warmup_steps
    if it > max_it:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_it - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (peak_lr - min_lr)

def exact_match_accuracy(model, sp, val_data, device, num_samples=100):
    model.eval()
    correct = 0
    total = 0
    # Evaluate on a subset or full depending on size
    eval_subset = random.sample(val_data.data, min(num_samples, len(val_data.data)))
    
    with torch.no_grad():
        for item in eval_subset:
            prompt = f"{item['input']} उत्तर: "
            gold = item['target'].strip()
            
            prompt_ids = [2] + sp.encode(prompt)
            x = torch.tensor([prompt_ids], dtype=torch.long).to(device)
            
            out_ids = model.generate(x, max_new_tokens=64, temperature=0.0)[0]
            # Strip prompt
            gen_ids = out_ids[len(prompt_ids):].tolist()
            if 3 in gen_ids:
                gen_ids = gen_ids[:gen_ids.index(3)]
                
            pred = sp.decode(gen_ids).strip()
            if pred == gold:
                correct += 1
            total += 1
            
    return correct / total if total > 0 else 0.0

# %%
def train(args=None):
    train_data_path = getattr(args, 'train_data', None) or TRAIN_DATA_PATH
    val_data_path = getattr(args, 'val_data', None) or VAL_DATA_PATH
    pretrained_checkpoint = getattr(args, 'pretrained_checkpoint', None) or PRETRAINED_CHECKPOINT
    tokenizer_path = getattr(args, 'tokenizer_path', None) or getattr(args, 'tokenizer', None) or TOKENIZER_PATH
    lm_test_bin = getattr(args, 'lm_test_bin', None) or LM_TEST_BIN
    output_dir = getattr(args, 'output_dir', None) or OUTPUT_DIR
    epochs = getattr(args, 'epochs', None) if getattr(args, 'epochs', None) is not None else 5
    eval_lm_batches = getattr(args, 'eval_lm_batches', None) if getattr(args, 'eval_lm_batches', None) is not None else 50
    baseline_ppl = getattr(args, 'baseline_lm_ppl', None) if getattr(args, 'baseline_lm_ppl', None) is not None else (getattr(args, 'baseline_ppl', None) if getattr(args, 'baseline_ppl', None) is not None else BASELINE_LM_PPL)
    batch_size = getattr(args, 'batch_size', None) if getattr(args, 'batch_size', None) is not None else 16
    grad_accum_steps = getattr(args, 'grad_accum', None) if getattr(args, 'grad_accum', None) is not None else 2
    peak_lr = getattr(args, 'lr', None) if getattr(args, 'lr', None) is not None else 1e-4

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Training on dataset: {train_data_path}")
    logger.info(f"Output directory: {output_dir}")
    
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    logger.info(f"Loaded tokenizer from {tokenizer_path}")
    
    train_dataset = ReasoningDataset(train_data_path, sp)
    val_dataset = ReasoningDataset(val_data_path, sp)
    logger.info(f"Loaded reasoning data — Train: {len(train_dataset)} examples, Val: {len(val_dataset)} examples")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # LM benchmark loader
    lm_loader = None
    if os.path.exists(lm_test_bin):
        lm_dataset = TokenizedDataset(lm_test_bin, seq_len=512)
        lm_loader = DataLoader(lm_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        logger.info(f"Loaded LM test dataset from {lm_test_bin} ({len(lm_dataset)} sequences)")
    else:
        logger.warning(f"LM test bin not found at {lm_test_bin}. PPL monitoring will be skipped.")
    
    # Nepali config
    config = TransformerConfig(vocab_size=4000, d_ff=1184)
    model = DecoderOnlyTransformer(config).to(device)
    
    # Load pretrained weights
    logger.info(f"Loading pretrained checkpoint from {pretrained_checkpoint}")
    try:
        checkpoint = torch.load(pretrained_checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(pretrained_checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    logger.info("Loaded pretrained weights successfully")
    
    # Pre-finetuning LM PPL check
    initial_lm_ppl = None
    if lm_loader is not None:
        init_loss, initial_lm_ppl = evaluate_lm_ppl(model, lm_loader, device, max_batches=eval_lm_batches)
        logger.info(f"Initial Pretrained LM Test Loss: {init_loss:.4f} | PPL: {initial_lm_ppl:.2f} (Reported baseline: {baseline_ppl:.2f})")
    
    min_lr = 1e-6
    weight_decay = 0.1
    optimizer = create_optimizer(model, weight_decay, peak_lr)
    
    scaler = torch.amp.GradScaler('cuda', enabled=device.type=='cuda')
    
    total_steps = len(train_loader) * epochs // grad_accum_steps
    step = 0
    
    best_val_acc = -1.0
    patience = 3
    patience_counter = 0
    
    training_log = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            lr = get_lr(step, total_steps, peak_lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            with torch.amp.autocast(device_type=device.type, enabled=device.type=='cuda'):
                res = model(x, targets=y)
                loss = res['loss'] / grad_accum_steps
                
            scaler.scale(loss).backward()
            
            train_loss += loss.item() * grad_accum_steps
            
            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                step += 1
                
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        # Validation on reasoning
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=device.type=='cuda'):
                    res = model(x, targets=y)
                val_loss += res['loss'].item()
        avg_val_loss = val_loss / len(val_loader)
        
        val_acc = exact_match_accuracy(model, sp, val_dataset, device)
        
        logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.4f} - LR: {lr:.2e}")
        # Perplexity evaluation on normal language
        lm_loss, lm_ppl = None, None
        degradation_pct = 0.0
        if lm_loader is not None:
            lm_loss, lm_ppl = evaluate_lm_ppl(model, lm_loader, device, max_batches=eval_lm_batches)
            base_ref = initial_lm_ppl if initial_lm_ppl else baseline_ppl
            degradation_pct = ((lm_ppl - base_ref) / base_ref) * 100.0
            if degradation_pct > 20.0:
                logger.warning(f"⚠️ Catastrophic forgetting risk! LM PPL degraded by {degradation_pct:+.1f}% ({base_ref:.2f} -> {lm_ppl:.2f})")
                
        status_msg = (
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Reasoning Val Acc: {val_acc*100:.2f}%"
        )
        if lm_ppl is not None:
            status_msg += f" | LM Test PPL: {lm_ppl:.2f} (Δ: {degradation_pct:+.1f}%)"
        logger.info(status_msg)
        
        log_entry = {
            "epoch": epoch,
            "train_samples": len(train_dataset),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_acc,
            "lm_test_loss": lm_loss,
            "lm_test_ppl": lm_ppl,
            "lm_degradation_pct": degradation_pct,
            "lr": lr
        }
        training_log.append(log_entry)
        
        # Save JSON log
        with open(os.path.join(output_dir, 'training_log.json'), 'w') as f:
            json.dump(training_log, f, indent=4)
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_path = os.path.join(output_dir, 'best_finetuned_model.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'val_accuracy': val_acc,
                'lm_test_ppl': lm_ppl,
                'config': config
            }, save_path)
            lm_ppl_str = f"{lm_ppl:.2f}" if lm_ppl is not None else "N/A"
            logger.info(f"Saved new best model with Val Acc: {val_acc*100:.2f}% (LM PPL: {lm_ppl_str})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break

    logger.info(f"Finetuning completed. Best Reasoning Val Acc: {best_val_acc*100:.2f}%")

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Nepali model on Reasoning dataset")
    parser.add_argument("--train_data", type=str, default=TRAIN_DATA_PATH)
    parser.add_argument("--val_data", type=str, default=VAL_DATA_PATH)
    parser.add_argument("--pretrained_checkpoint", type=str, default=PRETRAINED_CHECKPOINT)
    parser.add_argument("--tokenizer_path", "--tokenizer", dest="tokenizer_path", type=str, default=TOKENIZER_PATH, help="Path to tokenizer.model")
    parser.add_argument("--lm_test_bin", type=str, default=LM_TEST_BIN)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval_lm_batches", type=int, default=50)
    parser.add_argument("--baseline_lm_ppl", "--baseline_ppl", dest="baseline_lm_ppl", type=float, default=BASELINE_LM_PPL, help="Baseline pretrained LM test PPL")
    args, _ = parser.parse_known_args()
    
    train(args)

'''
# transformer_scratch.py
# This code implements a GPT-like transformer model from scratch using PyTorch.
# It includes a custom dataset class for tokenizing text, a multi-head attention 
# mechanism, layer normalization, feed-forward layers, and a complete transformer block.
# It also includes a trainer class for training the model on a text dataset, 
# evaluating it, and generating text samples. The model is trained on a text file, 
# and it can generate text based on a given context. # The code is structured to allow
# for easy configuration and training of the model.
'''
import os
import torch
import tiktoken
import urllib
import urllib.request
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# how the GPTDataset class works:
# The GPTDataset class is a custom PyTorch Dataset that prepares text data for training a GPT-like model.
# It tokenizes the input text using a tokenizer, splits it into chunks of a specified maximum length,
# and creates input-target pairs for training. Each input chunk is a sequence of tokens, and    
# the corresponding target chunk is the same sequence shifted by one token, which is used for next-token prediction.
class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.multiply(x, self.W_key, b, num_tokens)
        queries = self.multiply(x, self.W_query, b, num_tokens)
        values = self.multiply(x, self.W_value, b, num_tokens)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec
    
    def multiply(self, x, W, b, num_tokens):
        mul = W(x).view(b, num_tokens, self.num_heads, self.head_dim)
        return mul.transpose(1, 2)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class GPTTrainer:
    def __init__(self, config, seed=123):
        self.config = config
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GPTModel(config)
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.optimizer = self.get_optimizser()
        self.train_loader = None
        self.val_loader = None
        self._setup()

    def _setup(self):
        torch.manual_seed(self.seed)
        self.model.to(self.device)
    
    def get_optimizser(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=0.0004, weight_decay=0.1
        )
        return optimizer

    def text_to_token_ids(self, text):
        encoded = self.tokenizer.encode(text, allowed_special={""})
        return torch.tensor(encoded).unsqueeze(0)

    def token_ids_to_text(self, token_ids):
        return self.tokenizer.decode(token_ids.squeeze(0).tolist())

    def load_data(self, train_ratio=0.90):
        file_path = self.config["text_file_path"]
        url = self.config["url"]
        if not os.path.exists(file_path):
            with urllib.request.urlopen(url) as response:
                text_data = response.read().decode("utf-8")
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text_data)
        else:
            with open(file_path, "r", encoding="utf-8") as file:
                text_data = file.read()

        split_idx = int(train_ratio * len(text_data))
        train_data, val_data = text_data[:split_idx], text_data[split_idx:]

        self.train_loader = self.create_dataloader(
            train_data,
            batch_size=1,
            max_length=self.config["context_length"],
            stride=self.config["context_length"],
            drop_last=True,
            shuffle=True,
            num_workers=0,
        )
        self.val_loader = self.create_dataloader(
            val_data,
            batch_size=1,
            max_length=self.config["context_length"],
            stride=self.config["context_length"],
            drop_last=False,
            shuffle=False,
            num_workers=0,
        )

    def create_dataloader(
        self,
        txt,
        batch_size=4,
        max_length=256,
        stride=128,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    ):
        tokenizer = tiktoken.get_encoding("gpt2")
        dataset = GPTDataset(txt, tokenizer, max_length, stride) # dataset has # input_ids and target_ids after tokenization
        dataloader = DataLoader( #what it does is it creates a DataLoader object that can be used to iterate over the dataset in batches.
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )

        return dataloader

    def calc_loss_batch(self, input_batch, target_batch):
        #what id does is it moves the input and target batches to the specified device (CPU or GPU). Which is used for training the model. 
        # it returns the cross-entropy loss between the predicted logits and the target labels which is data type torch.float32.
        input_batch, target_batch = input_batch.to(self.device), target_batch.to( 
            self.device
        )
        logits = self.model(input_batch)
        #It computes the cross-entropy loss between the predicted logits and the target labels. This is a common loss function used in classification tasks, including language modeling. 
        return torch.nn.functional.cross_entropy( 
            logits.flatten(0, 1), target_batch.flatten()
        )

    def calc_loss_loader(self, data_loader, num_batches=None):
        total_loss = 0.0
        num_batches = num_batches or len(data_loader)
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break
            total_loss += self.calc_loss_batch(input_batch, target_batch).item()
        return total_loss / num_batches

    def evaluate_model(self, eval_iter):
        self.model.eval()
        with torch.no_grad():
            train_loss = self.calc_loss_loader(self.train_loader, num_batches=eval_iter)
            val_loss = self.calc_loss_loader(self.val_loader, num_batches=eval_iter)
        self.model.train()
        return train_loss, val_loss

    def generate_and_print_sample(self, start_context):
        self.model.eval()
        context_size = self.model.pos_emb.weight.shape[0]
        encoded = self.text_to_token_ids(start_context).to(self.device)
        with torch.no_grad():
            token_ids = self.generate(
                idx=encoded,
                max_new_tokens=50,
                context_size=context_size,
            )
            print(self.token_ids_to_text(token_ids).replace("\n", " "))
        self.model.train()

    def train_model(self, num_epochs=300, eval_freq=5, eval_iter=5, start_context="Every effort moves you"):
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, -1

        for epoch in range(num_epochs):
            self.model.train()
            for input_batch, target_batch in self.train_loader:
                self.optimizer.zero_grad()
                loss = self.calc_loss_batch(input_batch, target_batch)
                loss.backward()
                self.optimizer.step()
                tokens_seen += input_batch.numel()
                global_step += 1

                if global_step % eval_freq == 0:
                    train_loss, val_loss = self.evaluate_model(eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(
                        f"Ep {epoch+1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                    )

            self.generate_and_print_sample(start_context)

        return train_losses, val_losses, track_tokens_seen

    def generate(self, idx, max_new_tokens, context_size):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = self.model(idx_cond)
            logits = logits[:, -1, :]
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)
        return idx


if __name__ == "__main__":
    config = {
        "vocab_size": 50257,    # GPT-2 vocabulary size
        "context_length": 1024, #Context length (aka context window) is the maximum number of tokens an LLM can “see” and attend to in one forward pass
        "emb_dim": 768,     # Embedding dimension is the size of the vector representation for each token in the vocabulary.
        "n_heads": 1,      # Number of attention heads in the multi-head attention mechanism.
        "n_layers": 12,     # Number of transformer layers in the model.
        "drop_rate": 0.1,   # Dropout rate is the probability of dropping out a neuron during training to prevent overfitting.
        "qkv_bias": False,
        "text_file_path": "the-verdict.txt",
        "url": "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt",
    }

    trainer = GPTTrainer(config)
    trainer.load_data()
    train_losses, val_losses, tokens_seen = trainer.train_model(num_epochs=50)
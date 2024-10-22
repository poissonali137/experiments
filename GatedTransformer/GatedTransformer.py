import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset

class GatingMechanism(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.alphas = nn.Parameter(torch.randn(num_layers))
        self.betas = nn.Parameter(torch.randn(num_layers))

    def forward(self, t, layer_idx):
        return torch.cos(self.alphas[layer_idx] * t + self.betas[layer_idx])


class ContentDependentGating(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gating_proj = nn.Linear(hidden_size, 1)
        self.alpha = nn.Parameter(torch.randn(1))
        self.beta = nn.Parameter(torch.randn(1))

    def forward(self, x):
        # x shape: (seq_len, batch_size, hidden_size)
        proj = self.gating_proj(x)  # (seq_len, batch_size, 1)
        gate = torch.cos(self.alpha * proj + self.beta)
        return gate.squeeze(-1).t()  # (batch_size, seq_len)


class PhaseControlledTransformerLayer(nn.Module):
    def __init__(self, gpt2_layer):
        super().__init__()
        self.attn = gpt2_layer.attn
        self.mlp = gpt2_layer.mlp
        self.ln_1 = gpt2_layer.ln_1
        self.ln_2 = gpt2_layer.ln_2

    def forward(self, x, attention_mask=None, gating_value=None):
        attention_mask = attention_mask.to(dtype=x.dtype)  # Convert mask to float
        attention_mask = (1.0 - attention_mask) * torch.finfo(x.dtype).min
        
        attn_output = self.attn(self.ln_1(x), attention_mask=attention_mask[None, None, :, :])[0]
        x = x + attn_output
        ff_output = self.mlp(self.ln_2(x))
        x = x + ff_output
        
        if gating_value is not None:
            x = x * gating_value.unsqueeze(-1)
        
        return x
        
class PhaseControlledTransformer(nn.Module):
    def __init__(self, gpt2_model):
        super().__init__()
        self.wte = gpt2_model.transformer.wte
        self.wpe = gpt2_model.transformer.wpe
        self.drop = gpt2_model.transformer.drop
        self.ln_f = gpt2_model.transformer.ln_f
        
        self.layers = nn.ModuleList([PhaseControlledTransformerLayer(layer) for layer in gpt2_model.transformer.h])
        self.gating_mechanism = ContentDependentGating(gpt2_model.config.hidden_size)#GatingMechanism(len(self.layers))

    def forward(self, input_ids, attention_mask=None):
        device = input_ids.device
        input_shape = input_ids.size()
        position_ids = torch.arange(0, input_shape[1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[1])
        
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        
        t = torch.linspace(0, 1, hidden_states.size(1), device=device)
        
        for i, layer in enumerate(self.layers):
            gating_value = self.gating_mechanism(t, i)
            hidden_states = layer(hidden_states, attention_mask=attention_mask, gating_value=gating_value)
        
        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class LanguageModel(nn.Module):
    def __init__(self, gpt2_model):
        super().__init__()
        self.transformer = PhaseControlledTransformer(gpt2_model)
        self.lm_head = gpt2_model.lm_head

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.transformer(input_ids, attention_mask=attention_mask)
        lm_logits = self.lm_head(hidden_states)
        return lm_logits
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Create our phase-controlled model
model = LanguageModel(gpt2_model)

# Print the number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Number of parameters: {count_parameters(model):,}")

# Load dataset
dataset = load_dataset("tiny_shakespeare")
train_data = dataset["train"]

# Data preparation function
def prepare_data(dataset, tokenizer, max_length=1024, stride=512):
    all_input_ids = []
    all_attention_masks = []

    for item in dataset:
        text = item['text']
        encodings = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
        
        input_ids = encodings.input_ids[0]
        attention_mask = encodings.attention_mask[0]
        
        # Create overlapping sequences
        for i in range(0, input_ids.size(0) - max_length + 1, stride):
            all_input_ids.append(input_ids[i:i+max_length])
            all_attention_masks.append(attention_mask[i:i+max_length])
    
    return torch.stack(all_input_ids), torch.stack(all_attention_masks)

# Prepare dataset
input_ids, attention_masks = prepare_data(train_data, tokenizer)

# Create DataLoader
dataset = TensorDataset(input_ids, attention_masks)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Optimizer and learning rate scheduler
# Separate optimizer for alpha parameters
gating_params = list(model.transformer.gating_mechanism.parameters())
main_params = [p for n, p in model.named_parameters() if not any(gp is p for gp in gating_params)]

optimizer = optim.AdamW([
    {'params': main_params},
    {'params': gating_params, 'lr': 1e-3}  # Higher learning rate for gating parameters
], lr=5e-5)

# Learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader) * 1000)  # Adjusted for 1000 epochs

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Set up TensorBoard
log_dir = os.path.join("runs", "pt_gating_cos0")
writer = SummaryWriter(log_dir)

# Finetuning loop
num_epochs = 1000  # Set to 1000 epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

global_step = 0
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        input_ids, attention_mask = [t.to(device) for t in batch]
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask=attention_mask)
        
        # Shift the targets for language modeling
        targets = input_ids[:, 1:].contiguous()
        outputs = outputs[:, :-1, :].contiguous()
        
        loss = loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Log metrics
        writer.add_scalar('Loss/train', loss.item(), global_step)
        writer.add_scalar('Perplexity/train', math.exp(loss.item()), global_step)
        
        # Log gating parameters
        for i in range(len(model.transformer.layers)):
            writer.add_scalar(f'Alpha/layer_{i}', model.transformer.gating_mechanism.alphas[i].item(), global_step)
            writer.add_scalar(f'Beta/layer_{i}', model.transformer.gating_mechanism.betas[i].item(), global_step)
        
        # Log gating values
        t = torch.linspace(0, 1, input_ids.size(1), device=device)
        for i in range(len(model.transformer.layers)):
            gating_value = model.transformer.gating_mechanism(t, i)
            writer.add_histogram(f'GatingValue/layer_{i}', gating_value, global_step)
        
        global_step += 1
    
    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
    
    scheduler.step()

writer.close()

# Generate sample text after finetuning
def generate_text(model, prompt, max_length=50, temperature=1.0):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids, attention_mask=attention_mask)
            next_token_logits = outputs[:, -1, :] / temperature
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(input_ids[0])

sample_prompt = "First Citizen:"
generated_text = generate_text(model, sample_prompt)
print(f"Generated text: {generated_text}")

torch.save(model.state_dict(), "models/pt_adjusted.pt")

print(f"TensorBoard logs saved to: {log_dir}")
print("To view the logs, run:")
print(f"tensorboard --logdir={log_dir}")

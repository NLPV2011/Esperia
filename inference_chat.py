import torch
from esperia.model import TransformerModel
from esperia.model_config import Config
from minbpe import BasicTokenizer
from dataclasses import dataclass

tokenizer = BasicTokenizer()
tokenizer.load('./tokenizer/basic.model')
### --- CONFIG --- ###
block_size = 2048 # context-length
batch_size = 64 # mini-batch size
vocab_size = 512
train_size = 0.8 
n_embed = 384
n_heads = 12
head_size = n_embed // n_heads # computes to 384/12=32
n_layers = 4
train_iters = 5000 # no. of batches to train on
save_interval = 100
val_iters = 50 # no. of batches to validate on every eval_intervals
eval_interval = 50 # validate after every eval_interval iterations while training
lr = 6e-4 # also used by the GPT 3 Small, quite a lot more stable than 1e-3
attn_dropout = 0.1
block_dropout = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}
print("TRANSFORMER CONFIG:")
print(config)
### --- CONFIG --- ###

lm = TransformerModel(Config).to(device) # Ensure the model is on the correct device
state_dict = torch.load("EsperiaTransformer/EsperiaTransformer.pth", map_location=device) # Ensure the state dict is loaded on the correct device
lm.load_state_dict(state_dict["model_state_dict"])

history = ''

while True:
    start = input("U >: ") + '|'

    start_ids = tokenizer.encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    generated_texts = []
    for length in [100]:
        generated = lm.generate(
        idx=x,
        max_new_tokens=length
    )
        generated = tokenizer.decode(generated[0].cpu().numpy())
        text=f'generated ({length} tokens)\n{"="*50}\n{generated}\n{"="*50}\n\n'
        generated_texts.append(text)

    answers = ""
    
    with open('generated.txt','w', encoding="utf-8") as f:
        for text in generated_texts:
            answers = text.split("|")[1].split("\n")[0]
            print("A >: " + answers)
    
    history = start + answers + '\n'   
    
    
    

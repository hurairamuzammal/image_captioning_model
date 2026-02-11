
# --- Export Code for Streamlit ---
import json

# 1. Define paths
ENC_PATH = "encoder.pth"
DEC_PATH = "decoder.pth"
VOCAB_PATH = "vocab.pkl"
CONFIG_PATH = "config.json"

# 2. Save Models
torch.save(encoder.state_dict(), ENC_PATH)
torch.save(decoder.state_dict(), DEC_PATH)
print(f"Saved encoder to {ENC_PATH} and decoder to {DEC_PATH}")

# 3. Save Vocab
with open(VOCAB_PATH, 'wb') as f:
    pickle.dump(vocab, f)
print(f"Saved vocab to {VOCAB_PATH}")

# 4. Save Config
config = {
    'embed_dim': 512,
    'hidden_dim': 512,
    'vocab_size': len(vocab)
}
with open(CONFIG_PATH, 'w') as f:
    json.dump(config, f)
print(f"Saved config to {CONFIG_PATH}")

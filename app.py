import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle
import json
import os

# --- Model Classes (Directly from Notebook) ---

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2048, 512)

    def forward(self, x):
        return torch.tanh(self.fc(x))

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, hidden_dim=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions):
        emb = self.embed(captions[:, :-1])
        x = torch.cat([features.unsqueeze(1), emb], dim=1)
        out, _ = self.lstm(x)
        return self.fc(out)

    def greedy_search(self, features, vocab, max_len=20, repetition_penalty=1.0):
        device = features.device
        batch = features.size(0)
        caps = torch.full((batch, 1), vocab['<start>'], device=device, dtype=torch.long)

        out, state = self.lstm(features.unsqueeze(1))

        for _ in range(max_len):
            emb = self.embed(caps[:, -1:])
            out, state = self.lstm(emb, state)
            logits = self.fc(out[:, -1, :])


            if repetition_penalty != 1.0:
                for i in range(batch):
                    for prev_token in caps[i]:
                         if prev_token.item() == vocab.get('<pad>', 0): continue
                         if logits[i, prev_token] < 0:
                            logits[i, prev_token] *= repetition_penalty
                         else:
                            logits[i, prev_token] /= repetition_penalty

            pred = torch.argmax(logits, dim=1)
            caps = torch.cat([caps, pred.unsqueeze(1)], dim=1)
            if (pred == vocab['<end>']).all():
                break
        return caps

    def beam_search(self, feature, vocab, beam=3, max_len=20):
        device = feature.device
        # Start token
        seqs = [[ [vocab['<start>']], 0.0 ]]

        for _ in range(max_len):
            all_cands = []
            for seq, score in seqs:
                if seq[-1] == vocab['<end>']:
                    all_cands.append([seq, score])
                    continue

                inp = torch.tensor(seq, device=device).unsqueeze(0)
                emb = self.embed(inp)
                out,_ = self.lstm(emb)
                logits = self.fc(out[:,-1,:])
                logp = torch.log_softmax(logits, dim=1)
                topk = torch.topk(logp, beam)

                for k in range(beam):
                    cand = seq + [topk.indices[0,k].item()]
                    sc = score + topk.values[0,k].item()
                    all_cands.append([cand, sc])

            seqs = sorted(all_cands, key=lambda x: x[1], reverse=True)[:beam]

        return seqs[0][0]

# --- Configuration & Setup ---

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENCODER_PATH = "encoder.pth"
DECODER_PATH = "decoder.pth"
VOCAB_PATH = "vocab.pkl"
CONFIG_PATH = "config.json"

@st.cache_resource
def load_resources():
    # Load Config
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
    else:
        # Fallback config
        config = {
            'embed_dim': 512,
            'hidden_dim': 512, # Note: using hidden_dim as per Decoder __init__
            'vocab_size': None # Will be set from vocab
        }

    # Load Vocabulary
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    
    idx2word = {i: w for w, i in vocab.items()}
    vocab_size = len(vocab)
    config['vocab_size'] = vocab_size

    # Initialize Models
    encoder = Encoder().to(DEVICE)
    decoder = Decoder(
        vocab_size=vocab_size,
        embed_dim=config.get('embed_dim', 512),
        hidden_dim=config.get('hidden_dim', 512)
    ).to(DEVICE)

    # Load Weights
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    decoder.load_state_dict(torch.load(DECODER_PATH, map_location=DEVICE))

    encoder.eval()
    decoder.eval()

    # Feature Extractor (ResNet50)
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet = resnet.to(DEVICE)
    resnet.eval()

    return resnet, encoder, decoder, vocab, idx2word

def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def tokens_to_words(tokens, vocab, idx2word):
    words = []
    for t in tokens:
        if t not in [vocab['<pad>'], vocab['<start>'], vocab['<end>']]:
            words.append(idx2word[t])
    return words

# --- Streamlit UI ---

st.title("Image Captioning App")
st.write("Upload an image to generate a caption.")

# Sidebar Settings
st.sidebar.header("Settings")
scaling_factor = st.sidebar.slider("Repetition Penalty (Greedy Search)", 1.0, 2.0, 1.2, 0.1)
use_beam_search = st.sidebar.checkbox("Use Beam Search", value=False)
beam_width = 3
if use_beam_search:
    beam_width = st.sidebar.slider("Beam Width", 1, 10, 3, 1)

# Load Models
try:
    resnet, encoder, decoder, vocab, idx2word = load_resources()
    st.sidebar.success("Models Loaded!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Image Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Generating..."):
            try:
                img_tensor = process_image(image)
                
                # 1. Extract Features
                with torch.no_grad():
                    features = resnet(img_tensor)
                    features = features.view(features.size(0), -1) # Flatten (1, 2048)
                
                # 2. Encode
                with torch.no_grad():
                    enc_out = encoder(features)
                
                # 3. Decode
                with torch.no_grad():
                    if use_beam_search:
                        # Beam Search (Note: Decoder.beam_search takes feature, not enc_out in current notebook code? 
                        # Wait, notebook code: f = encoder(feat); decoder.beam_search(f[0:1], vocab)
                        # So it takes encoded features.
                        
                        # In Notebook:
                        # feat = torch.tensor(features[name]).unsqueeze(0).to(device)
                        # f = encoder(feat)
                        # greedy_tokens = decoder.greedy_search(f, vocab)
                        
                        # So 'features' arg in decoder methods is actually 'encoded_features'
                        
                        tokens = decoder.beam_search(enc_out, vocab, beam=beam_width)
                        # Beam search returns a list of token ids
                    else:
                        # Greedy Search
                        # Returns tensor of shape (batch, seq_len)
                        tokens_tensor = decoder.greedy_search(enc_out, vocab, repetition_penalty=scaling_factor)
                        tokens = tokens_tensor[0].cpu().tolist()

                # 4. Convert to Words
                caption = " ".join(tokens_to_words(tokens, vocab, idx2word))
                
                st.subheader("Caption:")
                st.write(caption)

            except Exception as e:
                st.error(f"An error occurred: {e}")

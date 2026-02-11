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
        batch = feature.size(0)
        # 1. Initialize with Image Feature
        # Note: feature shape should be (1, dim)
        out, state = self.lstm(feature.unsqueeze(1))
        
        # seqs = [(sequence, score, state)]
        # sequence is a list of token IDs
        seqs = [[ [vocab['<start>']], 0.0, state ]]

        for _ in range(max_len):
            all_cands = []
            for seq, score, hidden_state in seqs:
                if seq[-1] == vocab['<end>']:
                    all_cands.append([seq, score, hidden_state])
                    continue

                # Prepare input for next step (just the last token)
                # (1, 1)
                inp = torch.tensor([seq[-1]], device=device).unsqueeze(0)
                emb = self.embed(inp)
                
                # Run LSTM for one step
                out, new_state = self.lstm(emb, hidden_state)
                
                # Predict
                logits = self.fc(out[:, -1, :])
                logp = torch.log_softmax(logits, dim=1)
                
                # Get top k
                topk = torch.topk(logp, beam) # (1, beam)

                for k in range(beam):
                    token_idx = topk.indices[0, k].item()
                    token_prob = topk.values[0, k].item()
                    
                    cand_seq = seq + [token_idx]
                    cand_score = score + token_prob
                    all_cands.append([cand_seq, cand_score, new_state])

            # Select top beam candidates
            seqs = sorted(all_cands, key=lambda x: x[1], reverse=True)[:beam]
            
            # Check if all top candidates have ended (optimization)
            if all(s[0][-1] == vocab['<end>'] for s in seqs):
                break

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

# Image Source Selection
st.write("### Select an Image Source")
option = st.radio("Choose source:", ("Upload Image", "Sample Images"))

image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
else:
    # Sample Images
    sample_dir = "samples"
    if os.path.exists(sample_dir):
        sample_files = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if sample_files:
            # Display samples in a grid
            cols = st.columns(len(sample_files))
            for i, file_name in enumerate(sample_files):
                img_path = os.path.join(sample_dir, file_name)
                img = Image.open(img_path).convert("RGB")
                with cols[i]:
                    st.image(img, use_container_width=True)
                    if st.button(f"Select Sample {i+1}", key=f"sample_{i}"):
                        st.session_state['selected_sample'] = img_path
            
            # Load selected sample if exists in session state
            if 'selected_sample' in st.session_state:
                image = Image.open(st.session_state['selected_sample']).convert("RGB")
                st.info(f"Selected: {os.path.basename(st.session_state['selected_sample'])}")
        else:
            st.warning("No sample images found in 'samples' directory.")
    else:
        st.warning("'samples' directory not found.")

if image:
    st.image(image, caption='Selected Image', use_container_width=True)

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
                        tokens = decoder.beam_search(enc_out, vocab, beam=beam_width)
                    else:
                        tokens_tensor = decoder.greedy_search(enc_out, vocab, repetition_penalty=scaling_factor)
                        tokens = tokens_tensor[0].cpu().tolist()

                # 4. Convert to Words
                caption = " ".join(tokens_to_words(tokens, vocab, idx2word))
                
                st.subheader("Caption:")
                st.write(caption)

            except Exception as e:
                st.error(f"An error occurred: {e}")

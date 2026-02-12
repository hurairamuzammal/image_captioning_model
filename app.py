import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle
import json
import os

# --- NEW Model Classes (Init-Injection Architecture) ---

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.5):
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.h_init = nn.Linear(hidden_dim, hidden_dim * num_layers)
        self.c_init = nn.Linear(hidden_dim, hidden_dim * num_layers)

    def forward(self, images):
        features = self.feature_proj(images)
        h0 = self.h_init(features).view(-1, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c0 = self.c_init(features).view(-1, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        return h0, c0

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=0.5):
        super(LSTMDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, captions, h0, c0):
        embeds = self.dropout(self.embedding(captions))
        lstm_out, _ = self.lstm(embeds, (h0, c0))
        outputs = self.linear(lstm_out)
        return outputs

class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, images, captions):
        h0, c0 = self.encoder(images)
        captions_input = captions[:, :-1]
        outputs = self.decoder(captions_input, h0, c0)
        return outputs

# --- Configuration & Setup ---

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "Model/model.pth"
VOCAB_PATH = "Model/vocab.pkl"
CONFIG_PATH = "Model/config.json"

@st.cache_resource
def load_resources():
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    with open(VOCAB_PATH, 'rb') as f:
        vocab_data = pickle.load(f)

    stoi = vocab_data['stoi']
    itos = vocab_data['itos']
    if isinstance(itos, dict):
        itos = {int(k): v for k, v in itos.items()}
    vocab_size = config['vocab_size']

    encoder = LSTMEncoder(
        config['image_feature_dim'],
        config['hidden_dim'],
        config['encoder_layers'],
        config['dropout']
    )
    decoder = LSTMDecoder(
        vocab_size,
        config['embedding_dim'],
        config['hidden_dim'],
        config['decoder_layers'],
        config['dropout']
    )
    model = ImageCaptioningModel(encoder, decoder, DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet = resnet.to(DEVICE)
    resnet.eval()

    return resnet, model, stoi, itos

def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def generate_caption_greedy(model, image_feature, stoi, itos, max_len=30):
    model.eval()
    with torch.no_grad():
        h0, c0 = model.encoder(image_feature)
        inputs = torch.tensor([stoi["<START>"]]).unsqueeze(0).to(DEVICE)
        caption = []
        for _ in range(max_len):
            embeds = model.decoder.embedding(inputs)
            lstm_out, (h0, c0) = model.decoder.lstm(embeds, (h0, c0))
            outputs = model.decoder.linear(lstm_out.squeeze(1))
            unk_idx = stoi.get("<UNK>", -1)
            if 0 <= unk_idx < outputs.shape[1]:
                outputs[:, unk_idx] = float('-inf')
            predicted = outputs.argmax(1)
            if predicted.item() == stoi["<END>"]:
                break
            caption.append(itos[predicted.item()])
            inputs = predicted.unsqueeze(0)
    return " ".join(caption)

def generate_caption_beam(model, image_feature, stoi, itos, beam_width=3, max_len=30):
    model.eval()
    with torch.no_grad():
        h0, c0 = model.encoder(image_feature)
        seqs = [[[stoi["<START>"]], 0.0, (h0, c0)]]
        for _ in range(max_len):
            all_cands = []
            for seq, score, state in seqs:
                if seq[-1] == stoi["<END>"]:
                    all_cands.append([seq, score, state])
                    continue
                inp = torch.tensor([seq[-1]]).unsqueeze(0).to(DEVICE)
                embeds = model.decoder.embedding(inp)
                lstm_out, new_state = model.decoder.lstm(embeds, state)
                logits = model.decoder.linear(lstm_out[:, -1, :])
                unk_idx = stoi.get("<UNK>", -1)
                if 0 <= unk_idx < logits.shape[1]:
                    logits[:, unk_idx] = float('-inf')
                logp = torch.log_softmax(logits, dim=1)
                topk = torch.topk(logp, beam_width)
                for k in range(beam_width):
                    cand = seq + [topk.indices[0, k].item()]
                    sc = score + topk.values[0, k].item()
                    all_cands.append([cand, sc, new_state])
            seqs = sorted(all_cands, key=lambda x: x[1], reverse=True)[:beam_width]
            if all(s[0][-1] == stoi["<END>"] for s in seqs):
                break
        best = seqs[0][0]
        words = [itos[i] for i in best if i not in [stoi["<START>"], stoi["<END>"], stoi["<PAD>"]]]
    return " ".join(words)

# --- Streamlit UI ---

st.title("Image Captioning App")
st.write("Upload an image to generate a caption.")

st.sidebar.header("Settings")
use_beam_search = st.sidebar.checkbox("Use Beam Search", value=True)
beam_width = 3
if use_beam_search:
    beam_width = st.sidebar.slider("Beam Width", 1, 10, 3, 1)

try:
    resnet, model, stoi, itos = load_resources()
    st.sidebar.success("Models Loaded!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

st.write("### Select an Image Source")
option = st.radio("Choose source:", ("Upload Image", "Sample Images"))

image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
else:
    sample_dir = "samples"
    if os.path.exists(sample_dir):
        sample_files = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if sample_files:
            cols = st.columns(len(sample_files))
            for i, file_name in enumerate(sample_files):
                img_path = os.path.join(sample_dir, file_name)
                img = Image.open(img_path).convert("RGB")
                with cols[i]:
                    st.image(img, use_container_width=True)
                    if st.button(f"Select Sample {i+1}", key=f"sample_{i}"):
                        st.session_state['selected_sample'] = img_path
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

                with torch.no_grad():
                    features = resnet(img_tensor)
                    features = features.view(features.size(0), -1)

                if use_beam_search:
                    caption = generate_caption_beam(model, features, stoi, itos, beam_width=beam_width)
                else:
                    caption = generate_caption_greedy(model, features, stoi, itos)

                st.subheader("Caption:")
                st.write(caption)

            except Exception as e:
                st.error(f"An error occurred: {e}")

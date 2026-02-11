import torch
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter
import numpy as np
from tqdm import tqdm


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
                         if logits[i, prev_token] < 0: logits[i, prev_token] *= repetition_penalty
                         else: logits[i, prev_token] /= repetition_penalty

            pred = torch.argmax(logits, dim=1)
            caps = torch.cat([caps, pred.unsqueeze(1)], dim=1)
            if (pred == vocab['<end>']).all(): break
        return caps

    def beam_search(self, feature, vocab, beam=3, max_len=20):
        device = feature.device
        out, state = self.lstm(feature.unsqueeze(1))
        seqs = [[ [vocab['<start>']], 0.0, state ]]

        for _ in range(max_len):
            all_cands = []
            for seq, score, hidden_state in seqs:
                if seq[-1] == vocab['<end>']:
                    all_cands.append([seq, score, hidden_state])
                    continue

                inp = torch.tensor([seq[-1]], device=device).unsqueeze(0)
                emb = self.embed(inp)
                out, new_state = self.lstm(emb, hidden_state)
                
                logits = self.fc(out[:, -1, :])
                logp = torch.log_softmax(logits, dim=1)
                topk = torch.topk(logp, beam)

                for k in range(beam):
                    cand = seq + [topk.indices[0,k].item()]
                    sc = score + topk.values[0,k].item()
                    all_cands.append([cand, sc, new_state])

            seqs = sorted(all_cands, key=lambda x: x[1], reverse=True)[:beam]
            if all(s[0][-1] == vocab['<end>'] for s in seqs): break

        return seqs[0][0]

decoder = Decoder(vocab_size=vocab_size, embed_dim=512, hidden_dim=512).to(device)
# RELOAD weights if you trained it (assuming 'decoder' variable held trained weights, or use path)
# If you just trained 'decoder', the weights are lost unless saved. 
# Assuming you want to EVALUATE the loaded/trained model, you must use load_state_dict if it's a new instance.
# IMPORTANT: If 'decoder' already has weights, save them first!
# torch.save(decoder.state_dict(), 'temp_decoder.pth')
# decoder.load_state_dict(torch.load('temp_decoder.pth')) 


# --- 3. Evaluation Function ---

def calculate_precision_recall_f1(reference_tokens, candidate_tokens):
    # Bag of words (1-gram)
    ref_counts = Counter(reference_tokens)
    cand_counts = Counter(candidate_tokens)
    
    overlap = sum((ref_counts & cand_counts).values())
    
    precision = overlap / len(candidate_tokens) if len(candidate_tokens) > 0 else 0
    recall = overlap / sum(ref_counts.values()) if sum(ref_counts.values()) > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def evaluate_model(test_imgs, features_dict, captions_dict, encoder, decoder, vocab, idx2word, beam_size=3):
    bleu4_scores = []
    precisions = []
    recalls = []
    f1s = []
    
    smooth = SmoothingFunction().method1
    
    print(f"Evaluating on {len(test_imgs)} test images using Beam Search (k={beam_size})...")
    
    for name in tqdm(test_imgs):
        # 1. Get Image Features
        feat = torch.tensor(features_dict[name]).unsqueeze(0).to(device)
        
        # 2. Encode
        with torch.no_grad():
            f = encoder(feat)
            
        pred_ids = decoder.beam_search(f[0:1], vocab, beam=beam_size)
        
        pred_words = [idx2word[i] for i in pred_ids if i not in [vocab['<start>'], vocab['<end>'], vocab['<pad>']]]
        
        references = [cap.lower().split() for cap in captions_dict[name]]
        
        score = sentence_bleu(references, pred_words, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
        bleu4_scores.append(score)
        
        local_p, local_r, local_f1 = 0, 0, 0
        for ref in references:
            p, r, f = calculate_precision_recall_f1(ref, pred_words)
            if f > local_f1:
                local_p, local_r, local_f1 = p, r, f
        
        precisions.append(local_p)
        recalls.append(local_r)
        f1s.append(local_f1)
        
    print("\n--- Evaluation Results ---")
    print(f"BLEU-4: {np.mean(bleu4_scores):.4f}")
    print(f"Precision: {np.mean(precisions):.4f}")
    print(f"Recall: {np.mean(recalls):.4f}")
    print(f"F1-Score: {np.mean(f1s):.4f}")
    
    return np.mean(bleu4_scores)

evaluate_model(test_imgs[:50], features, captions_dict, encoder, decoder, vocab, idx2word, beam_size=3)

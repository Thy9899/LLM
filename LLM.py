import torch
import torch.nn as nn
import random
import string
import os
import requests

# --- Configuration ---
TEXT_CORPUS = (
    "Hello! How can I help you today? "
    "Please, go ahead! I'm ready to answer your questions."
    "The quick brown fox jumps over the lazy dog. "
    "A journey of a thousand miles begins with a single step. "
    "To be or not to be, that is the question. "
    "All's well that ends well. "
    "Nature is not a place to visit. It is home."
).lower()

HIDDEN_SIZE = 128
CHUNK_LEN = 20
TEMPERATURE = 0.8

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data Prep ---
all_characters = string.ascii_lowercase + string.punctuation + " "
n_chars = len(all_characters)
char_to_ix = {ch: i for i, ch in enumerate(all_characters)}
ix_to_char = {i: ch for ch, i in char_to_ix.items()}

# --- Model ---
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input_tensor, hidden):
        output, hidden = self.gru(input_tensor, hidden)
        output = self.decoder(output)
        return output, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

# --- Generate Text ---
def generate(model, prime_str="the ", predict_len=100, temperature=TEMPERATURE):
    model.eval()
    hidden = model.init_hidden()
    
    # Feed the prime string (all chars except last)
    for char in prime_str[:-1]:
        if char in char_to_ix:
            one_hot = torch.zeros(1, 1, n_chars, device=device)
            one_hot[0, 0, char_to_ix[char]] = 1
            _, hidden = model(one_hot, hidden)

    current_char_ix = char_to_ix.get(prime_str[-1], 0)
    generated = prime_str

    with torch.no_grad():
        for _ in range(predict_len):
            one_hot = torch.zeros(1, 1, n_chars, device=device)
            one_hot[0, 0, current_char_ix] = 1
            output, hidden = model(one_hot, hidden)
            output_dist = torch.softmax(output.squeeze() / temperature, dim=0)
            current_char_ix = torch.multinomial(output_dist, 1).item()
            generated += ix_to_char[current_char_ix]
    return generated

# --- Load or train model ---
def load_model(model_path="model.pth"):
    model = CharRNN(n_chars, HIDDEN_SIZE, n_chars).to(device)
    if os.path.exists(model_path):
        print("Loading model from disk...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        # If model file is missing, download from a public URL (optional)
        print("Model file missing. Please provide model.pth or implement training.")
    model.eval()
    return model

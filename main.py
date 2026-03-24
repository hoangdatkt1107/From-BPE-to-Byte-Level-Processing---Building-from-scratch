import collections
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# 1. BPE TOKENIZER

class BPETokenizer:
    def __init__(self, num_merges=2000):
        self.num_merges = num_merges
        self.merges = []
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.inverse_vocab = {0: "<PAD>", 1: "<UNK>"}
        self.next_id = 2

    def _get_hex(self, text):
        return [f"{b:02x}" for b in str(text).encode("utf-8")]

    def train(self, corpus):
        print(f"Training Tokenizer ({self.num_merges} merge rules)...")
        bytes_list = self._get_hex(corpus)

        unique_base_bytes = sorted(set(bytes_list))
        for b in unique_base_bytes:
            if b not in self.vocab:
                self.vocab[b] = self.next_id
                self.inverse_vocab[self.next_id] = b
                self.next_id += 1

        for step in range(self.num_merges):
            stats = collections.defaultdict(int)
            for i in range(len(bytes_list) - 1):
                stats[bytes_list[i], bytes_list[i+1]] += 1

            if not stats: break

            best_pair = max(stats, key=stats.get)
            new_token = "".join(best_pair)
            self.merges.append(best_pair)

            self.vocab[new_token] = self.next_id
            self.inverse_vocab[self.next_id] = new_token
            self.next_id += 1

            merged_list = []
            i = 0
            while i < len(bytes_list):
                if i < len(bytes_list) - 1 and bytes_list[i] == best_pair[0] and bytes_list[i+1] == best_pair[1]:
                    merged_list.append(new_token)
                    i += 2
                else:
                    merged_list.append(bytes_list[i])
                    i += 1
            bytes_list = merged_list

            if (step + 1) % 200 == 0:
                print(f"-> Learned {step + 1}/{self.num_merges} rules...")

        print(f"Training complete. Final vocabulary size: {len(self.vocab)}\n")

    def trace_encoding(self, word):
        """Prints the step-by-step encoding process of a word to the Terminal"""
        print("-" * 75)
        print(f"🔍 ALGORITHM TRACE: Tokenizing '{word}'")
        print("-" * 75 + "\n")

        current_seq = self._get_hex(word)

        def to_chars(hex_array):
            return [bytes.fromhex(h).decode('utf-8', errors='replace') for h in hex_array]

        print(f"STEP 0 (Raw Bytes):")
        print(f"[Text]: {to_chars(current_seq)}")
        print(f"[Hex]:  {current_seq}\n")

        step_idx = 1
        for pair in self.merges:
            i = 0
            did_merge = False
            while i < len(current_seq) - 1:
                if current_seq[i] == pair[0] and current_seq[i+1] == pair[1]:
                    current_seq[i] = pair[0] + pair[1]
                    del current_seq[i+1]
                    did_merge = True
                else:
                    i += 1

            if did_merge:
                merged_str = bytes.fromhex(pair[0]+pair[1]).decode('utf-8', errors='replace')
                print(f"STEP {step_idx} (Applied Rule: Merge '{merged_str}'):")
                print(f"[Text]: {to_chars(current_seq)}")
                print(f"[Hex]:  {current_seq}\n")
                step_idx += 1

        final_ids = [self.vocab.get(t, self.vocab["<UNK>"]) for t in current_seq]
        print(f"✅ FINAL NEURAL NET INPUT (IDs): {final_ids}")
        print("-" * 75 + "\n")
        return final_ids

    def encode(self, text):
        seq = self._get_hex(text)
        for pair in self.merges:
            i = 0
            while i < len(seq) - 1:
                if seq[i] == pair[0] and seq[i+1] == pair[1]:
                    seq[i] = pair[0] + pair[1]
                    del seq[i+1]
                else:
                    i += 1
        return [self.vocab.get(t, self.vocab["<UNK>"]) for t in seq]

    def decode(self, ids):
        hex_tokens = [self.inverse_vocab.get(i, "") for i in ids if i > 1]
        try:
            return bytes.fromhex("".join(hex_tokens)).decode("utf-8")
        except:
            return ""

# 2. EXPANDED ALICE IN WONDERLAND DATASET

def load_medium_dataset():
    print("[*] Downloading Dataset (Alice in Wonderland)")
    path = tf.keras.utils.get_file(
        'alice.txt',
        origin='https://www.gutenberg.org/files/11/11-0.txt')
    with open(path, encoding='utf-8') as f:
        text = f.read().lower()

    text = " ".join(text[1500:101500].split())
    return text

# 3. MAIN PIPELINE

if __name__ == "__main__":
    corpus = load_medium_dataset()

    # 1. Train Tokenizer
    tokenizer = BPETokenizer(num_merges=2000)
    tokenizer.train(corpus)

    # VISUALIZATION
    tokenizer.trace_encoding("learning")

    # 2. Data Preparation
    encoded_data = tokenizer.encode(corpus)
    SEQ_LEN = 20 

    X, y = [], []
    for i in range(len(encoded_data) - SEQ_LEN):
        X.append(encoded_data[i:i+SEQ_LEN])
        y.append(encoded_data[i+SEQ_LEN])

    X = np.array(X)
    y = np.array(y)

    # 3. Build a LSTM Network
    vocab_size = len(tokenizer.vocab)
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=SEQ_LEN),
        LSTM(256, return_sequences=True),
        Dropout(0.2),
        LSTM(256),
        Dense(vocab_size, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 4. Train Model
    model.fit(X, y, epochs=70, batch_size=128, verbose=1)

    # 5. Inference / Text Generation
    print(" PREDICTING THE NEXT SENTENCE (INFERENCE) ")
    seed = "the white rabbit "
    print(f"Input Seed: '{seed}'")

    curr_seq = tokenizer.encode(seed)
    predicted_ids = []

    # Predict the next 15 tokens
    for _ in range(15):
        in_seq = curr_seq[-SEQ_LEN:]
        if len(in_seq) < SEQ_LEN:
            in_seq = [0] * (SEQ_LEN - len(in_seq)) + in_seq

        pred_probs = model(np.array([in_seq]), training=False).numpy()[0]
        pred_id = np.argmax(pred_probs)

        predicted_ids.append(pred_id)
        curr_seq.append(pred_id)

    out_text = tokenizer.decode(predicted_ids)

    print(f"AI generated continuation: '{out_text}'")
    print(f"Full sentence: '{seed}{out_text}'\n")
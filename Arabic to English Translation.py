import numpy as np
import tensorflow as tf
from tensorflow import keras

# Hyperparameters
batch_size = 64  # Batch size for training.
epochs = 50  # Increased number of epochs.
latent_dim = 512  # Increased latent dimensionality.
dropout_rate = 0.2  # Dropout for regularization.
num_samples = 10000  # Number of samples to train on.

# Function to read and clean data
def clean_data(file_path, sample_size=10000):
    input_text = []
    target_text = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    
    for line in lines[:sample_size]:
        parts = line.split("\t")  # Split by tab
        if len(parts) >= 2:  # Ensure there are at least two parts (English & Arabic)
            input_text.append(parts[0].strip())  # English sentence
            target_text.append('\t' + parts[1].strip() + '\n')  # Arabic sentence with start/end tokens
        else:
            print(f"Skipping malformed line: {line}")
    
    return input_text, target_text

# Function to print sample data
def print_sample_data(input_text, target_text, num_samples=5):
    print("\nSample cleaned data:")
    for i in range(min(num_samples, len(input_text))):
        print(f"English: {input_text[i]}")
        print(f"Arabic: {target_text[i]}")
        print("-")

# Load and clean data
input_texts, target_texts = clean_data("/content/ara.txt", num_samples)
print_sample_data(input_texts, target_texts)  # Print sample cleaned data

# Create sets of unique characters
input_characters = set()
target_characters = set()
for input_text in input_texts:
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
for target_text in target_texts:
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print("Number of samples:", len(input_texts))
print("Number of unique input tokens:", num_encoder_tokens)
print("Number of unique output tokens:", num_decoder_tokens)
print("Max sequence length for inputs:", max_encoder_seq_length)
print("Max sequence length for outputs:", max_decoder_seq_length)

# Create token indices
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

# Vectorize the data
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype="float32",
)
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype="float32",
)
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype="float32",
)

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
    decoder_target_data[i, t:, target_token_index[" "]] = 1.0

# Define the model with dropout
encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
encoder = keras.layers.LSTM(latent_dim, return_state=True, dropout=dropout_rate)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))
decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True, dropout=dropout_rate)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model with a lower learning rate
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Add early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

# Train the model
history = model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    callbacks=[early_stopping],
)

# Save the model
model.save("s2s_model_improved.keras")

# Load the model for inference
model = keras.models.load_model("s2s_model_improved.keras")

# Define the encoder model
encoder_inputs = model.input[0]
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output
encoder_states = [state_h_enc, state_c_enc]
encoder_model = keras.Model(encoder_inputs, encoder_states)

# Define the decoder model
decoder_inputs = model.input[1]
decoder_state_input_h = keras.Input(shape=(latent_dim,))
decoder_state_input_c = keras.Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = keras.Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)

# Reverse-lookup token indices
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

# Beam search decoding
def beam_search_decode(input_seq, beam_width=3):
    states_value = encoder_model.predict(input_seq, verbose=0)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index["\t"]] = 1.0

    beam = [([], 0, states_value)]  # (sequence, score, states)
    final_candidates = []

    for _ in range(max_decoder_seq_length):
        new_beam = []
        for seq, score, states in beam:
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states, verbose=0
            )
            top_indices = np.argsort(output_tokens[0, -1, :])[-beam_width:]
            for idx in top_indices:
                new_seq = seq + [idx]
                new_score = score + np.log(output_tokens[0, -1, idx])
                new_states = [h, c]
                new_beam.append((new_seq, new_score, new_states))

        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_width]

    # Select the best sequence
    best_seq = beam[0][0]
    decoded_sentence = "".join([reverse_target_char_index[idx] for idx in best_seq])
    return decoded_sentence

# Test the model on some samples
for seq_index in range(20):
    input_seq = encoder_input_data[seq_index : seq_index + 1]
    decoded_sentence = beam_search_decode(input_seq, beam_width=3)
    print("-")
    print("Input sentence:", input_texts[seq_index])
    print("Decoded sentence:", decoded_sentence)
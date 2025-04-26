import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from datasets import load_dataset
from sklearn.model_selection import train_test_split
"""
### NEW
"""
from PIL import Image

# setting parameters
IMG_WIDTH, IMG_HEIGHT = 128, 32
MAX_LABEL_LENGTH = 110

# loading in the dataset IAM
dataset = load_dataset("Teklia/IAM-line")
sample = dataset['train'][0]
print("IAM Sample text:", sample['text'])
sample['image'].show()

# function to preprocess IAM
def preprocess_image_pil(image):
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image = np.array(image.convert("L")) / 255.0  
    return np.expand_dims(image, axis=-1)

iam_texts = [ex['text'] for ex in dataset['train']]

# the tokenizer for IAM
# Note: Tokenizer assigns tokens starting at 1, and 0 will be used for padding.
tokenizer = Tokenizer(char_level=True, lower=False, filters='')
tokenizer.fit_on_texts(iam_texts)

M = max(tokenizer.word_index.values())
# We remap valid tokens to range 0 ... M-1. Then we reserve index M as the blank.
NUM_CLASSES = M + 1
print("Computed NUM_CLASSES (including blank classes):", NUM_CLASSES)

# processing the images and labels for IAM dataset
train_images = np.array([preprocess_image_pil(ex['image']) for ex in dataset['train']])

# Get label sequences (values are in [1, M])
train_labels = tokenizer.texts_to_sequences(iam_texts)

# Pad with 0 (which is also our pad value)
train_labels = pad_sequences(train_labels, maxlen=MAX_LABEL_LENGTH, padding='post', value=0)

# Remap: subtract 1 from nonzero entries so that valid tokens become 0..M-1.
# (Padded zeros remain 0 – later we compute label lengths by counting nonzeros.)
train_labels = np.where(train_labels != 0, train_labels - 1, 0)

print("IAM images shape:", train_images.shape)
print("IAM labels shape:", train_labels.shape)

###jpg folder and txt. file provided
samples_image_folder = "jpg"  
samples_label_file = "/Users/reionishi/Desktop/SamplePassage.txt" ###change file directory accordingly

# reading the labels line by line
with open(samples_label_file, "r", encoding="utf-8") as f:
    custom_labels = [line.strip() for line in f.readlines()]

# list and sort collected data image paths 
image_paths = sorted([
    os.path.join(samples_image_folder, f)
    for f in os.listdir(samples_image_folder)
    if f.lower().endswith(".jpg")
])
print("Custom image count:", len(image_paths))
print("Custom label count:", len(custom_labels))

"""
##SHIKHU
def preprocess_image_from_path(path):
    img = load_img(path, color_mode='grayscale', target_size=(IMG_HEIGHT, IMG_WIDTH))
    img = img_to_array(img) / 255.0
    return img
"""
"""
##new REI - functions for resizing/resolution
"""
def resize_with_padding(img, target_width=128, target_height=32, upscale=True):
    img = img.convert("L")  # grayscale
    old_width, old_height = img.size
    scale = min(target_width / old_width, target_height / old_height)

    if not upscale:
        scale = min(scale, 1.0)  # don't upscale if flag is False

    new_width = int(old_width * scale)
    new_height = int(old_height * scale)

    # Resize using a high-quality resampling filter
    img = img.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)

    # Create a new white canvas and paste the resized image at the center
    new_img = Image.new("L", (target_width, target_height), color=255)
    upper_left = ((target_width - new_width) // 2, (target_height - new_height) // 2)
    new_img.paste(img, upper_left)

    # Normalize pixel values
    return np.expand_dims(np.array(new_img) / 255.0, axis=-1)


def preprocess_image_from_path(path):
    img = Image.open(path)
    return resize_with_padding(img)

"""
##end new REI
"""

custom_images_processed = np.array([preprocess_image_from_path(p) for p in image_paths])
custom_sequences = tokenizer.texts_to_sequences(custom_labels)
custom_sequences_padded = pad_sequences(custom_sequences, maxlen=MAX_LABEL_LENGTH, padding='post', value=0)
custom_sequences_padded = np.where(custom_sequences_padded != 0, custom_sequences_padded - 1, 0)

print("Custom images shape:", custom_images_processed.shape)
print("Custom labels shape:", custom_sequences_padded.shape)

# merging the IAM and the custom data
merged_images = np.concatenate([train_images, custom_images_processed], axis=0)
merged_labels = np.concatenate([train_labels, custom_sequences_padded], axis=0)
print("Merged images shape:", merged_images.shape)
print("Merged labels shape:", merged_labels.shape)

# split the data into training and testing
X_train, X_val, y_train, y_val = train_test_split(
    merged_images, merged_labels, test_size=0.2, random_state=42, shuffle=True
)
print("Training set:", X_train.shape, y_train.shape)
print("Validation set:", X_val.shape, y_val.shape)

#first run through of training merged data CELL BLOCK 
# model inputs
input_img = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name='image_input')
labels_input = layers.Input(name='label', shape=(MAX_LABEL_LENGTH,), dtype='int32')

# CNN + BiLSTM Architecture
x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(input_img)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Reshape((-1, 128))(x)
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.Dense(NUM_CLASSES, activation='softmax')(x)

# CTC loss using a lambda layer
def ctc_lambda_func(args):
    y_pred, labels = args
    # Prediction lengths: each sample has the same time dimension.
    time_steps = tf.cast(tf.shape(y_pred)[1], tf.float32)
    input_length = tf.ones((tf.shape(y_pred)[0], 1), dtype=tf.float32) * time_steps
    # Actual label length: count nonzero (i.e. valid) labels per sample.
    label_length = tf.cast(tf.math.count_nonzero(labels, axis=1, keepdims=True), dtype=tf.float32)
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

loss_out = layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc_loss')([x, labels_input])

# final model
ctc_model = Model(inputs=[input_img, labels_input], outputs=loss_out, name="CTC_Model")
ctc_model.compile(optimizer='adam', loss=lambda y_true, y_pred: y_pred)

dummy_train = np.zeros((len(X_train), 1))
dummy_val = np.zeros((len(X_val), 1))

# model training
history = ctc_model.fit(
    [X_train, y_train],
    dummy_train,
    validation_data=([X_val, y_val], dummy_val),
    epochs=20,
    batch_size=32
)

# a separate model that maps the image input to predictions (the softmax outputs)

# Get the image input from our original model
# (Note: our original model 'ctc_model' uses two inputs, but the prediction is based solely on the image input.)
prediction_model = Model(inputs=input_img, outputs=x)


# decode predictions using TensorFlow their built in CTC decoder
def decode_predictions(preds, charset):
    # preds: output from the prediction model, shape (batch, time_steps, NUM_CLASSES)
    # The 'input_length' for each sample is the full length of the time dimension:
    input_length = np.ones(preds.shape[0]) * preds.shape[1]
    # Use greedy CTC decode:
    decoded, log_prob = tf.keras.backend.ctc_decode(preds, input_length, greedy=True)
    decoded_sequences = decoded[0].numpy()
    results = []
    for seq in decoded_sequences:
        # Remove any padding or repeated characters:
        # Since we remapped token values earlier, our tokens are in [0, M-1]. To convert back:
        result = ""
        for token in seq:
            if token == -1:
                continue
            # Token value 0 corresponds to the lowest valid character.
            # Our tokenizer originally had word_index starting at 1,
            # and we subtracted one; so we add 1 back to look up the character.
            char = charset.get(token + 1, "")
            result += char
        results.append(result)
    return results

# build a reverse mapping dictionary from token index to character 
#(this is in order to read the test images and decode their outputs to evalluate them
reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
index_to_char = {index: char for char, index in tokenizer.word_index.items()}

#testing
import matplotlib.pyplot as plt

num_samples = 5
sample_imgs = X_val[:num_samples]
sample_labels = y_val[:num_samples]

#get the predictions
preds = prediction_model.predict(sample_imgs)

#decoding the predictions 
decoded_texts = []
for pred in preds:
    # CTC decode the prediction for a single sample
    input_len = np.array([pred.shape[0]])
    # The ctc_decode expects batch shape; we add batch dimension.
    decoded, _ = tf.keras.backend.ctc_decode(np.expand_dims(pred, axis=0), input_length=input_len, greedy=True)
    decoded_seq = decoded[0].numpy()[0]
    text = ""
    for token in decoded_seq:
        # Skip blank tokens (if any); token values range from 0 to (M-1)
        if token < 0: 
            continue
        text += index_to_char.get(token + 1, "")
    decoded_texts.append(text)

# predictions compared to the ground truth
for i in range(num_samples):
    plt.figure(figsize=(6, 2))
    plt.imshow(sample_imgs[i].squeeze(), cmap='gray')
    plt.title(f"Predicted: {decoded_texts[i]}\nGround Truth: " +
              "".join([index_to_char.get(token+1, "") for token in sample_labels[i] if token > 0]))
    plt.axis('off')
    plt.show()

"""
# NEW
--- Test on new handwriting images ---
"""

#test folder "test" provided
test_folder = "/Users/reionishi/Desktop/test" #change directory accordingly
test_image_paths = sorted([
    os.path.join(test_folder, f)
    for f in os.listdir(test_folder)
    if f.lower().endswith(".jpg")
])

test_images = np.array([preprocess_image_from_path(p) for p in test_image_paths])
test_preds = prediction_model.predict(test_images)
decoded_test_texts = decode_predictions(test_preds, index_to_char)

for i, img_path in enumerate(test_image_paths):
    plt.figure(figsize=(6, 2))
    plt.imshow(test_images[i].squeeze(), cmap='gray')
    plt.title(f"Predicted Text: {decoded_test_texts[i]}")
    plt.axis('off')
    plt.show()

# Save predictions to a file
output_file = "/Users/reionishi/Desktop/test_predictions.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for path, text in zip(test_image_paths, decoded_test_texts):
        filename = os.path.basename(path)
        f.write(f"{filename}: {text}\n")

print(f"Predictions saved to {output_file}")

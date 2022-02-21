import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, Embedding
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from HandleImbalances import use_smote

train_dataset_df = pd.read_csv('./Result/OSACT2022-taskA-train.csv', usecols=['tweet', 'off_label'], encoding='utf-8')
val_dataset_df = pd.read_csv('./Result/OSACT2022-taskA-dev.csv', usecols=['tweet', 'off_label'], encoding='utf-8')

tokenizer = Tokenizer()

tokenizer.fit_on_texts(train_dataset_df['tweet'])

encoded_dataset = tokenizer.texts_to_sequences(train_dataset_df['tweet'])

# max_length = max([len(s.split()) for s in train_dataset_df['tweet']])

max_length = 300

vocab_size = len(tokenizer.index_word) + 1

x_train = pad_sequences(encoded_dataset, maxlen=max_length, padding='post')

y_train = to_categorical(train_dataset_df['off_label'], 2)

x_train, y_train = shuffle(x_train, y_train)

encoded_val_dataset = tokenizer.texts_to_sequences(val_dataset_df['tweet'])

x_val = pad_sequences(encoded_val_dataset, maxlen=max_length, padding='post')
y_val = to_categorical(val_dataset_df['off_label'], 2)

model = Sequential()
model.add(Embedding(vocab_size, 300, input_length=max_length))
model.add(Conv1D(filters=64, kernel_size=4, activation='relu'))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(pool_size=1))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(x_train.shape[1], activation='relu'))
model.add(Dense(y_train.shape[1], activation='sigmoid'))
model.compile(optimizer=Adam(learning_rate=5e-5), loss='binary_crossentropy', metrics=[Precision(), Recall(), AUC()])

model_checkpoint_callback = ModelCheckpoint(
    filepath='./embedding_temp/checkpoint',
    save_weights_only=True,
    monitor='val_precision',
    mode='max',
    save_best_only=True)

model.fit(x_train, y_train,validation_data=(x_val, y_val), batch_size=32, epochs=10, verbose=2, callbacks=model_checkpoint_callback)

model.load_weights('./embedding_temp/checkpoint')

print("Saving model and weights started")
model_json = model.to_json()
with open('./embedding_model/model.json', 'w') as file:
    file.write(model_json)

model.save_weights('./embedding_model/weights.h5')
print("Saving model and weights ended")

#training on smote over sampling technique 

x_train_resampled, y_train_resampled = use_smote(x_train, train_dataset_df['off_label'])
y_train_resampled = to_categorical(y_train_resampled, num_classes=2)

model_checkpoint_callback = ModelCheckpoint(
    filepath='./embedding_resampled_temp/checkpoint',
    save_weights_only=True,
    monitor='val_precision',
    mode='max',
    save_best_only=True)

print("training on resampled data started")
model.fit(x_train, y_train,validation_data=(x_val, y_val), batch_size=32, epochs=10, verbose=2, callbacks=model_checkpoint_callback)
print("training on resampled data ended")

model.load_weights('./embedding_resampled_temp/checkpoint')

print("Saving model and weights started")
model_json = model.to_json()
with open('./embedding_resampled_model/model.json', 'w') as file:
    file.write(model_json)

model.save_weights('./embedding_resampled_model/weights.h5')
print("Saving model and weights ended")


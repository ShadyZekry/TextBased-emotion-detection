import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten
from tensorflow.keras.metrics import Precision, Recall, AUC


numerical_dataset_df = pd.read_csv('./Result/numerical_docs.csv', encoding='utf-8')

label_data = numerical_dataset_df['target']
sample_data = numerical_dataset_df.iloc[:, 1:]

x_train, y_train = shuffle(sample_data, label_data)

x_train, x_test, y_train, y_test = train_test_split(sample_data, label_data, test_size=0.2, shuffle=False)

#convert to binary classes 1 for medium depression and 0 for remaining classes
for record_index in range(len(y_train)):
        if y_train[record_index] == 2:
            y_train[record_index] = 1

y_train_encoded = to_categorical(y_train, dtype='int32')

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=1))
model.add(Conv1D(filters=64, kernel_size=6, activation='relu'))
model.add(MaxPooling1D(pool_size=1))
model.add(Flatten())
model.add(Dense(x_train.shape[1], activation='relu'))
model.add(Dense(units=y_train_encoded.shape[1], activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[AUC(), Precision(), Recall()])

# model = Sequential()
# model.add(Conv1D(filters=32, kernel_size = 3, activation='relu', input_shape=(x_train.shape[1], 1)))
# model.add(Dropout(0.5))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(filters=32, kernel_size = 4, activation='relu'))
# model.add(Dropout(0.5))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(filters=32, kernel_size = 5, activation='relu'))
# model.add(Dropout(0.5))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(x_train.shape[1], activation='relu'))
# model.add(Dense(units=y_train_encoded.shape[1], activation='sigmoid'))
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[AUC(), Precision(), Recall()])

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='./temp/checkpoint',
    save_weights_only=True,
    monitor='precision',
    mode='max',
    save_best_only=True)

print("Training model start:")
model.fit(x=x_train, y=y_train_encoded, batch_size=3, epochs=10, verbose=2, callbacks=model_checkpoint_callback)
print("Training model ended.")

model.load_weights('./Temp/checkpoint')

print("Saving model and weights started")
model_json = model.to_json()
with open('./Model/model.json', 'w') as file:
    file.write(model_json)

model.save_weights('./Model/weights.h5')
print("Saving model and weights ended")













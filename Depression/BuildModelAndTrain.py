import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.optimizers import Adam

numerical_dataset_df = pd.read_csv('./Result/numerical_dataset.csv', encoding='utf-8')

num_classes = 4

label_data = numerical_dataset_df['target']
sample_data = numerical_dataset_df.iloc[:, 1:]

#sample_data, label_data = shuffle(sample_data, label_data)

x_train, y_train = shuffle(sample_data, label_data)

#x_train, x_test, y_train, y_test = train_test_split(sample_data, label_data, test_size=0.2, shuffle=False)

std_scaler = StandardScaler()

x_train = std_scaler.fit_transform(x_train)
#x_test = std_scaler.transform(x_test)

y_train_encoded = to_categorical(y_train, num_classes= num_classes, dtype='int32')
#y_test_encoded = to_categorical(y_test, num_classes=num_classes, dtype='int32')

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=3))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=3))
model.add(Flatten())
model.add(Dense(x_train.shape[1], activation='relu'))
model.add(Dense(units=num_classes, activation='softmax'))
adam_optimizer = Adam(learning_rate=0.01)
model.compile(optimizer=adam_optimizer, loss='categorical_crossentropy', metrics=[AUC(), Precision(), Recall()])

model.fit(x=x_train, y=y_train_encoded, batch_size=64, epochs=100, verbose=2, validation_split=0.2)










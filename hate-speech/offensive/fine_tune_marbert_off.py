from utilities import load_train_features, load_val_features, load_train_labels, load_val_labels
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.metrics import Precision, Recall
from sklearn.preprocessing import StandardScaler, normalize
import numpy as np

x_train = load_train_features('offensive')
y_train = load_train_labels('offensive')

x_eval = load_val_features('offensive')
y_eval = load_val_labels('offensive')

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train, y_train)
x_eval = scaler.transform(x_eval)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3)

print(f'train samples:{x_train.shape[0]}')
print(f'test samples:{x_test.shape[0]}')
print(f'eval samples:{x_eval.shape[0]}')

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)
y_eval = to_categorical(y_eval, num_classes=2)

es_cb = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=2, restore_best_weights=True)

model = Sequential(name='fine_tuning_marbert')
model.add(Input(shape=(x_train.shape[1])))
model.add(Dense(2, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=5e-5), loss='binary_crossentropy', metrics=[Precision(name='p'), Recall(name='r')])
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=64, callbacks=es_cb, epochs=1000, use_multiprocessing=True, workers=8)

print('train data:')
train_data_conf_matrix = classification_report(np.argmax(y_train, axis=-1), np.argmax(model.predict(x_train), axis=-1), target_names=['not_off', 'off'])
print(train_data_conf_matrix)

print('test data:')
test_data_conf_matrix = classification_report(np.argmax(y_test, axis=-1), np.argmax(model.predict(x_test), axis=-1), target_names=['not_off', 'off'])
print(test_data_conf_matrix)

print('eval data:')
eval_data_conf_matrix = classification_report(np.argmax(y_eval, axis=-1), np.argmax(model.predict(x_eval), axis=-1), target_names=['not_off', 'off'])
print(eval_data_conf_matrix)

user_input = input('Save model?..(y/n)')
if user_input.lower() == 'y':
    model_json = model.to_json()
    with open(f'./{model.name}_off.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(f'./{model.name}_off.h5')
    with open('./results.txt', 'a') as results_file:
        results_file.write(f'\n{model.name} for offensive task results:\n')
        results_file.write('train data confusion matrix:\n')
        results_file.write(train_data_conf_matrix + '\n')
        results_file.write('test data confusion matrix:\n')
        results_file.write(test_data_conf_matrix + '\n')
        results_file.write('eval data confusion matrix:\n')
        results_file.write(eval_data_conf_matrix + '\n')
        results_file.write('-'*50)






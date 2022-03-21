from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from utilities import load_train_features, load_train_labels, load_val_features, load_val_labels
from sklearn.metrics import classification_report
from keras.metrics import Precision, Recall
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train = load_train_features()
x_val = load_val_features()

x_train = scaler.fit_transform(x_train)
x_val = scaler.fit_transform(x_val)

y_train_off = load_train_labels('offensive')
y_val_off = load_val_labels('offensive')

y_train_hs = load_train_labels('hs')
y_val_hs = load_val_labels('hs')

input_layer = Input(shape=(x_train.shape[1],1))
conv_layer = Conv1D(100, kernel_size=3)(input_layer)
max_pooling = MaxPooling1D(pool_size=1)(conv_layer)
flatten_layer = Flatten()(max_pooling)
output_1 = Dense(2, activation='sigmoid', name='1')(flatten_layer)
output_2 = Dense(2, activation='sigmoid', name='2')(flatten_layer)

model = Model(inputs=input_layer, outputs=[output_1, output_2], name='cnn_mtl')

model.compile(optimizer=Adam(learning_rate=2*10**-5), loss=BinaryCrossentropy(), metrics=[Precision(name='p', class_id=1), Recall(name='r', class_id=1)])

early_cb = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights = True, patience=300)

history = model.fit(x_train, [y_train_off, y_train_hs], validation_data=[x_val, (y_val_off, y_val_hs)], epochs=2000, verbose=2, batch_size=64, callbacks=[early_cb]).history

total_off_val = np.count_nonzero(np.argmax(y_val_off, axis=-1))
total_hs_val = np.count_nonzero(np.argmax(y_val_hs, axis=-1))

total_off_train = np.count_nonzero(np.argmax(y_train_off, axis=-1))
total_hs_train = np.count_nonzero(np.argmax(y_train_hs, axis=-1))

predicted_off_train = np.argmax(model.predict(x_train)[0], axis=-1)
predicted_off_val = np.argmax(model.predict(x_val)[0], axis=-1)

predicted_hs_train = np.argmax(model.predict(x_train)[1], axis=-1)
predicted_hs_val = np.argmax(model.predict(x_val)[1], axis=-1)

true_y_off_train = np.argmax(y_train_off, axis=-1)
true_y_hs_train = np.argmax(y_train_hs, axis=-1)
true_y_off_val = np.argmax(y_val_off, axis=-1)
true_y_hs_val = np.argmax(y_val_hs, axis=-1)

tp_off_train = 0
for i in range(len(predicted_off_train)):
    if predicted_off_train[i] == 1 and true_y_off_train[i] == 1:
        tp_off_train += 1
train_predict_off_str = f'{model.name} predicted {tp_off_train} offensive from {total_off_train} in training data'
print(train_predict_off_str)

tp_off_val = 0
for i in range(len(predicted_off_val)):
    if predicted_off_val[i] == 1 and true_y_off_val[i]:
        tp_off_val += 1
val_predict_off_str = f'{model.name} predicted {tp_off_val} offensive from {total_off_val} in validation data'
print(val_predict_off_str)

tp_hs_train = 0
for i in range(len(predicted_hs_train)):
    if predicted_hs_train[i] == 1 and true_y_hs_train[i] == 1:
        tp_hs_train += 1
train_predict_hs_str = f'{model.name} predicted {tp_hs_train} hate-speech from {total_hs_train} in training data'
print(train_predict_hs_str)

tp_hs_val = 0
for i in range(len(predicted_hs_val)):
    if predicted_hs_val[i] == 1 and true_y_hs_val[i] == 1:
        tp_hs_val += 1
val_predict_hs_str = f'{model.name} predicted {tp_hs_val} hate-speech from {total_hs_val} in validation data'
print(val_predict_hs_str)

conf_matrix_off = classification_report(np.argmax(y_val_off, axis=-1), np.argmax(model.predict(x_val)[0], axis=-1), target_names=['normal','off'])

conf_matrix_hs = classification_report(np.argmax(y_val_hs, axis=-1), np.argmax(model.predict(x_val)[1], axis=-1), target_names=['normal','hs'])

print(f'confusion matrix offensive:\n{conf_matrix_off}')
print(f'confusion matrix hs:\n{conf_matrix_hs}')

inp = input('Save results?..(y/n):')
if inp.lower() == 'y':
    with open('./mtl_results.txt', mode='a') as result_file:
        result_file.write(f'Results for model: {model.name}\n')
        result_file.write(f'confusion matrix offensive:\n{conf_matrix_off}\n')
        result_file.write(f'confusion matrix hs:\n{conf_matrix_hs}\n')
        result_file.write(f'{train_predict_off_str}\n')
        result_file.write(f'{val_predict_off_str}\n')
        result_file.write(f'{train_predict_hs_str}\n')
        result_file.write(f'{val_predict_hs_str}\n')

inp = input('Save model?..(y/n):')
if inp.lower() == 'y':
    model_json = model.to_json()
    with open(f'./models_mtl/{model.name}.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(f'./models_mtl/{model.name}_weights.h5')


inp = input('Plot loss?..(y/n):')
if inp.lower() == 'y':
    plt.plot(history['off_loss'])
    plt.plot(history['val_off_loss'])
    plt.title('model offensive task loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history['hs_loss'])
    plt.plot(history['val_hs_loss'])
    plt.title('model hate-speech task loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()




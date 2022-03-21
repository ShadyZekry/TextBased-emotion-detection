from models import *
from utilities import load_train_features, load_val_features, load_train_labels, load_val_labels
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from keras.metrics import Precision, Recall
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize

x_train = load_train_features()
y_train = load_train_labels('offensive')

x_val = load_val_features()
y_val = load_val_labels('offensive')

class_weights = {0:1, 1:2}

x_train, y_train = shuffle(x_train, y_train)
x_val, y_val = shuffle(x_val, y_val)

x_train = normalize(x_train)
x_val = normalize(x_val)

model = build_cnn_model((x_train.shape[1], 1), 2)

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[Precision(class_id=1), Recall(class_id=1)])

early_cb = EarlyStopping(monitor='val_precision', mode='max', restore_best_weights = True, patience=200)

history = model.fit(x_train, y_train, batch_size=64, epochs=2000, callbacks=[early_cb], validation_data=(x_val, y_val), class_weight=class_weights).history

model_json = model.to_json()
with open(f'./models/{model.name}_off.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights(f'./models/{model.name}_off_weights.h5')

# with open('./models/ann_off.json', 'r') as json:
#     model = model_from_json(json.read())

# model.load_weights(f'./models/{model.name}_off_weights.h5')

preds = np.argmax(model.predict(x_train), axis=-1)
truth = np.argmax(y_train, axis=-1)
total_off = np.count_nonzero(truth==1)
conf_matrix = classification_report(truth,preds,target_names=['normal','off'])
print(f'Confusion matrix for {model.name} model (training data):\n{conf_matrix}')

off_tp = 0
for index in range(len(preds)):
    if preds[index] == 1 and truth[index] == 1:
        off_tp += 1

predicted_res_str = f'{model.name} model predicted {off_tp} offensive from {total_off} offensive tweets(training data)'

print(f'{model.name} model predicted {off_tp} offensive from {total_off} offensive tweets (training data)')

with open('./off_results.txt', mode='a') as results_file:
    results_file.write(f'Confusion matrix for {model.name} model:\n')
    results_file.write(f'{conf_matrix}\n')
    results_file.write(f'{predicted_res_str}\n')

preds = np.argmax(model.predict(x_val), axis=-1)
truth = np.argmax(y_val, axis=-1)
total_off = np.count_nonzero(truth==1)
conf_matrix = classification_report(truth,preds,target_names=['normal','off'])
print(f'Confusion matrix for {model.name} model(validation data):\n{conf_matrix}')

off_tp = 0
for index in range(len(preds)):
    if preds[index] == 1 and truth[index] == 1:
        off_tp += 1

predicted_res_str = f'{model.name} model predicted {off_tp} offensive from {total_off} offensive tweets(validation data)'
print(f'{model.name} model predicted {off_tp} offensive from {total_off} offensive tweets (validation data)')

with open('./off_results.txt', mode='a') as results_file:
    results_file.write(f'Confusion matrix for {model.name} model:\n')
    results_file.write(f'{conf_matrix}\n')
    results_file.write(f'{predicted_res_str}\n')

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()



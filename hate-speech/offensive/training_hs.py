from models import *
from utilities import load_train_features, load_val_features, load_train_labels, load_val_labels
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras.utils import to_categorical
from imblearn.pipeline import Pipeline

x_train = load_train_features('hs_')
y_train = load_train_labels('hs')

x_val = load_val_features()
y_val = load_val_labels('hs')

# num_hs = np.argmax(y_train, axis=-1)
# print(np.count_nonzero(num_hs == 1), ' before smote')

# # oversample = SMOTE()
# # x_train, y_train = oversample.fit_resample(x_train, np.argmax(y_train, axis=-1))

# # over = SMOTE(sampling_strategy=0.5)
# # over = SMOTE(sampling_strategy=0.9, k_neighbors=15)
# # under = RandomUnderSampler(sampling_strategy=0.9)
# # steps = [('o', over),('u', under)]
# # pipeline = Pipeline(steps=steps)
# # x_train, y_train = pipeline.fit_resample(x_train, np.argmax(y_train, axis=-1))
# over = SMOTE(sampling_strategy=0.9, k_neighbors=1)
# x_train, y_train = over.fit_resample(x_train, np.argmax(y_train, axis=-1))

# y_train = to_categorical(y_train)

# num_hs = np.argmax(y_train, axis=-1)
# print(np.count_nonzero(num_hs == 1), ' after smote')

# class_weights = {0:1, 1:2}

x_train, y_train = shuffle(x_train, y_train)
x_val, y_val = shuffle(x_val, y_val)

x_train = normalize(x_train)
x_val = normalize(x_val)

# model = build_ann_model((x_train.shape[1], 1), 2)

# model.summary()

# model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[Precision(class_id=1), Recall(class_id=1)])

# early_cb = EarlyStopping(monitor='val_precision', mode='max', restore_best_weights = True, patience=200)

# history = model.fit(x_train, y_train, batch_size=64, epochs=2000, callbacks=[early_cb], validation_data=(x_val, y_val)).history

# model_json = model.to_json()
# with open(f'./models/{model.name}_hs.json', 'w') as json_file:
#     json_file.write(model_json)
# model.save_weights(f'./models/{model.name}_hs_weights.h5')

with open('./models/ann_hs.json', 'r') as json:
    model = model_from_json(json.read())

model.load_weights(f'./models/{model.name}_hs_weights.h5')

preds = np.argmax(model.predict(x_train), axis=-1)
truth = np.argmax(y_train, axis=-1)
total_hs = np.count_nonzero(truth==1)
conf_matrix = classification_report(truth,preds,target_names=['normal','hs'])
print(f'Confusion matrix for {model.name} model (training data):\n{conf_matrix}')

hs_tp = 0
for index in range(len(preds)):
    if preds[index] == 1 and truth[index] == 1:
        hs_tp += 1

predicted_res_str = f'{model.name} model predicted {hs_tp} hs from {total_hs} hs tweets(training data)'

print(f'{model.name} model predicted {hs_tp} hs from {total_hs} hs tweets (training data)')

# with open('./hs_results.txt', mode='a') as results_file:
#     results_file.write(f'Confusion matrix for {model.name} model:\n')
#     results_file.write(f'{conf_matrix}\n')
#     results_file.write(f'{predicted_res_str}\n')

preds = np.argmax(model.predict(x_val), axis=-1)
truth = np.argmax(y_val, axis=-1)
total_hs = np.count_nonzero(truth==1)
conf_matrix = classification_report(truth,preds,target_names=['normal','hs'])
print(f'Confusion matrix for {model.name} model(validation data):\n{conf_matrix}')

hs_tp = 0
for index in range(len(preds)):
    if preds[index] == 1 and truth[index] == 1:
        hs_tp += 1

predicted_res_str = f'{model.name} model predicted {hs_tp} hs from {total_hs} hs tweets(validation data)'
print(f'{model.name} model predicted {hs_tp} hs from {total_hs} hs tweets (validation data)')

# with open('./hs_results.txt', mode='a') as results_file:
#     results_file.write(f'Confusion matrix for {model.name} model:\n')
#     results_file.write(f'{conf_matrix}\n')
#     results_file.write(f'{predicted_res_str}\n')

# plt.plot(history['loss'])
# plt.plot(history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()



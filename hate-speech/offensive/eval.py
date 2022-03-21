from utilities import load_train_features, load_train_labels, load_val_features, load_val_labels
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
from keras.models import model_from_json
import numpy as np

def perf_measure(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_pred)): 
        if y_actual[i]==y_pred[i]==1:
           TP += 1
        if y_pred[i]==1 and y_actual[i]!=y_pred[i]:
           FP += 1
        if y_actual[i]==y_pred[i]==0:
           TN += 1
        if y_pred[i]==0 and y_actual[i]!=y_pred[i]:
           FN += 1
    return(TP, FP, TN, FN)

x_train_hs = load_train_features('hs_')
x_train_off = load_train_features()
x_val = load_val_features()

x_train_off = normalize(x_train_off)
x_train_hs = normalize(x_train_hs)
x_val = normalize(x_val)

y_train_off = load_train_labels('offensive')
y_val_off = load_val_labels('offensive')

y_train_hs = load_train_labels('hs')
y_val_hs = load_val_labels('hs')

with open('./models/cnn_off.json', 'r') as cnn_json:
    off_model = model_from_json(cnn_json.read())

off_model.load_weights(f'./models/{off_model.name}_off_weights.h5')

with open('./models/ann_hs.json', 'r') as ann_json:
    hs_model = model_from_json(ann_json.read())

hs_model.load_weights(f'./models/{hs_model.name}_hs_weights.h5')

total_off_val = np.count_nonzero(np.argmax(y_val_off, axis=-1))
total_hs_val = np.count_nonzero(np.argmax(y_val_hs, axis=-1))

total_off_train = np.count_nonzero(np.argmax(y_train_off, axis=-1))
total_hs_train = np.count_nonzero(np.argmax(y_train_hs, axis=-1))

predicted_off_train = np.argmax(off_model.predict(x_train_off), axis=-1)
predicted_off_val = np.argmax(off_model.predict(x_val), axis=-1)

predicted_hs_train = np.argmax(hs_model.predict(x_train_hs), axis=-1)
predicted_hs_val = np.argmax(hs_model.predict(x_val), axis=-1)

true_y_off_train = np.argmax(y_train_off, axis=-1)
true_y_hs_train = np.argmax(y_train_hs, axis=-1)
true_y_off_val = np.argmax(y_val_off, axis=-1)
true_y_hs_val = np.argmax(y_val_hs, axis=-1)

# tp_off, fp_off, tn_off, fn_off = perf_measure(np.argmax(y_val_off, axis=-1), np.argmax(off_model.predict(x_val), axis=-1))

# print(f'tp_off:{tp_off}, fp_off:{fp_off}, tn_off:{tn_off}, fn_off:{fn_off}')

# tp_hs, fp_hs, tn_hs, fn_hs = perf_measure(np.argmax(y_val_hs, axis=-1), np.argmax(hs_model.predict(x_val), axis=-1))

# print(f'tp_hs:{tp_hs}, fp_hs:{fp_hs}, tn_hs:{tn_hs}, fn_hs:{fn_hs}')

tp_off_train = 0
for i in range(len(predicted_off_train)):
    if predicted_off_train[i] == 1 and true_y_off_train[i] == 1:
        tp_off_train += 1
train_predict_off_str = f'{off_model.name} predicted {tp_off_train} offensive from {total_off_train} in training data'
print(train_predict_off_str)

tp_off_val = 0
for i in range(len(predicted_off_val)):
    if predicted_off_val[i] == 1 and true_y_off_val[i]:
        tp_off_val += 1
val_predict_off_str = f'{off_model.name} predicted {tp_off_val} offensive from {total_off_val} in validation data'
print(val_predict_off_str)

tp_hs_train = 0
for i in range(len(predicted_hs_train)):
    if predicted_hs_train[i] == 1 and true_y_hs_train[i] == 1:
        tp_hs_train += 1
train_predict_hs_str = f'{hs_model.name} predicted {tp_hs_train} hate-speech from {total_hs_train} in training data'
print(train_predict_hs_str)

tp_hs_val = 0
for i in range(len(predicted_hs_val)):
    if predicted_hs_val[i] == 1 and true_y_hs_val[i] == 1:
        tp_hs_val += 1
val_predict_hs_str = f'{hs_model.name} predicted {tp_hs_val} hate-speech from {total_hs_val} in validation data'
print(val_predict_hs_str)

conf_matrix_off = classification_report(np.argmax(y_val_off, axis=-1), np.argmax(off_model.predict(x_val), axis=-1), target_names=['not_off','off'])

conf_matrix_hs = classification_report(np.argmax(y_val_hs, axis=-1), np.argmax(hs_model.predict(x_val), axis=-1), target_names=['not_hs','hs'])

print(f'confusion matrix offensive validation:\n{conf_matrix_off}')
print(f'confusion matrix hs validation:\n{conf_matrix_hs}')

conf_matrix_off = classification_report(np.argmax(y_train_off, axis=-1), np.argmax(off_model.predict(x_train_off), axis=-1), target_names=['not_off','off'])

conf_matrix_hs = classification_report(np.argmax(y_train_hs, axis=-1), np.argmax(hs_model.predict(x_train_hs), axis=-1), target_names=['not_hs','hs'])

print(f'confusion matrix offensive training:\n{conf_matrix_off}')
print(f'confusion matrix hs training:\n{conf_matrix_hs}')



from utilities import load_train_features, load_val_features, load_train_labels, load_val_labels
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

x_train = load_train_features('hatespeech')
y_train = load_train_labels('hatespeech')

x_eval = load_val_features('hatespeech')
y_eval = load_val_labels('hatespeech')

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train, y_train)
x_eval = scaler.transform(x_eval)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3)

print(f'train samples:{x_train.shape[0]}')
print(f'test samples:{x_test.shape[0]}')
print(f'eval samples:{x_eval.shape[0]}')

# model = LogisticRegression(verbose=2, solver='saga', C=1e-3)
model = RandomForestClassifier(verbose=2, max_samples=0.4)

model.fit(x_train, y_train)

print('train data confusion matrix')
train_data_conf_matrix = classification_report(y_train, model.predict(x_train), target_names=['not_hs', 'hs'])
print(train_data_conf_matrix)
print('test data confusion matrix')
test_data_conf_matrix = classification_report(y_test, model.predict(x_test), target_names=['not_hs', 'hs'])
print(test_data_conf_matrix)
print('eval data confusion matrix')
eval_data_conf_matrix = classification_report(y_eval, model.predict(x_eval), target_names=['not_hs', 'hs'])
print(eval_data_conf_matrix)

user_input = input('Save model?(y/n)')
if user_input.lower() == 'y':
    model_filename = input('enter model filename:')
    pickle.dump(model, open(model_filename, 'wb'))
    with open('./results.txt', 'a') as results_file:
        results_file.write(f'\n{model_filename} for hate-speech task results:\n')
        results_file.write('train data confusion matrix:\n')
        results_file.write(train_data_conf_matrix + '\n')
        results_file.write('test data confusion matrix:\n')
        results_file.write(test_data_conf_matrix + '\n')
        results_file.write('eval data confusion matrix:\n')
        results_file.write(eval_data_conf_matrix + '\n')
        results_file.write('-'*50)

# lr_loaded_model = pickle.load(open(lr_model_filename, 'rb'))

# print('train')
# print(classification_report(y_train, lr_loaded_model.predict(x_train), target_names=['not_off', 'off']))
# print('test')
# print(classification_report(y_test, lr_loaded_model.predict(x_test), target_names=['not_off', 'off']))
# print('eval')
# print(classification_report(y_eval, lr_loaded_model.predict(x_eval), target_names=['not_off', 'off']))



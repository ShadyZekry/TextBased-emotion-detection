from keras.models import model_from_json
from captum.attr import IntegratedGradients
from utilities import load_train_features, load_train_labels, load_val_features, load_val_labels
from sklearn.preprocessing import normalize
from torch import as_tensor

x_train = load_train_features()
y_train = load_train_labels('offensive')

x_val = load_val_features()
y_val = load_val_labels('offensive')

x_val = normalize(x_val)
x_train = normalize(x_train)

with open('./models/ann_off.json', 'r') as json:
    model = model_from_json(json.read())

model.load_weights(f'./models/{model.name}_off_weights.h5')

ig = IntegratedGradients(model)
attributions, delta = ig.attribute(as_tensor(x_train), as_tensor(x_train), target=1, return_convergence_delta=True)

print(attributions)

print(delta)




from keras.models import model_from_json
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
import pandas as pd
from tensorflow.keras.metrics import Precision, Recall, AUC

dataset_df = pd.read_csv('./Result/numerical_docs.csv', encoding='utf-8')

y = dataset_df['target']
x = dataset_df.iloc[:, 1:] 

x, y = shuffle(x, y)

y = to_categorical(y, dtype='int32')

model_json = open('./Model/model.json', 'r')
loaded_model_json = model_json.read()
model_json.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("./Model/weights.h5")

loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[AUC(), Precision(), Recall()])
loaded_model.evaluate(x, y, verbose=2)



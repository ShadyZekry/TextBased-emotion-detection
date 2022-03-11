import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Dense, MaxPooling1D, Flatten, GRU, Dropout, Concatenate
from tensorflow.keras.models import Sequential, model_from_json, Model
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.activations import softmax
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm
import os
import numpy as np

class ModelsUtilities:
    models_names = ['gru', 'cnn', 'cnn-gru']

    def __init__(self, task:str, bert_model, bert_tokenizer):
        self.__task = task
        self.__train_metrics = [Precision(name='train_precision'), Recall(name='train_recall')]
        self.__val_metrics = [Precision(name='val_precision'), Recall(name='val_recall')]
        self.__optimizer = Adam()
        self.__loss_fn = BinaryCrossentropy()
        self.__bert_model = bert_model
        self.__bert_tokenizer = bert_tokenizer
        self.__train_shape = (60, 768)

    def build_and_fit_cnn(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, batch_size: int, epochs: int, feature_method: str, imbalance_handler: str, save_best = False) -> tf.keras.models.Sequential:
        cnn_model = self.__build_cnn_model(self.__train_shape)
        cnn_model.add(Dense(y_train.shape[1], activation='softmax'))
        cnn_model.summary()
        f1_scores, weights, f1_train = self.__train(x_train, y_train, x_val, y_val, epochs, cnn_model, self.__train_metrics, self.__val_metrics, batch_size, self.__optimizer, self.__loss_fn, feature_method, imbalance_handler)
        best_result_index = f1_scores.index(max(f1_scores))
        for layer_indx in range(len(cnn_model.layers)):
            cnn_model.layers[layer_indx].set_weights(weights[best_result_index][layer_indx])
        self.__print_training_output(f1_scores[best_result_index], f1_train[best_result_index], cnn_model.name, feature_method, imbalance_handler)
        if save_best:
            self.__save_model(cnn_model, feature_method, imbalance_handler)
            self.__save_result(cnn_model.name, f1_scores[best_result_index],f1_train[best_result_index], feature_method, imbalance_handler)
        return cnn_model
    
    def build_and_fit_gru(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, batch_size: int, epochs: int, feature_method: str, imbalance_handler: str, save_best = False) -> tf.keras.models.Sequential:
        gru_model = self.__build_gru_model(self.__train_shape)
        gru_model.add(Dense(y_train.shape[1], activation='softmax'))
        gru_model.summary()
        f1_scores, weights, f1_train = self.__train(x_train, y_train, x_val, y_val, epochs, gru_model, self.__train_metrics, self.__val_metrics, batch_size, self.__optimizer, self.__loss_fn, feature_method, imbalance_handler)
        best_result_index = f1_scores.index(max(f1_scores))
        for layer_indx in range(len(gru_model.layers)):
            gru_model.layers[layer_indx].set_weights(weights[best_result_index][layer_indx])
        self.__print_training_output(f1_scores[best_result_index], f1_train[best_result_index], gru_model.name, feature_method, imbalance_handler)
        if save_best:
            self.__save_model(gru_model, feature_method, imbalance_handler)
            self.__save_result(gru_model.name, f1_scores[best_result_index], f1_train[best_result_index],feature_method, imbalance_handler)
        return gru_model
    
    def build_and_fit_cnn_gru(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, batch_size: int, epochs: int, feature_method: str, imbalance_handler: str, save_best = False) -> tf.keras.models.Sequential:
        cnn_gru_model = self.__build_cnn_gru_model(self.__train_shape)
        cnn_gru_model.add(Dense(y_train.shape[1], activation='softmax'))
        cnn_gru_model.summary()
        f1_scores, weights, f1_train = self.__train(x_train, y_train, x_val, y_val, epochs, cnn_gru_model, self.__train_metrics, self.__val_metrics, batch_size, self.__optimizer, self.__loss_fn, feature_method, imbalance_handler)
        best_result_index = f1_scores.index(max(f1_scores))
        for layer_indx in range(len(cnn_gru_model.layers)):
            cnn_gru_model.layers[layer_indx].set_weights(weights[best_result_index][layer_indx])
        self.__print_training_output(f1_scores[best_result_index],f1_train[best_result_index], cnn_gru_model.name, feature_method, imbalance_handler)
        if save_best:
            self.__save_model(cnn_gru_model, feature_method, imbalance_handler)
            self.__save_result(cnn_gru_model.name, f1_scores[best_result_index], f1_train[best_result_index],feature_method, imbalance_handler)
        return cnn_gru_model

    def build_and_fit_gru_weighted(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, batch_size: int, epochs: int, feature_method: str, imbalance_handler: str, class_weights: list, save_best = False) -> tf.keras.models.Sequential:
        gru_model = self.__build_gru_model(self.__train_shape)
        gru_model.add(Dense(y_train.shape[1]))
        gru_model.summary()
        f1_scores, weights, f1_train = self.__train(x_train, y_train, x_val, y_val, epochs, gru_model, self.__train_metrics, self.__val_metrics, batch_size, self.__optimizer, self.__weighted_loss_fn(class_weights), feature_method, imbalance_handler, True)
        best_result_index = f1_scores.index(max(f1_scores))
        for layer_indx in range(len(gru_model.layers)):
            gru_model.layers[layer_indx].set_weights(weights[best_result_index][layer_indx])
        self.__print_training_output(f1_scores[best_result_index], f1_train[best_result_index],gru_model.name, feature_method, imbalance_handler)
        if save_best:
            self.__save_model(gru_model, feature_method, imbalance_handler)
            self.__save_result(gru_model.name, f1_scores[best_result_index], f1_train[best_result_index],feature_method, imbalance_handler)
        return gru_model
    
    def build_and_fit_cnn_weighted(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, batch_size: int, epochs: int, feature_method: str, imbalance_handler: str, class_weights: list, save_best = False) -> tf.keras.models.Sequential:
        cnn_model = self.__build_cnn_model(self.__train_shape)
        cnn_model.add(Dense(y_train.shape[1]))
        cnn_model.summary()
        f1_scores, weights, f1_train = self.__train(x_train, y_train, x_val, y_val, epochs, cnn_model, self.__train_metrics, self.__val_metrics, batch_size, self.__optimizer, self.__weighted_loss_fn(class_weights), feature_method, imbalance_handler, True)
        best_result_index = f1_scores.index(max(f1_scores))
        for layer_indx in range(len(cnn_model.layers)):
            cnn_model.layers[layer_indx].set_weights(weights[best_result_index][layer_indx])
        self.__print_training_output(f1_scores[best_result_index],f1_train[best_result_index], cnn_model.name, feature_method, imbalance_handler)
        if save_best:
            self.__save_model(cnn_model, feature_method, imbalance_handler)
            self.__save_result(cnn_model.name, f1_scores[best_result_index], f1_train[best_result_index],feature_method, imbalance_handler)
        return cnn_model

    def build_and_fit_cnn_gru_weighted(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, batch_size: int, epochs: int, feature_method: str, imbalance_handler: str, class_weights: list, save_best = False) -> tf.keras.models.Sequential:
        cnn_gru_model = self.__build_cnn_gru_model(self.__train_shape)
        cnn_gru_model.add(Dense(y_train.shape[1]))
        cnn_gru_model.summary()
        f1_scores, weights, f1_train = self.__train(x_train, y_train, x_val, y_val, epochs, cnn_gru_model, self.__train_metrics, self.__val_metrics, batch_size, self.__optimizer, self.__weighted_loss_fn(class_weights), feature_method, imbalance_handler, True)
        best_result_index = f1_scores.index(max(f1_scores))
        for layer_indx in range(len(cnn_gru_model.layers)):
            cnn_gru_model.layers[layer_indx].set_weights(weights[best_result_index][layer_indx])
        self.__print_training_output(f1_scores[best_result_index], f1_train[best_result_index], cnn_gru_model.name, feature_method, imbalance_handler)
        if save_best:
            self.__save_model(cnn_gru_model, feature_method, imbalance_handler)
            self.__save_result(cnn_gru_model.name, f1_scores[best_result_index], f1_train[best_result_index],feature_method, imbalance_handler)
        return cnn_gru_model

    def __weighted_loss_fn(self, weights):
        def wrapper(labels, logits):
            loss = tf.nn.weighted_cross_entropy_with_logits(
            labels, logits, weights
        )
            return loss
        return wrapper

    def __print_training_output(self, f1_score: float, f1_train: float, model_name: str, feature_method: str, imbalance_handler: str) -> None:
        print(f'best val_f1_score:{f1_score:.4f} train_f1_score:{f1_train:.4f} for {self.__task} and {model_name} model using {feature_method} and {imbalance_handler} for imbalance handling')

    def __save_model(self, model, feature_method, imbalance_handler):
        full_path = f'./{self.__task}/{feature_method}/{model.name}_{imbalance_handler}'
        if not os.path.isdir(f'./{self.__task}'):
            os.mkdir(f'./{self.__task}')
        if not os.path.isdir(f'./{self.__task}/{feature_method}'):
            os.mkdir(f'./{self.__task}/{feature_method}')
        if not os.path.isdir(full_path):
            os.mkdir(full_path)
        model_json = model.to_json()
        with open(f'{full_path}/{model.name}.json', 'w') as file:
            file.write(model_json)
        model.save_weights(f'{full_path}/weights.h5')
        print(f'Successfuly saved model and weights')

    def __save_result(self, model_name, f1_score, f1_train, feature_method, imbalance_handler):
        file_name = f'{self.__task}_results.txt'
        line = f'{model_name} achieved val_f1 score:{f1_score:.4f} train_f1_score:{f1_train:.4f} using {feature_method} and {imbalance_handler} for imbalance handling\n'
        if os.path.isfile(f'./{file_name}'):
            with open(f'./{file_name}', 'a') as result_file:
                result_file.write(line)
        else:
            with open(f'./{file_name}', 'w') as result_file:
                result_file.write(line)
        print(f'Successfuly written model best result in {file_name}')

    def read_model(self, model_name, feature_method: str, imbalance_handler: str):
        path = f'./{self.__task}/{feature_method}/{model_name}_{imbalance_handler}'
        if not os.path.isdir(path):
            print('Model doesnt exist')
            return None
        else:
            model = None
            with open(f'{path}/{model_name}.json', 'r') as json_model:
                model = model_from_json(json_model.read())
            model.load_weights(f'{path}/weights.h5')
            return model
    
    def __build_cnn_model(self, input_shape : tuple) -> Sequential:
        model = Sequential(name='cnn')
        i_shape = (input_shape[0], input_shape[1])
        input_layer = Input(shape=i_shape)
        kernel_size= {}
        kernel_size[0]= [3]
        kernel_size[1]= [4]
        kernel_size[2]= [5]
        convs = []
        for k_no in range(len(kernel_size)):
            conv = Conv1D(100, kernel_size=kernel_size[k_no][0])(input_layer)
            max_pool = MaxPooling1D(pool_size=4)(conv)
            drop = Dropout(0.2)(max_pool)
            convs.append(drop)
        out = Concatenate()(convs)
        model.add(Model(inputs=input_layer, outputs=out, name='3-parallel-cnns'))
        model.add(Dropout(0.2))
        model.add(MaxPooling1D(pool_size=4))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.2))
        return model

    def __build_gru_model(self, input_shape: tuple) -> Sequential:
        model = Sequential(name='gru')
        model.add(Input(shape=(input_shape[0], input_shape[1])))
        model.add(GRU(250))
        model.add(Dropout(0.2))
        model.add(Dense(100, activation='relu'))
        return model

    def __build_cnn_gru_model(self, input_shape: tuple) -> Sequential:
        model = Sequential(name='cnn-gru')
        model.add(Input(shape=(input_shape[0], input_shape[1])))
        model.add(Conv1D(100, kernel_size=4, activation='relu'))
        model.add(Dropout(0.2))
        model.add(MaxPooling1D(pool_size=4))
        model.add(GRU(200))
        model.add(Dropout(0.2))
        return model

    @tf.function
    def __train_step(self, x, y, model, optimizer, loss_fn, train_metrics, apply_softmax = False):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = loss_fn(y, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        if apply_softmax:
            logits = softmax(logits, axis = -1)
        for metric in train_metrics:
            metric.update_state(y, logits)
        return loss_value

    @tf.function
    def __val_step(self, x, y, model, val_metrics, apply_softmax = False):
        val_logits = model(x, training=False)
        if apply_softmax:
            val_logits = softmax(val_logits, axis = -1)
        for metric in val_metrics:
            metric.update_state(y, val_logits)
    
    def evaluate(self, model, x_val, y_val, batch_size, apply_softmax):
        val_rem = x_val.shape[0] % batch_size
        val_num_batches = x_val.shape[0] // batch_size
        for batch in tqdm(range(1, val_num_batches), desc=f'Evaluating {model.name}'):
            i = batch - 1
            x = x_val[i * batch_size: batch * batch_size]
            y = y_val[i * batch_size: batch * batch_size]
            x = self.__get_bert_embeddings(x)
            self.__val_step(x, y, model, self.__val_metrics, apply_softmax)
        if val_rem != 0:
            x = x_val[x_val.shape[0] - val_rem : x_val.shape[0]]
            x = self.__get_bert_embeddings(x)
            y = y_val[y_val.shape[0] - val_rem : y_val.shape[0]]
            self.__val_step(x, y, model, self.__val_metrics, apply_softmax)
        precision = float(self.__val_metrics[0].result())
        recall = float(self.__val_metrics[1].result())
        f1_score = (2 * precision * recall) / (precision + recall)
        print(f'Evaluating {model.name}: precision:{precision:.4f}, recall:{recall:.4f}, f1_score:{f1_score:.4f}')
        for metric in self.__val_metrics:
            metric.reset_states()
    
    def __get_bert_embeddings(self, texts):
        tokens_ids = []
        try:
            if type(texts) != str:
                for text in texts:
                    tokens_ids.append(self.__bert_tokenizer(text, padding='max_length', truncation=True, max_length=self.__train_shape[0], add_special_tokens=True)['input_ids'])
            else:
                tokens_ids.append(self.__bert_tokenizer(texts, padding='max_length', truncation=True, max_length=self.__train_shape[0], add_special_tokens=True)['input_ids'])
            hidden_states = self.__bert_model(tf.convert_to_tensor(tokens_ids))[2]
        except TypeError:
            tokens_ids.append(tokens_ids.append(self.__bert_tokenizer(texts, padding='max_length', truncation=True, max_length=self.__train_shape[0], add_special_tokens=True)['input_ids']))
        return tf.reduce_sum(tf.stack(hidden_states[-4:]), axis = 0)
    
    def __train(self, x_train, y_train, x_val, y_val, epochs, model, train_metrics, val_metrics, batch_size, optimizer, loss_fn, feature_method, imbalance_handler, apply_softmax = False) -> tuple:
        f1_scores = []
        out_weights = []
        f1_train = []
        monitored_f1 = -1
        patience = 0
        train_rem = x_train.shape[0] % batch_size
        train_num_batches = x_train.shape[0] // batch_size 
        val_num_batches = x_val.shape[0] // batch_size
        val_rem = x_val.shape[0] % batch_size
        for _ in tqdm(range(epochs), desc=f'Training {model.name} with features of {feature_method} and {imbalance_handler} for imbalance handling'):
            for batch_step in range(1,train_num_batches + 1):
                i = batch_step - 1
                x = x_train[i * batch_size : batch_step * batch_size]
                x = self.__get_bert_embeddings(x)
                y = y_train[i * batch_size : batch_step * batch_size]
                self.__train_step(x, y, model, optimizer, loss_fn, train_metrics, apply_softmax)
            if train_rem != 0:
                    x = x_train[x_train.shape[0] - train_rem : x_train.shape[0]]
                    x = self.__get_bert_embeddings(x)
                    y = y_train[y_train.shape[0] - train_rem : y_train.shape[0]]
                    self.__train_step(x,y, model, optimizer, loss_fn, train_metrics, apply_softmax)

            train_precision = float(train_metrics[0].result())
            train_recall = float(train_metrics[1].result())
            train_f1_score = 2 * (train_precision * train_recall) / (train_precision + train_recall)

            f1_train.append(train_f1_score)

            # Run a validation loop at the end of each epoch.
            for batch_step in range(1, val_num_batches + 1):
                i = batch_step - 1
                x = x_val[i * batch_size: batch_step * batch_size]
                x = self.__get_bert_embeddings(x)
                y = y_val[i * batch_size: batch_step * batch_size]
                self.__val_step(x, y, model, val_metrics, apply_softmax)
            if val_rem != 0:
                x = x_val[x_val.shape[0] - val_rem: x_val.shape[0]]
                x = self.__get_bert_embeddings(x)
                y = y_val[y_val.shape[0] - val_rem: y_val.shape[0]]
                self.__val_step(x,y, model, val_metrics, apply_softmax)

            val_precision = float(val_metrics[0].result())
            val_recall = float(val_metrics[1].result())
            val_f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall)
            
            weights = []
            for layer in model.layers:
                weights.append(layer.get_weights())
            out_weights.append(weights)
            f1_scores.append(val_f1_score)

            temp = monitored_f1
            monitored_f1 = max(val_f1_score, monitored_f1)
            if temp == monitored_f1:
                patience += 1
            else:
                for metric in train_metrics:
                    print(f'{metric.name} over epoch: {float(metric.result()):.4f}')
                print(f'train_f1_score:{train_f1_score:.4f}')
                for metric in val_metrics:
                    print(f'{metric.name} over epoch: {float(metric.result()):.4f}')
                print(f'val_f1_score:{val_f1_score:.4f}')
                patience = 0
            for metric in train_metrics:
                metric.reset_states()
            for metric in val_metrics:
                metric.reset_states()
            if patience == 50:
                print('early stopping..')
                break
        return (f1_scores, out_weights, f1_train)
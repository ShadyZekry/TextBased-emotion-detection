from transformers import AutoTokenizer, TFAutoModel
import pandas as pd
import numpy as np
import psutil
import os
import tensorflow as tf
from tqdm import tqdm

class BertUtilities:
    __bert_input_cols = ['input_ids']
    __input_ids_col_index = 0
    __features_length = 768
    __is_memapped = False

    def __init__(self, task: str, dataset_name: str, bert_path: str, tokenizer_path: str, dataset_df: pd.DataFrame, text_col : str, memory_threshold_percentage = 0.4, save_embeddings = False) -> None:
        """BertUtilities\n
        Parameters:\n
            task: str required\n
            the name of the task you're working on, this will be used as a directory name when saving data to disk\n
            example: taskA\n

            ----------------

            dataset_name: str required\n
            the dataset name, this will be used as a directory name when saving data to disk\n
            example: train or validation\n

            ----------------

            bert_path: str required\n
            the path of the bert model you're using can be the repository hosting the bert model or your local path
            this will be used to load the bert model\n

            ----------------

            tokenizer_path: str required\n
            similar to bert_path this is the path to the bert tokenizer you're using and will be used to load the tokenizer\n

            ----------------

            dataset: DataFrame required\n
            the pandas dataframe containing the dataset\n

            ----------------

            text_col: str required\n
            the text column name in the dataset\n

            ----------------

            label_col: str required\n
            the label column name in the dataset\n

            ----------------

            memory_threshold_percentage: float optional\n
            percentage of the total virtual memory size, if the allocation required for the bert embeddings exceeds
            this percentage it'll be saved on disk producing a memory mapped numpy array\n
            default value is 0.4\n

            ----------------

            save_embeddings: bool optional\n
            set to false by default, and if set to true will force the bert embeddings to be saved on disk producing a memory mapped numpy array\n

            ----------------

            encoding: str optional\n
            set to utf-8 by default, this is the encoding to use for reading the dataset\n

            ----------------

            **if embeddings are saved to disk the following path format will be used:\n
            /task_name/dataset_name/dataset_name_embeddings.npy\n

            ----------------

            **the embeddings shape will be (ndataset_rows, sequence_length, features_length):\n
            where ndataset_rows is the total number of samples in the dataset, sequence_length
            will vary according to the sequence_length_mode specified, and features_length will be 768\n
            **take this into consideration for memory allocation!\n
        """
        self.__bert = TFAutoModel.from_pretrained(bert_path, output_hidden_states=True)
        self.__tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, from_tf=True)
        self.__task = task
        self.__dataset = dataset_df
        self.__text_col = text_col
        self.__dataset_name = dataset_name
        self.__memory_threshold = memory_threshold_percentage * psutil.virtual_memory().total * 1e-9
        self.__save_embeddings = save_embeddings
        self.__embeddings_holder = None
        self.__seq_length = self.__calc_seq_length()


    def tokenize_dataset(self) -> None:
        """tokenize_dataset:\n
            this method is responsible for obtaining the required inputs of the bert model (tokenizing),
            converts the text data to the required inputs of the bert model\n
            **must call this method before calling forward_to_bert
            returns: none.
        """
        temp = {self.__bert_input_cols[self.__input_ids_col_index] : []}
        for index in tqdm(self.__dataset.index, desc=f'Tokenizing {self.__dataset_name} ...'):
            temp[self.__bert_input_cols[self.__input_ids_col_index]].append(self.__tokenizer(self.__dataset.loc[index, self.__text_col], padding='max_length', truncation=True, max_length=self.__seq_length, add_special_tokens=True)[self.__bert_input_cols[self.__input_ids_col_index]])
        self.__tokenized_dataset = pd.DataFrame(temp)
    
    def forward_to_bert(self) -> None:
        """forward_to_bert:\n
            this method is responsible for obtaining the bert embeddings by forwarding the tokenized text data
            to the bert model and obtaining the output of last 4 hidden layers of the bert model and summing it\n
            the bert embeddings can be used as features for other models/classifiers.\n
            **make sure you've called tokenize_dataset before calling this method.\n
            returns: none.
        """
        if self.__tokenized_dataset.empty:
            print('dataset not tokenized, returning none.\nuse tokenize_dataset method to be able to get the bert embeddings')
            return None
        self.__set_embeddings_holder()
        for index in tqdm(self.__tokenized_dataset.index, desc=f'Getting bert embeddings for {self.__dataset_name} ..'):
            input_ids = tf.expand_dims(tf.convert_to_tensor(self.__tokenized_dataset.loc[index, self.__bert_input_cols[self.__input_ids_col_index]]), axis=0)
            hidden_states = self.__bert(input_ids)[2]
            x = tf.reshape(tf.reduce_sum(tf.stack(hidden_states[-4:]), axis=0), shape=(self.__seq_length, self.__features_length))
            self.__embeddings_holder[index] = x
        if self.__is_memapped:
            del self.__embeddings_holder

    def get_embeddings_holder(self) -> np.ndarray:
        """get_embeddings_holder:\n
            this method returns the embeddings obtained from the bert model as a numpy array,
            the numpy array could be in memory or memory mapped (on disk) depending on if the 
            required memory to allocate exceeds the available memory or if save embeddings mode is set to true\n
            **make sure you've called forward_to_bert method before calling this method.\n
            returns: ndarray or none.\n
        """
        if self.__is_memapped:
            return self.read_embeddings_from_disk()
        else:
            if type(self.__embeddings_holder) == (None):
                print('dataset not forwarded to bert returning none\nuse forward_to_bert method to be able to get the embeddings')
                return None
            else:
                return self.__embeddings_holder
    
    def read_embeddings_from_disk(self) -> np.ndarray:
        """read_embeddings_from_disk:\n
            tries to read the embeddings saved on disk using the path format:\n
            /task_name/dataset_name/dataset_name_embeddings.npy\n
            where task_name and dataset_name are the values specified in the constructor.\n
            returns: ndarray or none.
        """
        if os.path.exists(self.__get_embeddings_holder_path()):
            self.__is_memapped = True
            return np.memmap(self.__get_embeddings_holder_path(), mode='r', dtype=np.float64, shape=(self.__dataset.shape[0], self.__seq_length, self.__features_length))
        else:
            print(f'bert embeddings doesnt exist on disk, returning none.\nuse forward_to_bert method to obtain bert embeddings and make sure its in memory map mode.')
        return None

    def is_memmaped(self):
        """is_memmaped:\n
            this methods returns the is memory mapped state indicating if the embeddings are stored on disk or ram\n
            returns: bool.\n
        """
        return self.__is_memapped

    def __calc_seq_length(self):
        # max_length = 0
        # for tweet_indx in self.__dataset.index:
        #     length = len(self.__tokenizer.tokenize(self.__dataset[self.__text_col][tweet_indx]))
        #     max_length = max(max_length, length)
        return 60

    def __get_embeddings_holder_path(self) -> str:
        return f'./{self.__task}/{self.__dataset_name}/{self.__dataset_name}_embeddings.npy'

    def __set_embeddings_holder(self) -> None:
        self.__embeddings_holder = np.empty(shape=(0, self.__seq_length, self.__features_length), dtype=np.float64)
        req_mem_aloc = self.__embeddings_holder.itemsize * self.__seq_length * self.__dataset.shape[0] * self.__features_length * 1e-9
        if req_mem_aloc >= self.__memory_threshold:
            if not os.path.isdir(f'./{self.__task}'):
                os.mkdir(f'./{self.__task}')
            if not os.path.isdir(f'./{self.__task}/{self.__dataset_name}'):
                os.mkdir(f'./{self.__task}/{self.__dataset_name}')
            self.__embeddings_holder = np.memmap(self.__get_embeddings_holder_path(), mode ='w+', shape=(self.__dataset.shape[0], self.__seq_length, self.__features_length), dtype=np.float64)
            self.__is_memapped = True
            print(f'Required memory to allocate {req_mem_aloc} exceeds specified memory threshold {self.__memory_threshold} the bert embeddings will be a memory mapped data.')
        elif self.__save_embeddings:
            if not os.path.isdir(f'./{self.__task}'):
                os.mkdir(f'./{self.__task}')
            if not os.path.isdir(f'./{self.__task}/{self.__dataset_name}'):
                os.mkdir(f'./{self.__task}/{self.__dataset_name}')
            self.__embeddings_holder = np.memmap(self.__get_embeddings_holder_path(), mode ='w+', shape=(self.__dataset.shape[0], self.__seq_length, self.__features_length), dtype=np.float64)
            self.__is_memapped = True
            print(f'Save embeddings mode is set to true, embeddings will be memory mapped')
        else:
            self.__embeddings_holder = np.empty(shape=(self.__dataset.shape[0], self.__seq_length, self.__features_length), dtype=np.float64)
            self.__is_memapped = False
            print(f'Required memory to allocate {req_mem_aloc} doesnt exceed specified memory threshold {self.__memory_threshold} the bert embeddings will be stored in ram.')
    
         



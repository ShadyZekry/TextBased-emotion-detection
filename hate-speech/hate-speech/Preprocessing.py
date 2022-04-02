from pyarabic import araby
import pandas as pd
import arabicstopwords.arabicstopwords as stp
import unicodedata as ucd

preprocessed_value = 'hs_label'

usedColumns = ['id', 'tweet', preprocessed_value]

dev_dataset_file = 'hate-speech\hate-speech\dataset\OSACT2022-sharedTask-dev.csv'
test_dataset_file = 'hate-speech\hate-speech\dataset\OSACT2022-sharedTask-train.csv'

train_result_file = 'hate-speech\hate-speech\\result\OSACT2022-taskA-train.csv'
dev_result_file = 'hate-speech\\hate-speech\\result\\OSACT2022-taskA-dev.csv'

test_progress_file = 'hate-speech\hate-speech\\test_progress.txt'
train_progress_file = 'hate-speech\hate-speech\\train_progress.txt'

testing_dataset_df = pd.read_csv(
    dev_dataset_file, usecols=usedColumns, encoding='utf-8')
training_dataset_df = pd.read_csv(
    test_dataset_file, usecols=usedColumns, encoding='utf-8')


def append_to_csv(file_name, row):
    with open(file_name, 'a', encoding='utf-8') as file:
        id = row[0]
        tweet = row[1]
        label = row[2]
        # off_label = row['off_label']
        # hs_label = row['hs_label']
        # vlg_label = row['vlg_label']
        # violence_label = row['violence_label']
        # line = f'{id},{tweet},{off_label},{hs_label},{vlg_label},{violence_label}\n'
        line = f'{id},{tweet},{label}\n'
        file.write(line)


def save_progress(curr_index, is_train):
    if is_train:
        with open(train_progress_file, 'w') as file:
            line = f'progress:{curr_index}'
            file.write(line)
    else:
        with open(test_progress_file, 'w') as file:
            line = f'progress:{curr_index}'
            file.write(line)


def retrieve_progress(is_train):
    if is_train:
        with open(train_progress_file, 'r') as file:
            line = file.read()
            return int(line.split(':')[1])
    else:
        with open(test_progress_file, 'r') as file:
            line = file.read()
            return int(line.split(':')[1])


def check_stopwords(x):
    return not stp.is_stop(x)


def remove_punctuation(x):
    return ''.join(c for c in x if not ucd.category(c).startswith('P'))


train_saved_progress = retrieve_progress(True)

append_to_csv(train_result_file, usedColumns)
print("Preprocessing training dataset started")
for index in range(train_saved_progress, training_dataset_df.index.shape[0]):
    # Tokenizer
    tokenized_tweet = araby.tokenize(
        training_dataset_df['tweet'].at[index],
        morphs=[
            araby.normalize_alef,
            araby.normalize_hamza,
            araby.normalize_teh,
            araby.strip_diacritics,
            araby.strip_tatweel,
            araby.strip_harakat,
            remove_punctuation,
        ],
        conditions=[araby.is_arabicrange, check_stopwords],
    )

    training_dataset_df['tweet'].at[index] = ' '.join(tokenized_tweet)

    score_enum = training_dataset_df[preprocessed_value].at[index]
    # tweeets that include commas will result with null values, and should be execluded from the model processing
    if pd.isna(score_enum) |  (not isinstance(score_enum, str)):
        training_dataset_df[preprocessed_value].at[index] = -1
    elif 'NOT' in score_enum:
        training_dataset_df[preprocessed_value].at[index] = 0
    elif '2' in score_enum:
        training_dataset_df[preprocessed_value].at[index] = 2
    elif '3' in score_enum:
        training_dataset_df[preprocessed_value].at[index] = 3
    elif '4' in score_enum:
        training_dataset_df[preprocessed_value].at[index] = 4
    elif '5' in score_enum:
        training_dataset_df[preprocessed_value].at[index] = 5
    else:
        training_dataset_df[preprocessed_value].at[index] = 1

    # progress
    progress = (index / training_dataset_df.index.shape[0]) * 100
    progress = '{:.2f}'.format(progress)
    print(f"Preprocessing training dataset is {progress}% done.")

    processed_span = training_dataset_df.iloc[index]
    append_to_csv(train_result_file, processed_span)
    save_progress(index, True)
print("Preprocessing training dataset ended")

test_saved_progress = retrieve_progress(False)

append_to_csv(dev_result_file, usedColumns)
print("Preprocessing test dataset started")
for index in range(test_saved_progress, testing_dataset_df.index.shape[0]):
    tokenized_tweet = araby.tokenize(
        testing_dataset_df['tweet'].at[index],
        morphs=[
            araby.normalize_alef,
            araby.normalize_hamza,
            araby.normalize_teh,
            araby.strip_diacritics,
            araby.strip_tatweel,
            araby.strip_harakat,
            remove_punctuation,
        ],
        conditions=[araby.is_arabicrange, check_stopwords],
    )

    testing_dataset_df['tweet'].at[index] = ' '.join(tokenized_tweet)
    

    score_enum = testing_dataset_df[preprocessed_value].at[index]
    # tweeets that include commas will result with null values, and should be execluded from the model processing
    if pd.isna(score_enum) | (not isinstance(score_enum, str)):
        testing_dataset_df[preprocessed_value].at[index] = -1
    elif 'NOT' in score_enum:
        testing_dataset_df[preprocessed_value].at[index] = 0
    elif '2' in score_enum:
        testing_dataset_df[preprocessed_value].at[index] = 2
    elif '3' in score_enum:
        testing_dataset_df[preprocessed_value].at[index] = 3
    elif '4' in score_enum:
        testing_dataset_df[preprocessed_value].at[index] = 4
    elif '5' in score_enum:
        testing_dataset_df[preprocessed_value].at[index] = 5
    else:
        testing_dataset_df[preprocessed_value].at[index] = 1

    # progress
    progress = (index / testing_dataset_df.index.shape[0]) * 100
    progress = '{:.2f}'.format(progress)
    print(f"Preprocessing test dataset is {progress}% done.")

    processed_span = testing_dataset_df.iloc[index]
    append_to_csv(dev_result_file, processed_span)
    save_progress(index, False)
print("Preprocessing test dataset ended")

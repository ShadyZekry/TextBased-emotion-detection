import arabicstopwords.arabicstopwords as stp

def check_stopwords(x):
    return not stp.is_stop(x)

vocab_file = ''
with open('./marbert-model/vocab.txt', mode='r') as file:
    vocab_file = file.read()

count = 0
for token in vocab_file:
    if stp.is_stop(token):
        count += 1

print(count)



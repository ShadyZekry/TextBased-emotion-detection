import pandas as pd

header = 'id,tweet\n'

with open('./OSACT2022-sharedTask-test-tweets.txt', mode='r', encoding='utf-8') as test_file:
  raw_file_content = test_file.readlines()

data_dict = {'tweet':[], 'id':[]}

for line in raw_file_content:
  line_split = line.split('\t')
  data_dict['id'].append(line_split[0])
  data_dict['tweet'].append(line_split[1].replace("\n", ""))
  print(line_split[0])
  print(line_split[1])

df = pd.DataFrame(data_dict)
df.to_csv('./dataset/OSACT2022-sharedTask-test-tweets.csv', index=False, columns=['id','tweet'])

# header = 'tweet,off_label,hs_label\n'

# with open("./osact4-dataset/OSACT2020-sharedTask-train.txt", 'r', encoding='utf-8') as myfile: 
#   raw_file_content = myfile.readlines()

# with open("./osact4-dataset/OSACT2020-sharedTask-train.csv", 'w', encoding='utf-8') as csv_file:
#     csv_file.write(header)
#     data_dict = {'tweet' : [], 'off_label': [], 'hs_label': []}
#     for line in raw_file_content:
#       line_split = line.split('\t')
#       data_dict['tweet'].append(line_split[0])
#       data_dict['off_label'].append(line_split[1].replace(' ', ''))
#       data_dict['hs_label'].append(line_split[2].replace('\n', ''))
    
# df = pd.DataFrame(data_dict)
# df.to_csv('./osact4-dataset/OSACT2020-sharedTask-train.csv', index=False, columns=data_dict.keys())

# with open("./osact4-dataset/OSACT2020-sharedTask-dev.txt", 'r', encoding='utf-8') as myfile: 
#   raw_file_content = myfile.readlines()

# data_dict = { 'tweet' : [], 'off_label': [], 'hs_label':[]}
# for line in raw_file_content:
#     line_split = line.split('\t')
#     data_dict['tweet'].append(line_split[0])
#     data_dict['off_label'].append(line_split[1].replace(' ', ''))
#     data_dict['hs_label'].append(line_split[2].replace('\n', ''))
    
# df = pd.DataFrame(data_dict)
# df.to_csv('./osact4-dataset/OSACT2020-sharedTask-dev.csv', index=False, columns=data_dict.keys())
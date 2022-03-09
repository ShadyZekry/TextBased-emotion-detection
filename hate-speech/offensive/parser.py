import pandas as pd

header = 'id,tweet,off_label,hs_label,vlg_label,violence_label\n'

with open("./dataset/OSACT2022-sharedTask-train.txt", 'r', encoding='utf-8') as myfile: 
  raw_file_content = myfile.readlines()

with open("./dataset/OSACT2022-sharedTask-train.csv", 'w', encoding='utf-8') as csv_file:
    csv_file.write(header)
    data_dict = {'id' : [], 'tweet' : [], 'off_label': [], 'hs_label': [], 'vlg_label':[], 'violence_label':[]}
    for line in raw_file_content:
      line_split = line.split('\t')
      data_dict['id'].append(line_split[0])
      data_dict['tweet'].append(line_split[1])
      data_dict['off_label'].append(line_split[2].replace(' ', ''))
      data_dict['hs_label'].append(line_split[3])
      data_dict['vlg_label'].append(line_split[4])
      data_dict['violence_label'].append(line_split[5].replace('\n', ''))
    
df = pd.DataFrame(data_dict)
df.to_csv('./dataset/OSACT2022-sharedTask-train.csv', index=False, columns=data_dict.keys())


with open("./dataset/OSACT2022-sharedTask-dev.txt", 'r', encoding='utf-8') as myfile: 
  raw_file_content = myfile.readlines()

data_dict = {'id' : [], 'tweet' : [], 'off_label': [], 'hs_label': [], 'vlg_label':[], 'violence_label':[]}
for line in raw_file_content:
    line_split = line.split('\t')
    data_dict['id'].append(line_split[0])
    data_dict['tweet'].append(line_split[1])
    data_dict['off_label'].append(line_split[2].replace(' ', ''))
    data_dict['hs_label'].append(line_split[3])
    data_dict['vlg_label'].append(line_split[4])
    data_dict['violence_label'].append(line_split[5].replace('\n', ''))
    
df = pd.DataFrame(data_dict)
df.to_csv('./dataset/OSACT2022-sharedTask-dev.csv', index=False, columns=data_dict.keys())
import os
import re
import json
import pickle
import pandas as pd

data_path ='/Users/zhangwanyu/Desktop/data_project_2/bert/baidu_95_bak.csv'
save_path = '/Users/zhangwanyu/Desktop/data_project_2/bert/data_small/'
df = pd.read_csv(data_path, header=None, names=["labels", "item"], dtype=str)

label_list = list(set(df['labels'].tolist()))
print(len(label_list))
label_map = {}
for (i, label) in enumerate(label_list):
    label_map[label] = i
js_label = json.dumps(label_map)
f = open('label_id.json', 'w')
f.write(js_label)
f.close()
df = df[0:1000]
df_train = df[:int(len(df)*0.8)]
df_valid = df[int(len(df)*0.8):int(len(df)*0.9)]
df_test = df[int(len(df)*0.9):]
# Save dataset
file_set_type_list = ["train", "dev", "test"]
for file_set_type, df_data in zip(file_set_type_list, [df_train, df_valid, df_test]):

    f = open(save_path+file_set_type+'.tsv','a',encoding='utf-8')
    for line in df_data.iterrows():
        label = line[1][0]
        label_id = label_map[label]
        item = line[1][1].replace(' ','')

        text = str(label) +'***'+item+'\n'
        f.write(text)
    f.close()

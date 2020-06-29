# -*- coding: UTF-8 -*-
from gensim.models import KeyedVectors
import time
import pickle
import json
from collections import OrderedDict
from annoy import AnnoyIndex

wv_model = KeyedVectors.load('/Users/zhangwanyu/w2v.model')

def timeit(f):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = f(*args, **kwargs)
        end_time = time.time()
        print('函数运行时间为：{}'.format(end_time - start_time))
        return res

    return wrapper

@timeit
def run():
    wv_model.wv.most_similar('轮子')
    print(wv_model.wv.most_similar('维修'))
run()

############################################

word_index = OrderedDict()  # 创建一个有序字典
for counter, key in enumerate(wv_model.wv.vocab.keys()):
    word_index[key] = counter

with open('word_index.json', 'w') as f:
    json.dump(word_index, f)

wv_index = AnnoyIndex(256)  # 向量维度为256

i = 0
for key in wv_model.wv.vocab.keys():
    v = wv_model[key]
    wv_index.add_item(i, v)
    i += 1

wv_index.build(10)  # 建立索引树，10棵
wv_index.save('wv_index_build10.index')  # 保存模型到磁盘

reverse_word_index = dict([(value, key) for key, value in word_index.items()])  # 键值对翻转


@timeit
def run():
    for item in wv_index.get_nns_by_item(word_index['维修'], 11):#使用item索引号进行计算,得到最近邻的11个词
        print(reverse_word_index[item])

run()
text = '''We propose two novel model architectures for computing continuous vector repre- sentations of words from 
        very large data sets. The quality of these representations is measured in a word similarity task, and the 
        results are compared to the previ- ously best performing techniques based on different types of neural 
        networks. We observe large improvements in accuracy at much lower computational cost, i.e. it takes less 
        than a day to learn high quality word vectors from a 1.6 billion words data set. Furthermore, we show that
        these vectors provide state-of-the-art perfor- mance on our test set for measuring syntactic and semantic 
        word similarities.'''
from collections import Counter
import numpy as np
import torch
from torch import nn, optim

embedding_dim = 2
print_every = 1000
epochs = 1000
batch_size = 15
n_samples = 3
window_size = 5
freq = 0
delete_words = False


# 数据预处理
def prepocess(text, freq):
    text = text.lower()
    words = text.split()
    word_couts = Counter(words)
    trimmed_words = [word for word in words if word_couts[word] > freq]
    return trimmed_words


words = prepocess(text, freq)

# 构建词典
vocab = set(words)
vocab2int = {w: c for c, w in enumerate(vocab)}
int2vocab = {c: w for c, w in enumerate(vocab)}

# 将文本转化为数值
int_words = [vocab2int[w] for w in words]
print(int_words)
# 计算单词频次
int_word_counts = Counter(int_words)
# print(int_word_counts)
total_count = len(int_words)
word_freqs = {w: c / total_count for w, c in int_word_counts.items()}
# print(word_freqs)
# 去除频次高的词
if delete_words:
    t = le - 5
    prob_drop = {w: 1 - np.sqrt(t / word_freq[w]) for w in int_word_counts}
    train_words = [w for w in int_words if random.random() < (1 - prob_drop[w])]
else:
    train_words = int_words
# 单词分布
word_freqs = np.array(list(word_freqs.values()))
unigram_dist = word_freqs / word_freqs.sum()
noise_dist = torch.from_numpy(unigram_dist ** (0.75) / np.sum(unigram_dist ** (0.75)))
print(noise_dist)


# 获取目标词汇
def get_target(words, idx, window_size):
    target_window = np.random.randint(1, window_size + 1)
    start_point = idx - target_window if (idx - target_window) > 0 else 0
    end_point = idx + target_window
    targets = set(words[start_point:idx] + words[idx + end_point + 1])
    return list(targets)


# 批次化数据
def get_batch(words, batch_size, window_size):
    n_batches = len(words) // batch_size
    words = words[:n_batches * batch_size]
    for idx in range(0, len(words), batch_size):
        batch_x, batch_y = [], []
        batch = words[idx, idx + batch_size]
        for i in range(len(batch)):
            x = batch[i]
            y = get_target[batch, i, windows_size]
            batch_x.extend([x] * len(y))
            batch_y = extend(y)
        yield batch_x, batch_y


class SkipGramNEG(nn.Module):
    def __init__(self, n_vocab, n_embed, noise_dist):
        super().__init__()
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.noise_dist = noise_dist
        # 定义词向量
        self.in_embed = nn.Embedding(n_vocab, n_embed)
        self.out_embed = nn.Embedding(n_vocab, n_embed)
        # 初始化
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)

    def forward_inputs(self, input_words):
        input_vector = self.in_embed(inputt_words)
        return input_vectors

    def forward_output(self, output_words):
        output_vectors = self.out_embed(output_words)
        return output_vectors

    def forward_noise(self, size, n_samples):
        noise_dist = self.noise_dist
        noise_words = torch.multinomial(noise_dist, size * n_sample, replacement=Trrue)
        noise_vector = self.out_embed(noise_words).view(size, n_samples, self.n_embed)

        return noise_vector


class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_vectors, output_vectors, noise_vectors):
        batch_size.embed_size = input_vectors.shape
        # 维度转换
        input_vectors = input_vectors.view(batch_size, embed_size, 1)
        output_vectors = output_vectors.view(batch_size, 1, embed_size)

        # 正样本损失
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log()
        # out_loss shape is [batch_size,1,1] [1,n]*[n,1]=[1,1]
        out_loss = out_loss.squeeze()
        # 负样本损失
        noise_loss = torch.bmm(noise_vectors.neg().input_vectors).sigmoid().log()
        noise_loss = noise_loss.squeeze().sum(1)

        return -(out_loss + noise_loss).mean()


model = SkipGramNEG(len(vocab2int), embedding_dim, noise_dist=noise_dist)
criterion = NegativeSamplingLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

steps = 0
for e in range(epochs):
    for input_words, target_words in get_batch(train_words, batch_size, window_size):
        steps += 1
        inputs, targets = torch.LongTensor(input_words), torch.LongTensor(target_words)
        inputs_vectors = model.forward_input(inputs)
        output_vectors = model.forward_output(targets)
        size, _ = input_vectors.shape
        noise_vectors = model.forward_noise(size, n_samples)
        loss = criterion(input_vectors, output_vectors, noise_vectors)
        if steps % print_every == 0:
            print('loss:', loss)

        optimizer.zero_grad()
        loss.backword()
        optimizer.step()
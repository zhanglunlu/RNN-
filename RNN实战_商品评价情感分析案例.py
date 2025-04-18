import torch
import torch.nn as nn   # 模型
import torch.optim as optim     # 优化器



# 原始语料
texts=[
"我喜欢这个产品","这真是太棒了","我讨厌这个商品","质量差劲",
"简直太棒了","最糟糕的购买","我对这个感觉很好","不好","我对这个很满意",
"真糟糕","我喜欢使用这个","非常失望",
]
# 标签:正面为 1，负面为 0
labels=[1,1,0,0,1,0,1,0,1,0,1,0]




# 分词和构建词表
vocab=['<P>','<U>']+sorted(set(''.join(texts)))

word_to_idx={word:i for i,word in enumerate(vocab)}
idx_to_word={i:word for i,word in enumerate(vocab)}

# 对所有文件进行转换和填充
def texts_to_sequences(texts,max_length=10):
    sequences = []

    for text in texts:

        sequences.append([word_to_idx.get(char,1) for char in text])
    # 截断和填充，使其长度一致
    padded_sequences = [seq[:10]+[0]*(max_length-len(seq)) for seq in sequences]
    return padded_sequences

sequences = texts_to_sequences(texts)


# 转化为张量
train_x=torch.tensor(sequences,dtype=torch.long)
train_y=torch.tensor(labels,dtype=torch.float)
# print(train_x.shape)
# print(train_x)
# print(train_y.shape)
# print(train_y)

# 定义模型
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size,output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 词表大小，每一个词转化为几纬的向量
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        # print(embedded.shape)   # torch.Size([12, 10, 100]) ,12个句子，每个句子10个字，每个字对应100维

        _, hidden = self.rnn(embedded)

        return self.fc(hidden.squeeze(0))


# 训练
# 定义参数
vocal_size=len(vocab)
embedding_dim=100
hidden_size=256
output_dim=1
learning_rate=1e-3
epochs=10

# 模型实例化
model=SimpleRNN(vocal_size,embedding_dim,hidden_size,output_dim)
# pred_y=model(train_x)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)

# 训练
for epoch in range(epochs):
    model.train()   # 开启训练模型

    pred_y=model(train_x)
    # 注意输出值的形状
    loss=criterion(pred_y.squeeze(1),train_y)
    print(f'epoch:{epoch} loss:{loss.item()}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# 测试集
model.eval()
with torch.no_grad():
    pred_y=model(train_x)
    output=torch.round(torch.sigmoid(pred_y.squeeze(1)))
    accuracy=sum(output==train_y)/len(train_y)
    print(f'accuracy:{accuracy.item()}')


# 预测
pred_texts =['简直太棒了','真糟糕']
pred_seqs=texts_to_sequences(pred_texts)
pred_x= torch.tensor(pred_seqs, dtype=torch.long)
pred_y= model(pred_x)
pred_y= torch.round(torch.sigmoid(pred_y.squeeze(1)))
print(pred_y)
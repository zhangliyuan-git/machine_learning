import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDataset(Dataset):
    """
    GPT数据集类，用于将文本数据转换为训练GPT模型所需的输入和目标序列
    
    该类将输入文本分词后，按照指定的最大长度和步长创建输入序列和对应的目标序列，
    其中目标序列是输入序列向右偏移一位的结果，用于训练语言模型的下一个词预测任务。
    
    参数:
        txt (str): 输入的原始文本数据
        tokenizer: 分词器对象，用于将文本转换为token ID序列
        max_length (int): 每个样本序列的最大长度
        stride (int): 序列滑动窗口的步长，控制样本间的重叠程度
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 将整个文本编码为token ID序列
        token_ids = tokenizer.encode(txt)
        # 使用滑动窗口创建训练样本，每个样本包含输入序列和目标序列
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i+max_length]
            target_chunk = token_ids[i+1: i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """
        返回数据集中样本的总数
        
        返回:
            int: 数据集中样本的数量
        """
        return len(self.input_ids)
    

    def __getitem__(self, idx):
        """
        获取指定索引的数据样本
        
        参数:
            idx (int): 样本的索引
            
        返回:
            tuple: 包含输入tensor和目标tensor的元组
        """
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    """
    创建用于GPT模型训练的数据加载器
    
    该函数封装了数据集创建和DataLoader初始化的过程，提供了一站式的数据加载解决方案。
    
    参数:
        txt (str): 输入的原始文本数据
        batch_size (int): 每个批次包含的样本数量，默认为4
        max_length (int): 每个样本序列的最大长度，默认为256
        stride (int): 序列滑动窗口的步长，默认为128
        shuffle (bool): 是否在每个epoch打乱数据，默认为True
        drop_last (bool): 是否丢弃最后一个不完整的批次，默认为True
        num_workers (int): 用于数据加载的子进程数量，默认为0（使用主进程）
        
    返回:
        DataLoader: 配置好的PyTorch数据加载器对象
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader

# 读取原始文本文件内容
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 定义测试用的简单token ID序列
input_ids = torch.tensor([2, 3, 5, 1])
# 设置词汇表大小和嵌入维度
vocab_size = 6
output_dim = 3
# 设置随机种子以确保结果可复现
torch.manual_seed(123)
# 创建嵌入层并打印权重和嵌入结果
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)
print(embedding_layer(torch.tensor([3])))
print(embedding_layer(input_ids))

# 设置实际GPT模型使用的参数
vocab_size = 50257
output_dim = 256
# 创建token嵌入层
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
max_length = 4
# 创建数据加载器
dataloader = create_dataloader(
    raw_text, batch_size=8, max_length=max_length,
   stride=max_length, shuffle=False
)
# 获取一个批次的数据并打印相关信息
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)
# 对输入进行token嵌入
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

# 设置上下文长度并创建位置嵌入层
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
# 生成位置嵌入向量
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

# 将token embeddings与position embeddings相加得到最终输入嵌入
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
import re
import urllib.request


class SimpleTokenizer:
    """词元编码解码"""
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # 用<|unk|>表示未知词元
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

# 下载源文件
url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

# 进行源文件读取
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("源文件字符长度：", len(raw_text))
# 按照标点符号进行分割词元
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print("该文本词元数：", len(preprocessed))
#进行去重并排序
all_words = sorted(set(preprocessed))
# 添加特殊词元
all_words.extend(["<|endoftext|>", "<|unk|>"])
vocab_size = len(all_words)
print("去重并排序之后的词元数：", vocab_size)
# 创建词元字典
vocab = {token:integer for integer,token in enumerate(all_words)}
# 测试词元编码解码
tokenizer = SimpleTokenizer(vocab)
text = """"It's the last he painted, you know,"
       Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))
# 测试特殊词元编码解码
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)
print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))
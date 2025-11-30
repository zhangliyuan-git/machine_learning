import tiktoken

# 使用gpt2编码器
tokenizer = tiktoken.get_encoding("gpt2")

# 读取文本文件内容
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 对文本进行编码
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

# 取编码后的文本从第50个字符开始的部分作为样本
enc_sample = enc_text[50:]

# 设置上下文大小
context_size = 4

# 阶段1: 打印编码后的数字序列形式的输入-目标对
print("=== 编码序列形式的输入-目标对 ===")
for i in range(1, context_size+1):
    context = enc_sample[:i]      # 输入:前i个token
    desired = enc_sample[i]       # 目标:第i+1个token
    print(context, "---->", desired)

# 阶段2: 打印解码后的人类可读形式的输入-目标对
print("\n=== 解码后的人类可读输入-目标对 ===")
for i in range(1, context_size+1):
   context = enc_sample[:i]       # 输入:前i个token
   desired = enc_sample[i]        # 目标:第i+1个token
   print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
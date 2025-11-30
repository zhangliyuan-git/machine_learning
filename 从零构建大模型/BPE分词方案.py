import tiktoken

# 使用gpt2编码器
tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)
# 文本编码
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
# 文本解码
strings = tokenizer.decode(integers)
print(strings)
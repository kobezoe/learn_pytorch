#  正则表达式

# try:
#     f = open("文件处理.txt", "r")
#     lines = f.readlines()
#     print(lines)
# finally:
#     f.close()


# 会自动关闭文件资源
sents = []
with open("../Python基础笔记/文件处理.txt", "r") as f:
    for line in f:
        line = line.strip();
        tokens = line.split('。')
        # for token in tokens:
        #     if len(tokens)>0:
        #         sents.append(token)
        res = list(filter(lambda x: len(x) > 0, tokens))
        sents.extend(res)
# print(sents)
print(sents)

# lens = []
# for sent in sents:
#     lens.append(len(sent))
# print(lens)
# # 函数式编程，列表生成式
# lens = [len(sent) for sent in sents]
# print(lens)


# regex = '文件'  #[百山]：匹配百字或者山字   ‘2[0-9]{3}’:2000年之后的年份 [0-9]+:前面这个数字至少出现一次
# for sent in sents:
#     if re.search(regex, sent) is not None:
#         print(sent)

with open('../Python基础笔记/文件处理copy.txt', 'w', encoding='utf-8') as f:
    # writelines() 的参数必须是一个列表，write()的参数是一个字符串
    sents = [sent + '\n' for sent in sents]
    f.writelines(sents)

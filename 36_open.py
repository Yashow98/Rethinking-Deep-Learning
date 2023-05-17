# @Author  : Yashowhoo
# @File    : 36_open.py
# @Description :

with open('./cn_stopwords.txt', encoding='utf-8') as f:
    sw = [line.strip() for line in f]
    print(sw)



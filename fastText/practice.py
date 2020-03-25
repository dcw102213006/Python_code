from gensim.models import FastText
sentences = [["你", "是", "誰"], ["我", "是", "台灣人"]]

model = FastText(sentences,  size=4, window=3, min_count=1, iter=10,min_n = 3 , max_n = 6,word_ngrams = 0)
print(model['你'])  # 詞向量獲得的方式
print(model.wv['你']) # 詞向量獲得的方式

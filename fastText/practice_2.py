import fasttext
# skipgram model


model = fasttext.train_supervised(input="data/data.txt")
model.save_model("model_cooking.bin")
model.predict("Industry")

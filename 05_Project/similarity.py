import numpy as np
import pandas as pd
from transformers import pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity_tfidf(sent1, sent2):
	tfidf_vectorizer = TfidfVectorizer()
	tfidf_matrix = tfidf_vectorizer.fit_transform([sent1, sent2])
	similarity_matrix = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
	return similarity_matrix[0][0]

model_name = "patrickvonplaten/bert2bert_cnn_daily_mail"
summarizer = pipeline("summarization", model=model_name, device=-1)

df = pd.read_csv("subset512.csv")
all_sims = []
for idx, row in df.iterrows():
	input_text = row["TEXT"]
	max_inp = len(input_text.split())
	unsup_summ = str(row["Summary"])
	min_len = len(unsup_summ.split())
	result = summarizer(input_text, min_length=min_len,max_length=max_inp, do_sample=False, truncation=True)
	gen_summ = result[0]["summary_text"]
	sim = cosine_similarity_tfidf(unsup_summ,gen_summ)
	all_sims.append(sim)

	print("one done")
	if (idx + 1) % 10 == 0:
		print(f"{idx + 1} summaries done")

np.save(f"{model_name.replace('/','-')}.npy",all_sims)
print(np.mean(all_sims))

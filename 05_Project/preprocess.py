import os
import re
import numpy as np
import spacy
import pandas as pd


def preprocess_strings(text):

    text = re.sub("\[\*\*[\d]{1,4}-[\d]{1,2}(-[\d]{1,2})?\*\*\]", " ", text)
    text = re.sub("\[\*\*[\d]{4}\*\*\]"," ",text)
    text = re.sub(r"\n{2,}",". ",text)
    text = text.replace("\n"," ")
    text = re.sub("\[\*\*[^\]]+\]"," ",text)
    text = text.replace("df'd","diagnosed")
    text = text.replace("tx'd","treated")
    text = re.sub("[^a-zA-Z0-9./\-: ]"," ",text)
    text = re.sub("\s+"," ",text)
    text = text.lower()

    return text


if __name__ == "__main__":

    df = pd.read_csv("sample.csv")

    texts = df["TEXT"].values

    text = texts[0]
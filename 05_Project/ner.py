from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

def main(example, tokenizer, model):
    # tokenizer = AutoTokenizer.from_pretrained("sschet/bert-base-uncased_clinical-ner")
    # model = AutoModelForTokenClassification.from_pretrained("sschet/bert-base-uncased_clinical-ner")

    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    # example = "The patient has severe hypertension and diabetes"

    ner_results = nlp(example)
    # print(ner_results)

    entities = []

    def remove_hashIs():
        for i, dicts in enumerate(ner_results):
            if dicts["word"].startswith("#"):
                ent = dicts["entity"]
                if ent.startswith("B"):
                    if ent == "B-problem":
                        ner_results[i]["entity"] = "I-problem"
                    elif ent == "B-treatment":
                        ner_results[i]["entity"] = "I-treatment"
                    else:
                        ner_results[i]["entity"] = "I-test"


    def add_next_word(i, entity, ent_label):
        while i < n and ner_results[i]["entity"] == "I-" + ent_label:
            next_word = ner_results[i]["word"]
            if next_word.startswith("#"):
                next_word = next_word.replace("#","")
                entity += next_word
            else:
                entity += " " + next_word.replace("#","")
            i += 1            
        return i, entity

    remove_hashIs()
    i = 0
    n = len(ner_results)

    while i < n:

        dicts = ner_results[i]

        if dicts["entity"] == "B-problem":
            
            entity = dicts["word"].replace("#","")
            i, entity = add_next_word(i + 1, entity, "problem")
            entities.append(entity)

        elif dicts["entity"] == "B-treatment":
            entity = dicts["word"].replace("#","")
            i, entity = add_next_word(i + 1, entity, "treatment")
            entities.append(entity)

        elif dicts["entity"] == "B-test":
            entity = dicts["word"].replace("#","")
            i, entity = add_next_word(i + 1, entity, "test")
            entities.append(entity)
        else:
            entities.append(dicts["word"].replace("#",""))
            i += 1

    return entities

if __name__ == "__main__":
    print(main("RADIOLOGIC STUDIES:  Radiologic studies also included a chest CT, which confirmed cavitary lesions in the left lung apex consistent with infectious process/tuberculosis.  This also moderate-sized left pleural effusion. HEAD CT:  Head CT showed no intracranial hemorrhage or mass effect, but old infarction consistent with past medical history. ABDOMINAL CT:  Abdominal CT showed lesions of T10 and sacrum most likely secondary to osteoporosis. These can be followed by repeat imaging as an outpatient. [**First Name8 (NamePattern2) **] [**First Name4 (NamePattern1) 1775**] [**Last Name (NamePattern1) **], M.D.  [**MD Number(1) 1776**] Dictated By:[**Hospital 1807**] MEDQUIST36 D:  [**2151-8-5**]  12:11 T:  [**2151-8-5**]  12:21 JOB#:  [**Job Number 1808**]"))
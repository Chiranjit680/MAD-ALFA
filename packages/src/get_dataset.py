from datasets import load_dataset
if __name__ == "__main__":
    ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial")
    ds.save_to_disk("pubmedqa_dataset")
    ds["train"].to_csv("pubmedqa_train.csv")
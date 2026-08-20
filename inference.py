import pickle
import torch

from models.gru import GRUClassifier
from utils import load_checkpoint, predict
from data import tokenize


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    with open("best_model/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    print("vocab size: ", len(vocab))

    model = GRUClassifier(vocab_size=len(vocab),
                          embedding_dim=128,
                          hidden_dim=128,
                          gru_layers=3,
                          dropout=0.3,
                          pad_idx=vocab.pad)

    load_checkpoint(model, "best_model/best_model_gru_3.pt", device)
    model.to(device)

    # 测试文本, 也可以自己输入
    texts = ["This movie is fantastic. The story is amazing.",
             "This is the worst movie, I have ever watched.",
             "The movie is acceptable, but nothing special."]

    # 预测
    results = predict(model, texts, vocab, tokenize, device)

    for text, result in zip(texts, results):
        print("=" * 50)

        print(f"Text:\n{text}")
        print(f"Prediction:\n{result['label']}")
        print(f"Probability:\n{result['probability']:.4f}")

if __name__ == "__main__":
    main()

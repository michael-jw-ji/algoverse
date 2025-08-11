from utils import load_model
import torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(revision="step0")

if __name__ == "__main__":
    main()

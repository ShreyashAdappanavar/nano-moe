from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from typing import List
import os

def train_tokenizer(file_path: List[str], save_path: str, max_vocab_size: int = 10_000):

    tokenizer = Tokenizer(model=models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False) # type: ignore
    tokenizer.decoder = decoders.ByteLevel() # type: ignore

    trainer = trainers.BpeTrainer(vocab_size=10_000, show_progress=True, 
                                special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"], 
                                initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

    tokenizer.train(file_path, trainer=trainer)

    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path}")

if __name__ == "__main__":
    file_path = os.path.join("data", "tokenizer.txt")    
    train_tokenizer(file_path=[file_path], save_path='tokenizer.json')


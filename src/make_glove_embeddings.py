# Libraries
import numpy as np
from pathlib import Path
#import pandas as pd
#import pickle
import MEN as men

# Useful paths
CWD = Path.cwd()
path_data = CWD / "data"


def load_embeddings(embeddings_path: Path) -> dict:
    """
    Opens the word embeddings file.
    ------------------
    Parameters:
        - embeddings_path: a Path object that indicates where the embeddings.
         are saved.
    ------------------
    Output:
        - embeddings_dict: dicto f the format {token: token_embedding, ...}.
    """
    embeddings_dict = {}
    with open(embeddings_path, mode="r", encoding="UTF-8") as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector

    print("succesfully loaded embeddings file!")

    return embeddings_dict


def main():
    """
    A function that takes a raw embeddings file and organizes it in a dict where each
    experimental category has the embeddings for its target words.
    """
    # Load MEN data
    pairs, humans = men.readMEN(path_data / "MEN_dataset_natural_form_full")

    # Get a list of the words used in MEN
    word_list = [i for pair in pairs for i in pair]
    word_list = [*set(word_list)]

    # Load Glove embeddings
    embeddings = load_embeddings(path_data / "glove.6B.300d.txt")
    glove_dict = {word: embeddings[word] for word in word_list}

    # Save file
    np.savez(path_data / "MEN_glove_data.npz", **glove_dict)    
    print("Files succesfully saved!")


if __name__ == "__main__":
    main()

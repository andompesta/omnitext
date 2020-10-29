from datasets import load_dataset
import numpy as np


if __name__ == '__main__':
    dataset = load_dataset('imdb')
    print(dataset)
    train = dataset['train']

    lens = []
    for t in train:
        lens.append(len(t['text'].split()))

    print(np.mean(lens))
    print(np.std(lens))
    print(np.max(lens))
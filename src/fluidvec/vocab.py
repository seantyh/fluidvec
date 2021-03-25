import pickle

class Vocabulary:
    def __init__(self):
        self.vocab = {"<UNK>": 0, "<PAD>": 1}
        self.freq = {v: 0 for v in self.vocab.values()}        
        self.make_rev_vocab()
        self.is_dirty = False
    
    def __repr__(self):
        return f"<Vocabulary: {len(self.vocab)} term(s)>"
    
    def __len__(self):
        return len(self.vocab)
    
    def add(self, term):
        if term not in self.vocab:
            term_idx = len(self.vocab)
            self.vocab[term] = term_idx
        else:
            term_idx = self.vocab[term]
        self.freq[term_idx] = self.freq.get(term_idx, 0) + 1
        self.is_dirty = True
    
    def make_rev_vocab(self):
        self.rev_vocab = {v: k for k, v in self.vocab.items()}
        self.is_dirty = False
        
    def encode(self, term):
        if not isinstance(term, str):
            raise ValueError("Expect term as string")
        return self.vocab.get(term, 0)
    
    def decode(self, index):
        if not isinstance(index, int):
            raise ValueError("Expect index as integer")
        if not self.rev_vocab or self.is_dirty:
            self.make_rev_vocab()
        return self.rev_vocab.get(index, "<UNK>")
    
    def save(self, fpath):
        with open(fpath, "wb") as fout:
            pickle.dump((self.vocab, self.freq), fout)
    
    @classmethod
    def load(self, fpath):
        with open(fpath, "rb") as fin:
            vocab, freq = pickle.load(fin)
        loaded = Vocabulary()
        loaded.vocab = vocab
        loaded.freq = freq
        loaded.make_rev_vocab()
        return loaded
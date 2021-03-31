from pathlib import Path
from gensim.models import KeyedVectors
import torch
import numpy as np
from .model import FluidVecSG
from .vocab import VocabSet

def create_fluid_keyvec(model_path):
    model_dir = Path(model_path)
    vs = VocabSet.load(model_dir)
    hypers = torch.load(model_dir/"hypers.pt")
    new_model = FluidVecSG(**hypers)
    new_model.load_state_dict(torch.load(model_dir/"model.pt", map_location=torch.device("cpu")))    

    words = sorted(list(vs.word_vocab.vocab.keys()), key=vs.word_vocab.vocab.get)
    chars = sorted(list(vs.char_vocab.vocab.keys()), key=vs.char_vocab.vocab.get)    
    word_vecs = new_model.word_emb.weight.detach().numpy()
    char_vecs = new_model.char_emb.weight.detach().numpy()

    wvocab = vs.word_vocab    
    wfreq = {w: wvocab.freq[i] for w, i in wvocab.vocab.items()}
    assert len(words) == word_vecs.shape[0]
    assert len(chars) == char_vecs.shape[0]
    fkv = FluidKeyedVectors(hypers["dim"], wfreq)
    fkv.add_words(words, word_vecs)
    assert fkv.vectors.shape[0] == len(words)
    fkv.add_chars(chars, char_vecs)    
    assert fkv.vectors.shape[0] == len(words) + len(chars)

    return fkv

class FluidKeyedVectors(KeyedVectors):
    def __init__(self, dim, wfreq):
        super(FluidKeyedVectors, self).__init__(dim)
        self.n_char = 0
        self.n_word = 0
        self.chars = []
        self.words = []
        self.wfreq = wfreq        
        self.dim = dim
    
    def __repr__(self):
        return (f"<FluidKeyedVectors: {self.n_word} words, "
                f"{self.n_char} chars, {self.dim} dimensions>")

    def add_words(self, words, vecs):
        self.add_vectors(words, vecs)
        self.words = words
        self.n_word = len(words)
        
    def add_chars(self, chars, vecs):
        new_chars = []
        for char_x in chars:
            char_x = char_x.replace('_', '/')
            if not char_x.startswith('/'):
                char_x = '.' + char_x
            if not char_x.endswith('/'):
                char_x = char_x + '.'
            new_chars.append(char_x)
        self.add_vectors(new_chars, vecs)
        self.chars = new_chars
        self.n_char = len(new_chars)
    
    def most_similar(self, positives, negatives=[], 
            topn=10, use_char_vec=False, word_only=False, min_wfreq=10):
                
        if use_char_vec and isinstance(positives, str):            
            print("in word vocab: ", positives in self)
            char_vec, chars = self.char_vec(positives)
            positives = [char_vec]            
            print("char tokens: ", chars)
        
        if word_only:
            topn_offset = self.n_char + topn
        else:
            topn_offset = topn

        ret = super(FluidKeyedVectors, self)\
              .most_similar(positives, negatives, topn=topn_offset)
        if word_only:
            return [(t, s, self.wfreq[t]) for t, s in ret 
                     if t and '/' not in t 
                     and '.' not in t 
                     and self.wfreq.get(t, 0) > min_wfreq][:topn]
        else:
            return ret
        
    def char_vec(self, txt):        
        chars = []
        for i, char_x in enumerate(txt):
            if len(txt) == 1:
                chars.append(f"/{char_x}/")                
                break
            if i == 0:
                chars.append(f"/{char_x}.")
            elif i == len(txt)-1:
                chars.append(f".{char_x}/")
            else:
                chars.append(f".{char_x}.")
                
        mat = np.zeros((len(chars), self.vectors.shape[1]))
        for char_i, char_x in enumerate(chars):
            mat[char_i, :] = self.get_vector(char_x)
        return mat.mean(0), chars
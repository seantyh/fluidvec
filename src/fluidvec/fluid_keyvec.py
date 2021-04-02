from pathlib import Path
from gensim.models import KeyedVectors
import torch
import numpy as np
from .model import FluidVecSG
from .vocab import VocabSet

def create_fluid_keyvec(model_char_path, model_compo_path=None):
    char_model_dir = Path(model_char_path)
    vs = VocabSet.load(char_model_dir)
    hypers = torch.load(char_model_dir/"hypers.pt")
    new_model = FluidVecSG(**hypers)
    new_model.load_state_dict(torch.load(char_model_dir/"model.pt", map_location=torch.device("cpu")))

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

    # load component model
    if model_compo_path:
        compo_model_dir = Path(model_compo_path)
        hypers = torch.load(compo_model_dir/"hypers.pt")
        compo_model = FluidVecSG(**hypers)
        compo_model.load_state_dict(torch.load(compo_model_dir/"model.pt",
                        map_location=torch.device("cpu")))
        compos = sorted(list(vs.compo_vocab.vocab.keys()), key=vs.compo_vocab.vocab.get)
        compo_vecs = compo_model.compo_emb.weight.detach().numpy()        
        assert len(compos) == compo_vecs.shape[0]
        fkv.add_compos(compos, compo_vecs)
        # account for duplicated <UNK> and <PAD> in words and compos        
        assert fkv.vectors.shape[0] == len(words)+len(chars)+len(compos)-2

    return fkv

class FluidKeyedVectors(KeyedVectors):
    def __init__(self, dim, wfreq):
        super(FluidKeyedVectors, self).__init__(dim)
        self.n_char = 0
        self.n_word = 0
        self.n_compo = 0
        self.words = []
        self.chars = []
        self.compos = []
        self.wfreq = wfreq
        self.dim = dim
        self.ctree = None

    def __repr__(self):
        return (f"<FluidKeyedVectors: {self.n_word} words, "
                f"{self.n_char} chars, {self.n_compo} compos, {self.dim} dimensions>")

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

    def add_compos(self, compos, vecs):
        self.add_vectors(compos, vecs)
        self.compos = compos
        self.n_compo = len(compos)

    def most_similar(self, positives, negatives=[],
            topn=10, use_char_vec=False, use_compo_vec=False,
            word_only=False, no_compo=False, min_wfreq=10):

        if use_char_vec and isinstance(positives, str):
            use_compo_vec = False
            print("in word vocab: ", positives in self)
            char_vec, chars = self.char_vec(positives)
            positives = [char_vec]
            print("char tokens: ", chars)

        if use_compo_vec and isinstance(positives, str):
            compo_vec, compos = self.compo_vec(positives)
            positives = [compo_vec]
            print("compo tokens: ", compos)

        if word_only:
            topn_offset = self.n_char + self.n_compo + topn
        else:
            topn_offset = topn        

        ret = super(FluidKeyedVectors, self)\
              .most_similar(positives, negatives, topn=topn_offset)

        if word_only:
            ret = [(t, s, self.wfreq[t]) for t, s in ret
                     if t and '/' not in t
                     and '.' not in t
                     and self.wfreq.get(t, 0) > min_wfreq][:topn]
        elif no_compo:
            ret = [(t, s) for t, s in ret if "-" not in t]

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
    
    def compo_vec(self, txt):
        if not self.ctree:
            try:
                from CompoTree import ComponentTree
            except ImportError as ex:
                print("Cannot import CompoTree, maybe set the package path?")
            self.ctree = ComponentTree.load()
        
        compo_list = self.serialize_components(txt)
            
        mat = np.zeros((len(compo_list), self.vectors.shape[1]))
        for compo_i, compo_x in enumerate(compo_list):
            mat[compo_i, :] = self.get_vector(compo_x)
        return mat.mean(0), compo_list

    def serialize_components(self, txt):
        compo_list = []
        for ch in txt:
            try:
                compos = self.ctree.query(ch, use_flag="shortest", max_depth=1)[0]
                if isinstance(compos, str):
                    continue
                idc = compos.idc
                for i, c in enumerate(compos.components()):                    
                    compo_list.append(f"{idc}{i}-{str(c)}")        
            except Exception:
                continue
        
        return compo_list

        




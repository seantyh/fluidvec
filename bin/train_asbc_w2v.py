from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

sentences = LineSentence("../data/asbc5_plaintext.txt")
model = Word2Vec(sentences=sentences, vector_size=100, window=4, min_count=2)
import torch
import torch.nn as nn

class FluidVecSG(nn.Module):
    def __init__(self, n_word, n_char, n_compo, dim=300, winsize=4, 
            weights=None, neg_sample=.75):
        super(FluidVecSG, self).__init__()
        self.word_emb = nn.Embedding(n_word, dim)
        self.char_emb = nn.Embedding(n_char, dim)
        self.compo_emb = nn.Embedding(n_compo, dim)
        if not weights:
            self.weights = torch.ones(n_word)
        else:
            self.weights = torch.tensor(weights)
        self.n_neg_sample = int(n_word ** neg_sample)

    def forward(self, batch_data):
        return self.loss(batch_data)
        
    def transform_batch_data(self, batch_data):
        # in skip-gram, we use target chars and compos to predict context words
        tgt_data = [x[0] for x in batch_data]  # Entry x batch_size
        ctx_data = [x[1] for x in batch_data]  # Entry x winsize x batch_size
        tgt = torch.stack([self.get_prediction_vector(x) for x in tgt_data])

        ctx_batch = []
        for batch_x in ctx_data:
            ctx_win = []
            for ctx_entry in batch_x:
                ctx_win.append(self.get_word_vector(ctx_entry))
            ctx_batch.append(torch.stack(ctx_win))
        ctx = torch.stack(ctx_batch)
        return {
            "tgt": tgt, # batch_size x emb_dim
            "ctx": ctx  # batch_size x winsize x emb_dim
        }

    def get_word_vector(self, entry):
        return self.build_embedding([entry["word"]], self.word_emb)

    def get_prediction_vector(self, entry):
        compo_vec = self.get_compo_vector(entry)
        char_vec = self.get_char_vector(entry)
        return compo_vec + char_vec

    def get_char_vector(self, entry):
        return self.build_embedding(entry["chars"], self.char_emb)

    def get_compo_vector(self, entry):
        return self.build_embedding(entry["compos"], self.compo_emb)

    def build_embedding(self, idx_list, emb):
        assert max(x for x in idx_list) < emb.num_embeddings
        idx_list = [x for x in idx_list if x != 1]
        if idx_list:
            vec = emb(torch.tensor(idx_list)).sum(0)
        else:
            vec = torch.zeros(emb.embedding_dim)
        return vec
    
    def sample(self, num_sample):
        return 

    def loss(self, batch_data):
        vec_dict = self.transform_batch_data(batch_data)
        tgt = vec_dict["tgt"]
        ctx = vec_dict["ctx"]

        batch_size = ctx.size(0)
        win_size = ctx.size(1)
        n_noise = batch_size * win_size * self.n_neg_sample
        draw = torch.multinomial(self.weights, n_noise, True)
        noise = draw.view(batch_size, win_size, self.n_neg_sample)
        
        log_target = (tgt.unsqueeze(1) * ctx).sum(2).sigmoid().log().sum()
        sum_log_noise = ((tgt.unsqueeze(1).unsqueeze(1)*noise)
                        .neg().sum(3).sigmoid()+1e-32).log().sum()
        loss = log_target + sum_log_noise

        return -loss / batch_size


import torch
import torch.nn as nn

class FluidVecSG(nn.Module):
    def __init__(self, n_word, n_char, n_compo, dim=300,
            weights=None, n_neg_sample=5, use_cuda=False):
        super(FluidVecSG, self).__init__()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.n_word = n_word
        self.n_char = n_char
        self.n_compo = n_compo
        self.word_emb = nn.Embedding(n_word, dim)
        self.char_emb = nn.Embedding(n_char, dim)
        self.compo_emb = nn.Embedding(n_compo, dim)
        self.dim = dim        

        if weights is None:
            self.weights = torch.ones(n_word)
        elif isinstance(weights, torch.Tensor):
            self.weights = weights
        else:
            self.weights = torch.tensor(weights, dtype=torch.float32)
        self.n_neg_sample = n_neg_sample
        print("device: ", self.device)
        print("n_neg_sample: ", self.n_neg_sample)

    def hyperparameters(self):
        hypers = dict(
            weights=self.weights,
            n_word=self.n_word,
            n_char=self.n_char,
            n_compo=self.n_compo,
            n_neg_sample=self.n_neg_sample,
            dim=self.dim
        )
        return hypers
                
    def forward(self, batch_data):
        return self.loss(batch_data)

    def transform_batch_data(self, batch_data):
        # in skip-gram, we use target chars and compos to predict context words
        tgt_data = [x[0] for x in batch_data]  # Entry x batch_size
        ctx_data = [x[1] for x in batch_data]  # Entry x winsize x batch_size
        tgt = torch.stack([self.get_prediction_vector(x) for x in tgt_data])

        ctx_batch = []
        ctx_mask = []
        for batch_x in ctx_data:
            batch_mask = []
            ctx_win = []
            for ctx_entry in batch_x:
                if ctx_entry["word"] < 2:
                    batch_mask.append(0)
                else:
                    batch_mask.append(1)
                ctx_win.append(self.get_word_vector(ctx_entry))
                            
            ctx_batch.append(torch.stack(ctx_win))
            ctx_mask.append(torch.tensor(batch_mask, dtype=torch.long))
        ctx_mask = torch.stack(ctx_mask)
        ctx = torch.stack(ctx_batch)

        return {
            "tgt": tgt.to(self.device), # batch_size x emb_dim
            "ctx": ctx.to(self.device),  # batch_size x winsize x emb_dim
            "ctx_mask": ctx_mask.to(self.device)   # batch_size x winsize
        }

    def get_word_vector(self, entry):    
        word_idx = torch.tensor(entry["word"]).to(self.device)
        return self.word_emb(word_idx)


    def get_prediction_vector(self, entry):
        if self.n_compo:
            compo_vec = self.get_compo_vector(entry)
            n_compo = max(len(entry["compos"]), 1)
        else:
            compo_vec = torch.zeros(self.dim).to(self.device)
            n_compo = 0

        if self.n_char:
            char_vec = self.get_char_vector(entry)
            n_char = max(len(entry["chars"]), 1)
        else:
            char_vec = torch.zeros(self.dim).to(self.device)
            n_char = 0

        pred_vec = (compo_vec + char_vec) / (n_compo + n_char)
        if (compo_vec+char_vec).norm() < 1e-5:
            print("zero pred vec, why??")
        return pred_vec

    def get_char_vector(self, entry):
        if not entry["chars"]:
            return self.build_embedding([1], self.char_emb)
        else:
            return self.build_embedding(entry["chars"], self.char_emb)

    def get_compo_vector(self, entry):
        if not entry["compos"]:
            return self.build_embedding([1], self.compo_emb)
        else:
            return self.build_embedding(entry["compos"], self.compo_emb)

    def build_embedding(self, idx_list, emb):
        assert max(x for x in idx_list) < emb.num_embeddings                
        vec = emb(torch.tensor(idx_list).to(self.device)).sum(0)                    
        vec = vec.to(self.device)
        return vec

    def loss(self, batch_data):
        # negative sampling implementation following
        # https://github.com/kefirski/pytorch_NEG_loss
        vec_dict = self.transform_batch_data(batch_data)
        tgt = vec_dict["tgt"] # (batch_size, dim)
        ctx = vec_dict["ctx"] # (batch_size, win_size, dim)
        mask = vec_dict["ctx_mask"] # (batch_size, win_size)

        batch_size = ctx.size(0)
        win_size = ctx.size(1)
        n_noise = batch_size * win_size * self.n_neg_sample
        draw = torch.multinomial(self.weights, n_noise, True)
        noise = draw.view(batch_size, win_size*self.n_neg_sample)
        noise = noise.to(self.device)
        noise_vec = self.word_emb(noise)  # (batch_size, win_size*n_neg, dim)

        log_target = ((tgt.unsqueeze(1) * ctx).sum(2).sigmoid()+1e-5).log()
        log_target = log_target * mask
        log_target_val = log_target.sum()        

        sum_log_noise = ((tgt.unsqueeze(1)*noise_vec)
                        .neg().sum(2).sigmoid()+1e-5).log()
        sum_log_noise = (sum_log_noise.view(batch_size, win_size, -1)
                         * mask.unsqueeze(2)).view(batch_size, -1)
        sum_log_noise_val = sum_log_noise.sum()        
        loss = log_target_val + sum_log_noise_val
        
        # return -loss / batch_size, vec_dict, log_target, sum_log_noise
        return -loss / batch_size


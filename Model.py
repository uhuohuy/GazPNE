import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pdb

def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer, num_embeddings, embedding_dim
class SelfAttentionLSTM(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )
    
    def forward(self, encoder_outputs):
        # encoder_outputs = [batch size, sent len, hid dim]
        energy = self.projection(encoder_outputs)
        # energy = [batch size, sent len, 1]
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # weights = [batch size, sent len]
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        # outputs = [batch size, hid dim]
        return outputs, weights
class SelfAttention(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(filters, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )
    
    def forward(self, encoder_outputs):
        # encoder_outputs = [batch size, sent len, filters]
        energy = self.projection(encoder_outputs)
        # energy = [batch size, sent len, 1]
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # weights = [batch size, sent len]
        outputs = (encoder_outputs * weights.unsqueeze(-1))
        # outputs = [batch size, sent len, filters]
        return outputs

# only use 3-gram filter and one 
class C_LSTM(nn.Module):
    def __init__(self, weights_matrix, n_filters, filter_size, hidden_dim ,output_dim, flex_feat_len, dropout):
        super().__init__()
        self.embedding,self.vocab_size,self.embedding_dim = create_emb_layer(weights_matrix, True)
        self.conv = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_size,self.embedding_dim+flex_feat_len)) 
        self.lstm = nn.LSTM(n_filters, hidden_dim, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hc): 
        #x = [sent len, batch size]
        #x = x.permute(1, 0)   
        #x = [batch size, sent len]
        embedded = self.embedding(x)    
        embedded = torch.cat((embedded,hc),2) 
        #embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        #embedded = [batch size, 1, sent len, emb dim]
        conved = F.relu(self.conv(embedded)).squeeze(3)
        #conved = [batch size, n_filters, sent len - filter_size]
        x = conved.permute(2, 0, 1)
        #x = [sent len- filter_size, batch size, n_filters]
        output, (hidden, cell) = self.lstm(x)
        #output = [sent len - filter_size, batch size, hid_dim]
        #hidden = [1, batch size, hid dim]
        hidden = hidden.squeeze(0)
        #hidden = [batch size, hid dim]
        return self.fc(self.dropout(hidden))
    def predict(self,embedded):
        embedded = embedded.unsqueeze(1)
        #embedded = [batch size, 1, sent len, emb dim]
        conved = F.relu(self.conv(embedded)).squeeze(3)
        #conved = [batch size, n_filters, sent len - filter_size]
        x = conved.permute(2, 0, 1)
        #x = [sent len- filter_size, batch size, n_filters]
        output, (hidden, cell) = self.lstm(x)
        #output = [sent len - filter_size, batch size, hid_dim]
        #hidden = [1, batch size, hid dim]
        hidden = hidden.squeeze(0)
        return self.fc(self.dropout(hidden))

# use only 3-gram filter and bidirection
class C_LSTMAttention(nn.Module):
    def __init__(self, weights_matrix, n_filters, filter_size, bidirectional, hidden_dim , output_dim, flex_feat_len, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        #self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding,self.vocab_size,self.embedding_dim = create_emb_layer(weights_matrix, True)
        self.conv = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_size,self.embedding_dim+flex_feat_len)) 
        self.lstm = nn.LSTM(n_filters, hidden_dim, bidirectional=bidirectional, dropout=dropout)
        self.attention = SelfAttentionLSTM(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x,hc): 
        #x = [sent len, batch size]
        #x = x.permute(1, 0)   
        #x = [batch size, sent len]
        embedded = self.embedding(x)     
        #embedded = [batch size, sent len, emb dim]
        embedded = torch.cat((embedded,hc),2)
        embedded = embedded.unsqueeze(1)
        #embedded = [batch size, 1, sent len, emb dim]
        conved = F.relu(self.conv(embedded)).squeeze(3)
        #conved = [batch size, n_filters, sent len - filter_size]
        x = conved.permute(2, 0, 1)
        #x = [sent len- filter_size, batch size, n_filters]
        output, (hidden, cell) = self.lstm(x)
        #output = [sent len - filter_size, batch size, hid_dim*num directional]
        output = output[:, :, :self.hidden_dim] + output[:, :, self.hidden_dim:]
        # output = [sent len, batch size, hid dim]
        ouput = output.permute(1, 0, 2)
        # ouput = [batch size, sent len, hid dim]
        new_embed, weights = self.attention(ouput)
        # new_embed = [batch size, hid dim]
        # weights = [batch size, sent len]
        new_embed = self.dropout(new_embed)
        return self.fc(new_embed)

    def predict(self,embedded):
        embedded = embedded.unsqueeze(1)
        #embedded = [batch size, 1, sent len, emb dim]
        conved = F.relu(self.conv(embedded)).squeeze(3)
        #conved = [batch size, n_filters, sent len - filter_size]
        x = conved.permute(2, 0, 1)
        #x = [sent len- filter_size, batch size, n_filters]
        output, (hidden, cell) = self.lstm(x)
        #output = [sent len - filter_size, batch size, hid_dim*num directional]
        output = output[:, :, :self.hidden_dim] + output[:, :, self.hidden_dim:]
        # output = [sent len, batch size, hid dim]
        ouput = output.permute(1, 0, 2)
        # ouput = [batch size, sent len, hid dim]
        new_embed, weights = self.attention(ouput)
        # new_embed = [batch size, hid dim]
        # weights = [batch size, sent len]
        new_embed = self.dropout(new_embed)
        return self.fc(new_embed)
class AttentionCNN(nn.Module):
    def __init__(self, weights_matrix, n_filters, filter_sizes, output_dim, flex_feat_len, dropout=0.5):
        super().__init__()
        #self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding,self.vocab_size,self.embedding_dim = create_emb_layer(weights_matrix, True)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs,self.embedding_dim+flex_feat_len)) for fs in filter_sizes])
        self.attention = SelfAttention(n_filters)
        self.fc = nn.Linear(len(filter_sizes)*n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x,hc):
        #x = [sent len, batch size]
        #x = x.permute(1, 0)   
        #x = [batch size, sent len]
        embedded = self.embedding(x)
        embedded = torch.cat((embedded,hc),2)
        #embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        #embedded = [batch size, 1, sent len, emb dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        conved_att = [(self.attention(conv.permute(0, 2, 1))).permute(0, 2, 1) for conv in conved]
        #conved_att = [batch size, n_filters, sent len - filter sizes[i]]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved_att]
        #pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim=1))
        #cat = [batch size, n_filters * len(filter_sizes)]
        return self.fc(cat)
    def predict(self,embedded):
        #embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        #embedded = [batch size, 1, sent len, emb dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        conved_att = [(self.attention(conv.permute(0, 2, 1))).permute(0, 2, 1) for conv in conved]
        #conved_att = [batch size, n_filters, sent len - filter sizes[i]]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved_att]
        #pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim=1))
        #cat = [batch size, n_filters * len(filter_sizes)]
        return self.fc(cat)
class AugmentedConv(nn.Module):
    def __init__(self, weights_matrix, flex_feat_len, out_channels, kernel_size_in, dk=20, dv=10, Nh=2, shape=0, relative=False, stride=1):
        super(AugmentedConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size_in
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.shape = shape
        self.relative = relative
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."

        self.embedding,self.vocab_size,self.embedding_dim = create_emb_layer(weights_matrix, True)
        self.in_channels = 1 #flex_feat_len + self.embedding_dim
        self.conv_out = nn.Conv2d(self.in_channels, self.out_channels - self.dv,  kernel_size = kernel_size_in,stride=stride, padding=self.padding)  #(self.kernel_size, flex_feat_len + self.embedding_dim)

        self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size = 1)

        self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size = 1)

        if self.relative:
            self.key_rel_w = nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))
            self.key_rel_h = nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))

    def forward(self, x,hc):
        # Input x
        # (batch_size, channels, height, width)
        # batch, _, height, width = x.size()

        # conv_out
        # (batch_size, out_channels, height, width)
        barch_size=x.size()[0]
        embedded = self.embedding(x)
        embedded = torch.cat((embedded,hc),2)
        embedded = embedded.unsqueeze(1)
        # embedded (batch_size, 1, height, width)

        pdb.set_trace()
        conv_out = self.conv_out(embedded)
        batch, _, height, width = conv_out.size()

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, height * width, dvh or dkh)
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v
        # (batch_size, Nh, height, width, dv or dk)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(embedded, self.dk, self.dv, self.Nh)
        pdb.set_trace()

        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits
        weights = F.softmax(logits, dim=-1)

        # attn_out
        # (batch, Nh, height * width, dvh)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(attn_out, (batch, self.Nh, self.dv // self.Nh, height, width))
        # combine_heads_2d
        # (batch, out_channels, height, width)
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)
        cat_out = torch.cat((conv_out, attn_out), dim=1)
        pdb.set_trace()
        return cat_out

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        N, _, H, W = qkv.size()
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        pdb.set_trace()
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q *= dkh ** -0.5
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, H * W))
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, height, width)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.size()
        ret_shape = (batch, Nh * dv, H, W)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, H, W = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, H, W, Nh, "w")
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), self.key_rel_h, W, H, Nh, "h")

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))

        if case == "w":
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == "h":
            rel_logits = torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = torch.zeros((B, Nh, L, 1)).to(x)
        x = torch.cat((x, col_pad), dim=3)

        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x

class MulitChaCNN(nn.Module):
    def __init__(self, weights_matrix, n_filters, filter_sizes, output_dim,segment_lens, final_filter_w, flex_feat_len,
                 dropout=0.5):
        super().__init__()
        self.filter_sizes = filter_sizes
        oneD_filter_sizes = []
        for i in filter_sizes:
            oneD_filter_sizes.extend(i)
        oneD_seg_lens = []
        for j, i in enumerate(segment_lens):
            if len(final_filter_w[j]):
               oneD_seg_lens.extend(final_filter_w[j])
            else:
               oneD_seg_lens.extend([i]*len(filter_sizes[j]))
        comb = []
        for i in range(len(oneD_filter_sizes)):
            comb.append(tuple([oneD_filter_sizes[i],oneD_seg_lens[i]]))
        self.segment_lens = segment_lens
        #self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.embedding,self.vocab_size,self.embedding_dim = create_emb_layer(weights_matrix, True)
        #self.embedding_dim+flex_feat_len
        self.convs = []
        self.convs=nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fsd, fsw)) 
                                    for fsd,fsw in comb
                                    ])
        self.fc = nn.Linear(n_filters * len(comb), output_dim) #len(filter_sizes[0]) * len(segment_lens)
        self.dropout = nn.Dropout(dropout)

    def core(self, embedded):
        #embedded = [batch size, sent len, emb dim]
        split_embedded = torch.split(embedded,self.segment_lens,dim=2)
        pooled = []
        last_f_d = 0
        for i in range(len(self.segment_lens)):
            embedded_c = split_embedded[i].unsqueeze(1)
            #embedded = [batch size, 1, sent len, emb dim]
            conved = []
            for j in range(0,len(self.filter_sizes[i])):
                cur_index = last_f_d+j
                temp_r = F.relu(self.convs[cur_index](embedded_c))
                if temp_r.shape[3] > 1:
                    temp_r = temp_r.view(temp_r.shape[0],temp_r.shape[1],-1)
                    conved.append(temp_r)
                else:
                    conved.append(temp_r.squeeze(3))
            #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
            pooled.extend([F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved])
            last_f_d = last_f_d+len(self.filter_sizes[i])
            #pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim = 1))
            #cat = [batch size, n_filters * len(filter_sizes)]
        return self.fc(cat)

    def forward(self, text,hc):
                
        #text = [batch size, sent len]
        barch_size=text.size()[0]
        embedded = self.embedding(text)
        embedded = torch.cat((embedded,hc),2)
        output = self.core(embedded)        
        return output



class CNN(nn.Module):
    def __init__(self, weights_matrix, n_filters, filter_sizes, output_dim,flex_feat_len,
                 dropout=0.5):
        super().__init__()
        #self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.embedding,self.vocab_size,self.embedding_dim = create_emb_layer(weights_matrix, True)
        #
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fsd, self.embedding_dim+flex_feat_len)) 
                                    for fsd in filter_sizes
                                    ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def core(self, embedded):
        #embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)

        #embedded = [batch size, 1, sent len, emb dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        #cat = [batch size, n_filters * len(filter_sizes)]
        return self.fc(cat)

    def forward(self, text,hc):
                
        #text = [batch size, sent len]
        barch_size=text.size()[0]
        embedded = self.embedding(text)
        embedded = torch.cat((embedded,hc),2)
        output = self.core(embedded)        
        return output




class BiLSTM_CRF(nn.Module):

    def __init__(self, weights_matrix, tagset_size, hidden_dim,num_layers, flex_feat_len, dropout=0.5):
        super(BiLSTM_CRF, self).__init__()
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        self.num_layers = num_layers
        self.word_embeds,self.vocab_size,self.embedding_dim = create_emb_layer(weights_matrix, True)
        
        self.lstm = nn.LSTM(self.embedding_dim+flex_feat_len, hidden_dim // 2,
                            self.num_layers, bidirectional=True)

        self.dropout = nn.Dropout(dropout)
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        self.relu = nn.Tanh()
        #self.hidden = self.init_hidden()
        self.softmax = nn.LogSoftmax(dim=1)
        self.prediction = nn.Softmax(dim=1)

    def init_hidden(self,barch_size=100):
        return (torch.randn(2*self.num_layers, barch_size, self.hidden_dim // 2),
                torch.randn(2*self.num_layers, barch_size, self.hidden_dim // 2))

    def predict(self,embeds):
        barch_size=embeds.size()[0]
        lstm_out, (final_hidden_state, final_cell_state) = self.lstm(embeds)
        word_size=lstm_out.size()[1]
        last_cell_out = lstm_out.narrow(1,word_size-1,1)
        drop_out_feat = self.dropout(last_cell_out)
        lstm_feats = self.hidden2tag(drop_out_feat)
        score = self.prediction(lstm_feats.view(barch_size,self.tagset_size))
        return score

    def _get_lstm_features(self, sentence,hc):
        barch_size=sentence.size()[0]
        #self.hidden = self.init_hidden(barch_size), self.hidden

        embeds = self.word_embeds(sentence)

        embeds = torch.cat((embeds,hc),2)
        lstm_out, (final_hidden_state, final_cell_state) = self.lstm(embeds)
        return lstm_out,final_hidden_state, final_cell_state

    def forward(self, sentence, hc):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        barch_size=sentence.size()[0]
        lstm_out, final_hidden_state, final_cell_state = self._get_lstm_features(sentence,hc)
        word_size=lstm_out.size()[1]
        last_cell_out = lstm_out.narrow(1,word_size-1,1)
        drop_out_feat = self.dropout(last_cell_out)
        #word_size=drop_out_feat.size()[1]
        #lstm_feats = self.hidden2tag(drop_out_feat.narrow(1,word_size-1,1))
        lstm_feats = self.hidden2tag(drop_out_feat)
        score = self.softmax(lstm_feats.view(barch_size,self.tagset_size))
        return score



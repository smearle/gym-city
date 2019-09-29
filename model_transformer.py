
import heapq
from copy import deepcopy
from math import log
import random
import torch
import torch.nn as nn
import numpy as np

from data_helpers import vocab
from data_helpers import data_loader_ks as data_loader
import models.utils_models as utils


#########################################
#               CONSTANTS               #
#########################################
WORD_EMB_DIM    = 1000
NB_HIDDEN_ATT   = 512


#########################################
#               MAIN MODEL              #
#########################################
class Transformer(nn.Module):
    def __init__(self, dict_size, image_feature_dim, vocab, tf_ratio):
        super(Transformer, self).__init__()
        self.dict_size = dict_size
        self.image_feature_dim = image_feature_dim
        self.vocab = vocab
        self.tf_ratio = tf_ratio

        self.embed_word = Embed_Word(dict_size, WORD_EMB_DIM)
        self.img_attention = Visual_Attention(image_feature_dim,
                                          WORD_EMB_DIM,
                                          NB_HIDDEN_ATT)
        self.word_attention = Visual_Attention(WORD_EMB_DIM,
                                          image_feature_dim,
                                          NB_HIDDEN_ATT)
        self.predict_word = Predict_Word(image_feature_dim+WORD_EMB_DIM,
                                         1024,
                                         2,
                                         dict_size)

    def forward(self, image_feats, nb_timesteps, true_words, beam=None):
        if beam is not None:
            return self.beam_search(image_feats, nb_timesteps, beam)

        nb_batch, nb_image_feats, _ = image_feats.size()
        use_cuda = image_feats.is_cuda

        v_mean = image_feats.mean(dim=1)

        state, current_word = self.init_inference(nb_batch, use_cuda)
        y_out = utils.make_zeros((nb_batch, nb_timesteps-1, self.dict_size),
                                 cuda = use_cuda)

        for t in range(nb_timesteps-1):
            y, state = self.forward_one_step(state,
                                             current_word,
                                             v_mean,
                                             image_feats)
            y_out[:,t,:] = y

            current_word = self.update_current_word(y, true_words, t, use_cuda)

        return y_out

    def forward_one_step(self, state, current_word, v_mean, image_feats):
        nb_batch, nb_image_feats, image_feat_dim = image_feats.size()
        word_emb = self.embed_word(current_word).unsqueeze(0)
        if state is None:
            state = word_emb
        else:
            state = torch.cat((state, word_emb), dim=1)
        _, nb_timestep, word_dim = state.size()
        pos = get_sinusoid_encoding_table(nb_timestep, word_dim)
        if image_feats.is_cuda:
            pos = pos.cuda()
        state_with_pos = state + pos.unsqueeze(0).repeat(nb_batch, 1, 1)
        attended_words = self.word_attention(state, v_mean)
        attended_images = self.img_attention(image_feats, attended_words)
        all_features = torch.cat((attended_words, attended_images), dim=1)

        y = self.predict_word(all_features)
        return y, state

    def update_current_word(self, y, true_words, t, cuda):
        use_tf = True if random.random() < self.tf_ratio else False
        if use_tf:
            next_word = true_words[:,t+1]
        else:
            next_word = torch.argmax(y, dim=1)

        current_word = data_loader.indexto1hot(len(self.vocab), next_word)
        current_word = torch.from_numpy(current_word).float()
        
        if cuda:
            current_word = current_word.cuda()
        return current_word

    def init_inference(self, nb_batch, cuda):
        start_word = data_loader.indexto1hot(len(self.vocab), self.vocab('<start>'))
        start_word = torch.from_numpy(start_word).float().unsqueeze(0)
        start_word = start_word.repeat(nb_batch, 1)

        if cuda:
            start_word = start_word.cuda()


        return None, start_word

    #########################################
    #               BEAM SEARCH             #
    #########################################
    def beam_search(self, image_features, max_nb_words, beam_width):
        # Initialize model
        use_cuda = image_features.is_cuda
        nb_batch, nb_image_feats, _ = image_features.size()

        v_mean = image_features.mean(dim=1)
        state, current_word = self.init_inference(nb_batch, use_cuda)

        # Initialize beam search
        end_word = data_loader.indexto1hot(len(self.vocab), self.vocab('<end>'))
        end_word = torch.from_numpy(end_word).float().unsqueeze(0)
        end_word = end_word.cuda() if image_features.is_cuda else end_word
        beam = Beam(beam_width)
        s = Sentence(max_nb_words, beam_width, end_word, self.vocab)
        s.update_state(1.0, state, current_word)
        beam.push(s)

        # Perform beam search
        final_beam = Beam(beam_width)
        while len(beam) > 0:
            s = beam.pop()
            new_sentences = self.update_states(s, image_features, v_mean)
            for s in new_sentences:
                if s.ended:
                    final_beam.push(s)
                else:
                    beam.push(s)
            # Get rid of low scoring sentences
            beam.trim() 
            final_beam.trim()

        # Extract final sentence
        s = final_beam.pop() # Best sentence on top of heap
        sentence = s.extract_sentence()
        
        return sentence

    def update_states(self, s, image_feats, v_mean):
        state, current_word = s.get_states()
        y, state = self.forward_one_step(state, current_word, v_mean, image_feats)
        y = self.remove_consecutive_words(y, current_word)
        new_sentences = s.update_words(s, state, y)
        return new_sentences

    def remove_consecutive_words(self, y, prev_word):
        # give previous word low score so as not to repeat words
        y = y - 10**10 * prev_word
        return y

#################################################
#                   BEAM SEARCH                 #
#################################################
class Sentence(object):
    def __init__(self, max_nb_words, beam_width, end_word, vocab):
        self.max_nb_words = max_nb_words
        self.beam_width = beam_width
        self.end_word = end_word
        self.vocab = vocab

        self.words = []
        self.probability = 0
        self.ended = False

        self.act = nn.Softmax(dim=1)

    def update_words(self, s, state, y):
        y = self.act(y)
        new_s = []
        for i in range(self.beam_width):
            val, idx = y.max(dim=1)
            y[0, idx] -= val
            current_word = y.clone()
            current_word[0,:] = 0
            current_word[0,idx] = 1
            s2 = s.copy()
            s2.update_state(val,state,current_word)
            new_s.append(s2)
            if s2.ended:
                break
        return new_s

    def update_state(self, p, state, current_word):
        self.state = [s.clone() for s in state]
        self.words.append(current_word)
        self._update_probability(p)
        self._update_finished()

    def get_states(self):
        return self.state, self.words[-1]

    def extract_sentence(self):
        sentence = []
        for w in self.words:
            idx = w.max(1)[1].item()
            sentence.append(self.vocab.get_word(idx))
        return [self.probability, sentence]

    def _update_probability(self, p):
        self.probability += log(p,2)

    def _update_finished(self):
        n = len(self.words)
        f = self.words[-1]
        if (n > self.max_nb_words) or (f == self.end_word).all():
            self.ended = True

    def copy(self):
        new = Sentence(self.max_nb_words,
                       self.beam_width,
                       self.end_word,
                       self.vocab)
        new.words = [w.clone() for w in self.words]
        new.probability = self.probability
        return new

    def __lt__(self, other):
        return self.probability < other.probability

    def __repr__(self):
        s = ''
        for w in self.words:
            idx = w.max(1)[1].item()
            s += "{}, ".format(self.vocab.get_word(idx))
        return s


class Beam(object):
    def __init__(self, beam_width):
        self.beam_width = beam_width
        self.heap = []

    def push(self, s):
        s.probability *= -1
        heapq.heappush(self.heap, s)

    def pop(self):
        s = heapq.heappop(self.heap)
        s.probability *= -1
        return s

    def trim(self):
        h2 = []
        for i in range(self.beam_width):
            if len(self.heap) == 0:
                break
            heapq.heappush(h2, heapq.heappop(self.heap))
        self.heap=h2

    def __len__(self):
        return len(self.heap)

#####################################################
#               LANGUAGE SUB-MODULES                #
#####################################################
class Embed_Word(nn.Module):
    def __init__(self, dict_size, word_emb_dim):
        super(Embed_Word,self).__init__()
        self.embed_word = nn.Linear(dict_size, word_emb_dim, bias=False)

    def forward(self, word):
        emb_word = self.embed_word(word)
        return emb_word


class Visual_Attention(nn.Module):
    def __init__(self, dim_image_feats, dim_word_emb, nb_hidden):
        super(Visual_Attention,self).__init__()
        self.fc_image_feats = nn.Linear(dim_image_feats, nb_hidden, bias=False)
        self.fc_att_lstm = nn.Linear(dim_word_emb, nb_hidden, bias=False)
        self.act_tan = nn.Tanh()
        self.fc_att = nn.Linear(nb_hidden, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image_feats, word_emb):
        nb_batch, nb_feats, feat_dim = image_feats.size()
        att_lstm_emb = self.fc_att_lstm(word_emb).unsqueeze(1)
        image_feats_emb = self.fc_image_feats(image_feats)
        all_feats_emb = image_feats_emb + att_lstm_emb.repeat(1,nb_feats,1)

        activate_feats = self.act_tan(all_feats_emb)
        unnorm_attention = self.fc_att(activate_feats)
        normed_attention = self.softmax(unnorm_attention)

        #print(normed_attention.shape)
        #print(nb_feats)
        #print(image_feats.shape)
        #weighted_feats = normed_attention.repeat(1,1,nb_feats) * image_feats
        weighted_feats = normed_attention * image_feats
        #print(weighted_feats.shape)
        attended_image_feats = weighted_feats.sum(dim=1)
        #print(attended_image_feats.shape)
        return attended_image_feats

class Predict_Word(nn.Module):
    def __init__(self, input_dim, nb_hidden, nb_layers, dict_size):
        super(Predict_Word, self).__init__()
        layers = [nn.Linear(input_dim, nb_hidden)]
        for _ in range(nb_layers-1):
            layers.append(nn.Linear(nb_hidden, nb_hidden))
        layers.append(nn.Linear(nb_hidden, dict_size))
        self.layers = nn.ModuleList(layers)
        self.act = nn.ReLU()
        
    def forward(self, emb):
        for l in self.layers[:-1]:
            emb = self.act(l(emb))
        y = self.layers[-1](emb)
        return y

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        all_hid_idx = np.arange(d_hid)
        pos = position / np.power(10000, 2 * (all_hid_idx // 2) / d_hid)
        return pos.tolist()

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def test_for_nan(x, name="No name given"):
    if torch.isnan(x).sum() > 0:
        print("{} has NAN".format(name))
        exit()



import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import math
import numpy as np
from variational_dropout import VariationalDropout
from layers import GraphConvolution

def masked_softmax(vector, mask):
    result = F.softmax(vector * mask, -1)
    result = result * mask
    result = result / (result.sum(1, keepdim=True) + 1e-13)
    return result

class ProdLDA(nn.Module):

    def __init__(self, net_arch):
        super(ProdLDA, self).__init__()
        ac = net_arch
        self.net_arch = net_arch
        dropout_ratio = 0.6
        self.sparse = 1.
        # encoder

        #self.en1_fc = nn.Linear(484, ac.en1_units) 
        
        self.gcn1       = GraphConvolution(1995, 100)
        self.gcn2       = GraphConvolution(100, 100)
        self.gcn3       = GraphConvolution(100, 1)
        self.en1_fc     = nn.Linear(ac.num_input, ac.en1_units)             # 1995 -> 100
        self.en2_fc     = nn.Linear(ac.en1_units, ac.en2_units)             # 100  -> 100
        self.en2_drop   = VariationalDropout(dropout_ratio)
        self.mean_fc    = nn.Linear(ac.en2_units, ac.num_topic)             # 100  -> 50
        self.mean_bn    = nn.BatchNorm1d(ac.num_topic)                      # bn for mean
        self.logvar_fc  = nn.Linear(ac.en2_units, ac.num_topic)             # 100  -> 50
        self.logvar_bn  = nn.BatchNorm1d(ac.num_topic)                      # bn for logvar
        # z
        self.p_drop     = VariationalDropout(dropout_ratio)
        self.w_drop     = nn.Dropout(dropout_ratio                      )
        # decoder
        self.decoder    = nn.Linear(ac.num_topic, ac.num_input, bias=False)            # 50   -> 1995
        self.decoder_bn = nn.BatchNorm1d(ac.num_input)                      # bn for decoder

        self.h_dim = ac.num_topic
        self.a = 1*np.ones((1 , self.h_dim)).astype(np.float32)
        prior_mean = torch.from_numpy((np.log(self.a).T - np.mean(np.log(self.a), 1)).T)
        prior_var = torch.from_numpy((((1.0 / self.a) * (1 - (2.0 / self.h_dim))).T +
                                 (1.0 / (self.h_dim * self.h_dim)) * np.sum(1.0 / self.a, 1)).T)
        prior_logvar = prior_var.log()

        self.register_buffer('prior_mean',    prior_mean)
        self.register_buffer('prior_var',     prior_var)
        self.register_buffer('prior_logvar',  prior_logvar)

        nn.init.xavier_normal_(self.decoder.weight,1)
        nn.init.xavier_normal_(self.en1_fc.weight, 1)
        nn.init.xavier_normal_(self.en2_fc.weight, 1)
        nn.init.xavier_normal_(self.mean_fc.weight, 1)
        nn.init.xavier_normal_(self.logvar_fc.weight, 1)

        nn.init.constant_(self.en1_fc.bias, 0)
        nn.init.constant_(self.en2_fc.bias, 0)
        nn.init.constant_(self.mean_fc.bias, 0)
        nn.init.constant_(self.logvar_fc.bias, 0)

        self.logvar_bn.weight.requires_grad = False
        self.mean_bn.weight.requires_grad = False
        self.decoder_bn.weight.requires_grad = False

        self.logvar_bn.weight.fill_(1)
        self.mean_bn.weight.fill_(1)
        self.decoder_bn.weight.fill_(1)


        self.params = list(self.en1_fc.parameters()) + list(self.en2_fc.parameters()) +             list(self.mean_fc.parameters()) \
                       + list(self.logvar_fc.parameters()) + list(self.decoder.parameters()) + \
                        list([self.mean_bn.bias]) + list([self.logvar_bn.bias]) + \
                           list([self.decoder_bn.bias]) + list(self.gcn1.parameters()) + list(self.gcn2.parameters()) + list(self.gcn3.parameters())

    
    def batch_diag(self, mat, res):
        return res.as_strided(mat.size(), [res.stride(0), res.size(2) + 1]).copy_(mat)
    
    def forward(self, inputs, eye, compute_loss=False, avg_loss=True, l1=False, target=None):

        gcns = []
        for input, adj in inputs:
            gcn = self.w_drop(F.relu(self.gcn1(input, adj)))
            gcn = self.w_drop(F.relu(self.gcn2(gcn, adj))) + gcn
            gcn = F.relu(self.gcn3(gcn, adj).squeeze())
            gcns.append(gcn.unsqueeze(0))
        gcn = torch.cat(gcns, 0)
        en1 = F.relu(self.en1_fc(gcn))                # en1_fc   output
        en2 = F.relu(self.en2_fc(en1))                              # encoder2 output
        en2 = self.en2_drop(en2)
        posterior_mean   = self.mean_bn  (self.mean_fc  (en2))          # posterior mean
        posterior_logvar = self.logvar_bn(self.logvar_fc(en2))          # posterior log variance
        posterior_var    = posterior_logvar.exp()
        # take sample
        eps = Variable(posterior_mean.data.new().resize_as_(posterior_mean.data).normal_(0,1)) # noise
        z = posterior_mean + posterior_var.sqrt() * eps                 # reparameterization
        self.p = F.softmax(z, -1)                                               # mixture probability
        self.z = z
        p = self.p_drop(self.p)

        recon = F.softmax(self.decoder_bn(self.decoder(p)),-1)

        if compute_loss:
            return recon, self.loss(input, recon, posterior_mean, posterior_logvar, posterior_var, avg_loss, l1, target)
        else:
            return recon

    def loss(self, input, recon, posterior_mean, posterior_logvar, posterior_var, avg=True,l1=False,target=None):
        # NLword_code
        target = input if target is None else target

        recon = (recon + 1e-10).log().unsqueeze(-1)
        NL = []
        for i, t in enumerate(target):
            NL.append(torch.mm(t.sum(0).unsqueeze(0), recon[i]) + torch.mm(t.sum(1).unsqueeze(0), recon[i]))
        NL = torch.cat(NL, 0).squeeze()
        # KLD, see Section 3.3 of Akash Srivastava and Charles Sutton, 2017,
        # https://arxiv.org/pdf/1703.01488.pdf
        prior_mean   = Variable(self.prior_mean).expand_as(posterior_mean)
        prior_var    = Variable(self.prior_var).expand_as(posterior_mean)
        prior_logvar = Variable(self.prior_logvar).expand_as(posterior_mean)
        var_division    = posterior_var  / prior_var
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        # put KLD together
        KLD = 0.5 * ( (var_division + diff_term + logvar_division).sum(1) - self.net_arch.num_topic )
        # loss
        loss = (-NL*0.0001 + KLD )

        # in traiming mode, return averaged loss. In testing mode, return individual loss
        if avg:
            return loss.mean()
        else:
            return loss



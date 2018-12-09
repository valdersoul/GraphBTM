import numpy as np
from torch.autograd import Variable
import torch.cuda
import _pickle as pickle
import argparse
import os
from tqdm import tqdm

from pytorch_model import ProdLDA
from dataloader import bitermsDataset
from sparseMM import *

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--en1-units',        type=int,   default=100)
parser.add_argument('-s', '--en2-units',        type=int,   default=100)
parser.add_argument('-t', '--num-topic',        type=int,   default=50)
parser.add_argument('-b', '--batch-size',       type=int,   default=100)
parser.add_argument('-o', '--optimizer',        type=str,   default='Adam')
parser.add_argument('-r', '--learning-rate',    type=float, default=0.002)
parser.add_argument('-m', '--momentum',         type=float, default=0.99)
parser.add_argument('-e', '--num-epoch',        type=int,   default=200)
parser.add_argument('-q', '--init-mult',        type=float, default=1.0)    # multiplier in initialization of decoder weight
parser.add_argument('-v', '--variance',         type=float, default=0.995)  # default variance in prior normal
parser.add_argument('--start',                  action='store_true')        # start training at invocation
parser.add_argument('--nogpu',                  action='store_true')        # do not use GPU acceleration
parser.add_argument('-d', '--data-root', type=str, default='data/20news_clean/')
parser.add_argument('-a', '--betavariance',     type=float, default=0.04)

args = parser.parse_args()
batch_size=args.batch_size

biterms = []
window_length = 30
mini_doc = 3

# default to use GPU, but have to check if GPU exists
if not args.nogpu:
    if torch.cuda.device_count() == 0:
        args.nogpu = True


def to_onehot(data, min_length):
    return np.bincount(data, minlength=min_length)

def padding(data):
    max_len = max([len(doc) for doc in data])
    new_data = np.array([np.append(doc,[0] * (max_len - len(doc))) for doc in data]).astype(np.int64)
    return new_data

def make_corpus(dataset, vocab):
    file_name = args.data_root + "corpus.txt"
    import os.path
    if os.path.isfile(file_name):
        return
    with open(file_name, "w") as f:
        for i in range(dataset.size):
            line = [vocab[x] for x in dataset[i]]
            f.write(" ".join(line) + "\n")

def put_dic(biterm, biterms):
    if biterm not in biterms:
        biterms[biterm] = 1
    else:
        biterms[biterm] += 1

def make_biterm(data, biterms):
    import itertools
    print (len(biterms))
    for doc in data:
        if np.sum(doc) == 0:
            continue
        doc_len = len(doc)
        temp = {}
        if doc_len > window_length:
            for i in range(doc_len - window_length):
                for biterm in itertools.combinations(doc[i:i+window_length], 2):
                    put_dic(frozenset(biterm), temp)
        else:
            for biterm in itertools.combinations(doc, 2):
                put_dic(frozenset(biterm), temp)
        biterms.append(temp)
    print (len(biterms))
    with open(args.data_root+'biterms.pickle', 'wb') as f:
        pickle.dump(biterms, f)

def make_graph(biterms):
    merge = {}
    for d in biterms:
        merge = {**merge, **d}
    temp = np.zeros((args.num_input, args.num_input))
    for k, v in merge.items():
        if len(k) < 2:
            a = b = list(k)[0]
        else:
            a, b = k
        temp[a, b] = v
    return temp

def make_data():
    global data_tr, data_te, tensor_tr, tensor_te, vocab, vocab_size, n_samples_tr
    global data_tr_index, data_tr_count, data_te_index, data_te_count, biterms
    dataset_tr = args.data_root + 'train.txt.npy'
    data_tr = np.load(dataset_tr, encoding='bytes')

    vocab = args.data_root + 'vocab.pkl'
    vocab = pickle.load(open(vocab,'rb'))
    vocab_size=len(vocab)
    args.num_input = vocab_size

    biterm_file_name = args.data_root+'biterms.pickle'
    import os.path
    if os.path.isfile(biterm_file_name):
        print('loading biterms')
        with open(biterm_file_name, 'rb') as f:
            biterms = pickle.load(f, encoding='bytes')
    else:
        biterms = np.array(biterms)
        print('generating biterm graphs')
        make_biterm(data_tr, biterms)
    biterms = np.array(biterms)

    make_corpus(data_tr, list(zip(*sorted(vocab.items(), key=lambda x:x[1])))[0])

    print('Data Loaded')

def make_model():
    global model
    net_arch = args # en1_units, en2_units, num_topic, num_input
    net_arch.num_input = vocab_size
    model = ProdLDA(net_arch)
    if not args.nogpu:
        model = model.cuda()

def make_optimizer():
    global optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adadelta(model.params, lr=1, rho=0.99)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum)
    else:
        assert False, 'Unknown optimizer {}'.format(args.optimizer)

def train(dataloader):
    for epoch in range(args.num_epoch):
        loss_epoch = 0.0
        model.train()                   # switch to training mode
        #total_batch = int(n_samples_tr / batch_size)
        sparsity = False
        count = 0
        b_count = 0
        GCNsInputList = []
        target_input = []
        losses = []
        data_size = len(dataloader)
        for biterm in dataloader:

            mm = SparseMM.apply
            biterm = torch.FloatTensor(biterm).float().cuda()
            target_input.append(Variable(biterm))

            sparse_biterms = to_sparse(biterm.float().cuda())
            ones = torch.cuda.FloatTensor(biterm.shape[0]).fill_(1).unsqueeze(-1)

            indices = to_sparse(mm(sparse_biterms, ones))._indices()
            values = torch.cuda.FloatTensor(indices.size()[1]).fill_(1)

            adj_mask = torch.cuda.sparse.FloatTensor(indices, values, (sparse_biterms.size()[0], 1))

            eye = sparse_ones(biterm.size()[0]).cuda()

            adj = (sparse_biterms + eye).coalesce()

            degree_matrix = mm(adj, ones)
            degree_matrix = torch.pow(degree_matrix, -0.5)
            degree_matrix = degree_matrix * adj_mask.to_dense()
            degrees = sparse_diag(degree_matrix.squeeze(-1)).coalesce()

            adj = mm(adj, degrees.to_dense())
            adj = mm(degrees, adj)
            indices = (sparse_biterms + eye).coalesce()._indices()
            values = adj[tuple(indices[i] for i in range(indices.shape[0]))]
            adj = torch.cuda.sparse.FloatTensor(indices, values, sparse_biterms.size())

            GCNsInputList.append( (Variable(sparse_biterms),
                                    Variable(adj)) )
            b_count += 1
            if b_count % batch_size != 0:
                continue
            if b_count > data_size:
                break

            _, loss = model(GCNsInputList, None, compute_loss=True, l1=False, target=target_input)

            # optimize
            optimizer.zero_grad()       # clear previous gradients
            loss.backward()             # backprop

            #torch.nn.utils.clip_grad_norm(model.params, 5)
            optimizer.step()            # update parameters
            # report
            loss_epoch += loss.data[0]    # add loss to loss_epoch
            GCNsInputList = []
            target_input = []
            if count % 10 == 0:
                print('Epoch {}, count {}, loss={}'.format(epoch, count, loss_epoch / (count+1)))
            count += 1

        emb = torch.nn.functional.softmax(model.decoder.weight, 0).data.cpu().numpy().T
        print_top_words(emb, list(zip(*sorted(vocab.items(), key=lambda x: x[1])))[0])

def print_top_words(beta, feature_names, n_top_words=10):
    print( '---------------Printing the Topics------------------')
    with open('topic_interpretability/data/topics_20news.txt', 'w') as f:
        for i in range(len(beta)):
            print(" ".join([feature_names[j+1]
                for j in beta[i][1:].argsort()[:-n_top_words - 1:-1]]))
            f.write(" ".join([feature_names[j+1]
                for j in beta[i][1:].argsort()[:-n_top_words - 1:-1]]) + '\n')
    print( '---------------End of Topics------------------')

def sample(data,biterms):
    rng = np.random.RandomState(10)
    data_new = []
    biterms_new = []
    for i in tqdm(range(len(data))):
        ixs = rng.randint(data.shape[0], size=mini_doc)
        sample_biterms = make_graph(biterms[ixs])
        sample_datas = data[ixs].sum(0)
        data_new.append(sample_datas)
        biterms_new.append(sample_biterms)
    return np.array(data_new), biterms_new

def save_checkpoint(model, path):
    torch.save(model, os.path.join(path, 'model.pt'))
        
if __name__=='__main__' and args.start:
    make_data()
    make_model()
    make_optimizer()
    data_new = []
    print (len(biterms))
    dataset = bitermsDataset(np.array(biterms), args.num_input, mini_doc, data=None)
    print ('start training')
    train(dataset)
    emb = torch.nn.functional.softmax(model.decoder.weight, 0).data.cpu().numpy().T
    print_top_words(emb, list(zip(*sorted(vocab.items(), key=lambda x:x[1])))[0])
    save_checkpoint(model, 'saved_models')




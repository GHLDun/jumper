#!/usr/bin/python3
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch
from torch.nn import functional, init
from torch.nn import Parameter
try:
    import utils
except ImportError:
    import nntools.utils as utils


def repackage_var(vs, requires_grad = False):

    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(vs) == Variable:
        return Variable(vs.data, requires_grad = requires_grad)
    elif type(vs) == Parameter:
        return Parameter(vs.data,requires_grad = requires_grad)
    else:
        return tuple(repackage_var(v) for v in vs)

def onehot(data, n_dimension):
    assert data.size(1) ==1 and (torch.max(data)< n_dimension).data.all() # bs,1
    y_onehot = Variable(torch.FloatTensor(data.size(0),n_dimension).zero_())
    ones = Variable(torch.FloatTensor(data.size()).fill_(1))
    if data.is_cuda:
        y_onehot = y_onehot.cuda()
        ones = ones.cuda()

    y_onehot.scatter_(1,data,ones)
    return y_onehot

def cal_loss(distribution, target):
    # assert distribution.size[0] == target.size[0]
    target_label = target.view(-1,1)
    y_onehot = Variable(torch.FloatTensor(distribution.size()).zero_()).cuda()
    ones = Variable(torch.FloatTensor(target_label.size()).fill_(1)).cuda()
    y_onehot.scatter_(1,target_label,ones)
    log_dis = torch.log(distribution) 
    loss = torch.sum(-y_onehot*log_dis, dim = 1)
    return loss.view(-1,1) #(b_s,1)
 
def cal_loss_cpu(distribution, target):
    # assert distribution.size[0] == target.size[0]
    target_label = target.view(-1,1)
    y_onehot = Variable(torch.FloatTensor(distribution.size()).zero_())
    ones = Variable(torch.FloatTensor(target_label.size()).fill_(1))
    # print target_label
    # print ones
    # print y_onehot
    y_onehot.scatter_(1,target_label,ones)
    log_dis = torch.log(distribution) 
    loss = torch.sum(-y_onehot*log_dis, dim = -1)
    return loss #(b_s,1)

def cal_sf_loss(distribution, target):
    # assert distribution.size[0] == target.size[0]
    target_label = target.view(-1,1)
    y_onehot = Variable(torch.FloatTensor(distribution.size()).zero_()).cuda()
    ones = Variable(torch.FloatTensor(target_label.size()).fill_(1)).cuda()
    y_onehot.scatter_(1,target_label,ones)
    dec_out = nn.LogSoftmax()(distribution) 
    loss = torch.sum(-y_onehot*dec_out, dim = -1)
    return loss
 
def map_tensor(map, tensor):
    shape = tensor.size()
    data = tensor.view(-1).tolist()
    # print data
    data_map = [map[i] for i in data]
    return torch.FloatTensor(data_map).view(shape)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.weight_fm = nn.Linear(hidden_size, hidden_size)
        self.weight_im = nn.Linear(hidden_size, hidden_size)
        self.weight_cm = nn.Linear(hidden_size, hidden_size)
        self.weight_om = nn.Linear(hidden_size, hidden_size)
        self.weight_fx = nn.Linear(input_size, hidden_size)
        self.weight_ix = nn.Linear(input_size, hidden_size)
        self.weight_cx = nn.Linear(input_size, hidden_size)
        self.weight_ox = nn.Linear(input_size, hidden_size)

        self.init_weights()

    def forward(self, inputs, input_m = None, hidden = None):
        '''
        inputs: (b_s, l, emb_size)
        '''
        inputs_p = inputs.permute(1,0,2) 
        b_s = inputs_p.size()[1]
        if hidden is  not None:
            h_t,c_t = hidden
        else:
            h_t,c_t = self.init_hidden(b_s)

        if input_m is not None:
            inputs_mask_p = input_m.permute(1,0).contiguous() 
        else:
            inputs_mask_p = Variable(torch.ones(inputs_p.size()[:-1]).cuda())
        steps = len(inputs_p)

        outputs = Variable(torch.zeros(b_s,steps, self.hidden_size).cuda())

        for i in range(steps):
            input = inputs_p[i]
            input_mask = inputs_mask_p[i]
            h_t, c_t = self.step(input, input_mask, h_t, c_t)
            outputs[:,i,:] = h_t

  #       result = outputs.permute(1,0,2).contiguous() 
        return  outputs,(h_t,c_t)

    def step(self, inp, input_mask, h_0, c_0):
        # forget gate
        f_g = nn.Sigmoid()(self.weight_fx(inp) + self.weight_fm(h_0))
        # f_g = F.sigmoid(self.weight_fx(inp) + self.weight_fm(h_0))
        # input gate
        i_g = nn.Sigmoid()(self.weight_ix(inp) + self.weight_im(h_0))
        # output gate
        o_g = nn.Sigmoid()(self.weight_ox(inp) + self.weight_om(h_0))
        # intermediate cell state
        c_tilda = nn.Tanh()(self.weight_cx(inp) + self.weight_cm(h_0))
        # current cell state
        cx = f_g * c_0 + i_g * c_tilda
        # hidden state
        hx = o_g * nn.Tanh()(cx)

        mask =  input_mask.view(-1,1).expand_as(hx)
        ho = hx *mask + h_0 * (1-mask) # (1,b_s,hids)
        return ho, cx

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(batch_size, self.hidden_size).zero_()),
                Variable(weight.new(batch_size, self.hidden_size).zero_()))


    def init_weights(self):
        initrange = 0.1
        self.weight_fm.weight.data.uniform_(-initrange, initrange)
        self.weight_im.weight.data.uniform_(-initrange, initrange)
        self.weight_cm.weight.data.uniform_(-initrange, initrange)
        self.weight_om.weight.data.uniform_(-initrange, initrange)
        self.weight_fx.weight.data.uniform_(-initrange, initrange)
        self.weight_ix.weight.data.uniform_(-initrange, initrange)
        self.weight_cx.weight.data.uniform_(-initrange, initrange)
        self.weight_ox.weight.data.uniform_(-initrange, initrange)

        self.weight_fm.bias.data.fill_(0)
        self.weight_im.bias.data.fill_(0)
        self.weight_cm.bias.data.fill_(0)
        self.weight_om.bias.data.fill_(0)
        self.weight_fx.bias.data.fill_(0)
        self.weight_ix.bias.data.fill_(0)
        self.weight_cx.bias.data.fill_(0)
        self.weight_ox.bias.data.fill_(0)

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.weight_fm = nn.Linear(hidden_size, hidden_size)
        self.weight_im = nn.Linear(hidden_size, hidden_size)
        self.weight_cm = nn.Linear(hidden_size, hidden_size)
        self.weight_fx = nn.Linear(input_size, hidden_size)
        self.weight_ix = nn.Linear(input_size, hidden_size)
        self.weight_cx = nn.Linear(input_size, hidden_size)
        # self.init_weights()

    def forward(self, inputs, input_m = None, hidden = None):
        '''
        inputs: (b_s, l, emb_size)
        '''
        inputs_p = inputs.permute(1,0,2) 
        b_s = inputs_p.size()[1]
        if hidden is  not None:
            h_t = hidden
        else:
            h_t = self.init_hidden(b_s)
        steps = len(inputs_p)
        hts = []
        for i in range(steps):
            input = inputs_p[i]
            h_t= self.step(input, h_t)
            hts.append(h_t.unsqueeze(0))
        outputs = torch.cat(hts,0)
        return  outputs.permute(1,0,2),h_t

    def step(self, inp, h_0):
        # forget gate
        z_g = nn.Sigmoid()(self.weight_fx(inp) + self.weight_fm(h_0))
        r_g = nn.Sigmoid()(self.weight_ix(inp) + self.weight_im(h_0))
        h_tilda = nn.Tanh()(self.weight_cx(inp) + r_g * self.weight_cm(h_0))
        # current cell state
        h_t = (1-z_g) * h_0 + z_g * h_tilda

        # mask =  input_mask.view(-1,1).expand_as(h_t)
        # ho = h_t *mask + h_0 * (1-mask) # (1,b_s,hids)
        return h_t

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(batch_size, self.hidden_size).zero_()))

    def init_weights(self):
        initrange = 0.1
        self.weight_fm.weight.data.uniform_(-initrange, initrange)
        self.weight_im.weight.data.uniform_(-initrange, initrange)
        self.weight_cm.weight.data.uniform_(-initrange, initrange)
        self.weight_fx.weight.data.uniform_(-initrange, initrange)
        self.weight_ix.weight.data.uniform_(-initrange, initrange)
        self.weight_cx.weight.data.uniform_(-initrange, initrange)

        self.weight_fm.bias.data.fill_(0)
        self.weight_im.bias.data.fill_(0)
        self.weight_cm.bias.data.fill_(0)
        self.weight_fx.bias.data.fill_(0)
        self.weight_ix.bias.data.fill_(0)
        self.weight_cx.bias.data.fill_(0)

class GRU_F(nn.Module):
    '''
    add feedback
    '''
    def __init__(self, input_size, hidden_size, cate_num):
        super(GRU_F, self).__init__()
        self.hidden_size = hidden_size
        self.cate_num = cate_num
        self.weight_fm = nn.Linear(hidden_size, hidden_size)
        self.weight_im = nn.Linear(hidden_size, hidden_size)
        self.weight_cm = nn.Linear(hidden_size, hidden_size)
        self.weight_fx = nn.Linear(input_size+self.cate_num, hidden_size)
        self.weight_ix = nn.Linear(input_size+self.cate_num, hidden_size)
        self.weight_cx = nn.Linear(input_size+self.cate_num, hidden_size)
        self.cf = nn.Linear(hidden_size, cate_num)

    def forward(self, inputs,targets, input_m = None):
        '''
        inputs: (b_s, l, emb_size)
        '''
        inputs_p = inputs.permute(1,0,2) 
        b_s = inputs_p.size()[1]
        h_t = self.init_hidden(b_s)

        if input_m is not None:
            inputs_mask_p = input_m.permute(1,0).contiguous() 
        else:
            inputs_mask_p = Variable(torch.ones(inputs_p.size()[:-1]).cuda())

        steps = len(inputs_p)
        outputs = Variable(torch.zeros(b_s,steps, self.cate_num).cuda())
        ones = Variable(torch.ones(b_s,1).cuda())
        cate_tm1 = Variable(torch.zeros(b_s,self.cate_num).cuda())
        for i in range(steps):
            input = inputs_p[i]
            input_mask = inputs_mask_p[i]
            h_t= self.step(input, input_mask, h_t, cate_tm1)
            output = nn.Softmax()(self.cf(h_t))
            outputs[:,i,:] = output
            if  self.training:
                cate_tm1 = Variable(torch.zeros(b_s,self.cate_num).cuda())
                cate_tm1.scatter_(1,targets[:,i].contiguous().view(-1,1),ones)

            else:
                value,pred_t1 = torch.max(output,1)
                cate_tm1 = Variable(torch.zeros(b_s,self.cate_num).cuda())
                cate_tm1.scatter_(1,pred_t1.view(-1,1),ones)

        return  outputs

    def step(self, input_x, input_mask, h_0, cate_tm1):

        inp = torch.cat([input_x,cate_tm1],1) # (b_s, input+cate)
        # forget gate
        z_g = nn.Sigmoid()(self.weight_fx(inp) + self.weight_fm(h_0))
        # input gate
        r_g = nn.Sigmoid()(self.weight_ix(inp) + self.weight_im(h_0))

        h_tilda = nn.Tanh()(self.weight_cx(inp) + r_g * self.weight_cm(h_0))
        # current cell state
        h_t = (1-z_g) * h_0 + z_g * h_tilda

        mask =  input_mask.view(-1,1).expand_as(h_t)
        ho = h_t *mask + h_0 * (1-mask) # (1,b_s,hids)
        return ho

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(batch_size, self.hidden_size).zero_()))

    def init_weights(self):
        initrange = 0.1
        self.weight_fm.weight.data.uniform_(-initrange, initrange)
        self.weight_im.weight.data.uniform_(-initrange, initrange)
        self.weight_cm.weight.data.uniform_(-initrange, initrange)
        self.weight_fx.weight.data.uniform_(-initrange, initrange)
        self.weight_ix.weight.data.uniform_(-initrange, initrange)
        self.weight_cx.weight.data.uniform_(-initrange, initrange)

        self.weight_fm.bias.data.fill_(0)
        self.weight_im.bias.data.fill_(0)
        self.weight_cm.bias.data.fill_(0)
        self.weight_fx.bias.data.fill_(0)
        self.weight_ix.bias.data.fill_(0)
        self.weight_cx.bias.data.fill_(0)


class  KimCNN(nn.Module):
    
    def __init__(self, size_filter, n_out_kernel, embsize, drate = 0.01):
        super(KimCNN,self).__init__()
        self.size_filter = size_filter 
        self.drate = drate
        self.n_filter = n_out_kernel
        Ci = 1 #n_in_kernel
        Co = self.n_filter # args.kernel_num
        Ks = [1,2,3,4,5]
        self.n_out = len(Ks) * self.n_filter
        #self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, embsize)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3) #(N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        '''
        x: (b_s, len, embsize)
        '''
        x = x.unsqueeze(1) # (N,Ci,len,embsize)
        x1 = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,len), ...]*len(Ks)
        self.fea_map = x1
        x2 = [nn.MaxPool1d(i.size(2))(i).squeeze(2) for i in x1] #[(N,Co), ...]*len(Ks)
        x3 = torch.cat(x2, 1) # (b_s, co*len(Ks))
        return x3

class  CNN(nn.Module):
    
    def __init__(self, len_filter, n_out_kernel, embsize):
        super(CNN,self).__init__()
        self.drate = drate
        self.n_filter = n_out_kernel
        Ci = 1 #n_in_kernel
        Co = self.n_filter # args.kernel_num
        #self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.conv = nn.Conv2d(Ci, Co, (len_filter, embsize),padding = (int((len_filter)/2),0)) 
        self.conv1d = nn.Conv1d(Co, 2*Co, len_filter, padding = (int((len_filter)/2),0))
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.poolsize = 3

    def conv_and_pool(self, x, conv):
        '''
        x: (b_s, 1, W, l)
        return: (b_s,co,(W-len_filter+1)/poolsize) 
        '''
        x = F.relu(conv(x)).squeeze(3) #(N,Co,W)
        x = F.max_pool1d(x, self.poolsize)
        return x

    def forward(self, x):
        '''
        x: (b_s, len, embsize)
        '''
        x = x.unsqueeze(1) # (N,Ci,len,embsize)
        x = self.conv_and_pool(x,self.conv)
        x = self.conv1d(x)
        x = F.max_pool1d(x, x.size()[2]).squeeze(2)
        return x

class CNN_Text(nn.Module):

    def __init__(self, size_filter, n_out_kernel, embsize, drate=0.05):
        """
        - size_filter: like [1, 2, 3, 4, 5]
        - n_out_kernel: like 10
        """
        super(CNN_Text,self).__init__()
        self.size_filter = size_filter
        assert(size_filter > 4)
        self.drate = drate
        self.embsize = embsize
        Co = n_out_kernel
        Ci = 1 #n_in_kernel
        size_filter += 1 
        # poch = int(round(size_filter/5.0)) if (size_filter/5.0)>1 else 1
        Ks = [3,4,5]# list(range(1,size_filter, poch))
        self.n_out = len(Ks) * n_out_kernel
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, embsize)) for K in Ks])

    def forward(self, x):
        '''
        x: (b_s, len, embsize)
        '''
        length = x.size(1)
        x = x.unsqueeze(1) # (N,Ci,len,embsize), N = bs, Ci = 1
        x = [F.relu(conv(x)).squeeze(3)[:,:,:length] for conv in self.convs1] #[(N,Co,len), ...]*len(Ks)

        x = torch.cat(x, 1) # (N, co*len(Ks), len)
        x = x.permute(2, 0, 1)  # (length, batch, len(KS)*Co = embsize)
        return x

class  FIX_CNN(nn.Module):
    
    def __init__(self, len_filter, n_out_kernel, embsize):
        super(FIX_CNN,self).__init__()
        self.n_filter = int(n_out_kernel)
        Ci = 1 #n_in_kernel
        Co = int(self.n_filter) # args.kernel_num
        #self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.conv = nn.Conv2d(int(Ci), int(Co/2), (len_filter, embsize),padding = (int((len_filter)/2),0)) 
        self.conv1d = nn.Conv1d(int(Co/2), Co, len_filter, padding = int((len_filter)/2))
        self.n_out = self.n_filter * 2
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''

    def conv_and_pool(self, x, conv):
        '''
        x: (b_s, 1, W, l)
        return: (b_s,co,(W-len_filter+1)/poolsize) 
        '''
        x = F.relu(conv(x)).squeeze(3) #(N,Co,W)
        x = F.max_pool1d(x, self.poolsize)
        return x

    def forward(self, x):
        '''
        x: (b_s, len, embsize)
        '''
        x = x.unsqueeze(1) # (N,Ci,len,embsize)
        x = F.relu(self.conv(x)).squeeze(3) #(N,Co,len)
        x = self.conv1d(x).permute(0,2,1) # (N,2*Co,len)
        return x

class LayerNorm(nn.Module):

    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(size))
        self.beta = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta




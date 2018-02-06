import numpy, time, os
import re
try: # python2
    import cPickle
    from itertools import izip
except ImportError:
    import _pickle as cPickle
import logging,sys
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def say(s, stream=sys.stdout):
    stream.write("{}".format(s))
    stream.flush()


class MyRecorder(object):
    """
    MyRecorder keeps a dict of things, every entry in the dict is a list.
    """

    def __init__(self, savefolder=None):
        self.content = {}
        self.savefolder = savefolder

    def add(self, content_dict):
        """
        content_dict should be a dict, key is the entry name; value is the entry content.
        """
        for key,value in content_dict.iteritems():
            if not self.content.has_key(key):
                self.content[key] = [value]
            else:
                self.content[key].append(value)

    def save(self):
        with open(os.path.join(self.savefolder, "records.pkl"), 'w') as f:
            cPickle.dump(self.savefolder, f, -1)

    def visualize(self, IFSHOW=True):
        for key, value in self.content.iteritems():
            if isinstance(value[0], int) or isinstance(value[0], float):
                plt.figure(key)
                plt.title(key)
                series = numpy.array(value)
                plt.plot(series, label=key)
                plt.legend(loc='best')
                if self.savefolder:
                    plt.savefig(os.path.join(self.savefolder, key))
            elif isinstance(value[0], numpy.float):
                pass
            else:
                print("type wrong")
            if IFSHOW:
                plt.show()


def load_dataset(*filenames):
    data = []
    for filename in filenames:
        if filename.endswith(".pkl"):
            with open(filename, 'r') as f:
                data.append(cPickle.load(f))
        elif filename.endswith(".txt"):
            data.append(numpy.loadtxt(filename, dtype='int64'))
        else:
            raise NotImplementedError
    return data


def save(model, filename):
    say('saving model to {} ...\n'.format(filename))
    params_value = [value.get_value() for value in model.params]
    with open(filename, 'w') as f:
        cPickle.dump(params_value, f, -1)


def load(model, filename):
    say('load model from {} ...\n'.format(filename))
    if filename.endswith('.pkl'):
        with open(filename, 'r') as f:
            params_value = cPickle.load(f)
        assert len(params_value) == len(model.params)
        for i in xrange(len(model.params)):
            model.params[i].set_value(params_value[i])
    else:
        raise NotImplementedError


class param_init(object):

    def __init__(self,**kwargs):

        self.shared = kwargs.pop('shared', True)

    def param(self, size, init_type=None, name=None, **kwargs):
        try:
            if init_type is not None:
                func = getattr(self, init_type)
            elif len(size) == 1:
                func = getattr(self, 'constant')
            elif size[0] == size[1]:
                func = getattr(self, 'orth')
            else:
                func = getattr(self, 'normal')
        except AttributeError:
            logger.error('AttributeError, {}'.format(init_type))
        else:
            param = func(size, name=None, **kwargs)
        return param

    def uniform(self, size, name=None,**kwargs):
        #low = kwargs.pop('low', -6./sum(size))
        #high = kwargs.pop('high', 6./sum(size))
        low = kwargs.pop('low', -0.01)
        high = kwargs.pop('high', 0.01)
        rng = kwargs.pop('rng', numpy.random.RandomState(1234))
        param = numpy.asarray(
            rng.uniform(low=low, high=high, size=size),
            dtype=theano.config.floatX)
        if self.shared:
            param = theano.shared(value=param, borrow=True, name=name)
        return param

    def normal(self, size, name=None,**kwargs):
        loc = kwargs.pop('loc', 0.)
        scale = kwargs.pop('scale', 0.05)
        rng = kwargs.pop('rng', numpy.random.RandomState(1234))
        param = numpy.asarray(
            rng.normal(loc=loc, scale=scale, size=size),
            dtype=theano.config.floatX)
        if self.shared:
            param = theano.shared(value=param, borrow=True, name=name)
        return param

    def constant(self, size, name=None,**kwargs):
        value = kwargs.pop('scale', 0.)
        param = numpy.ones(size, dtype=theano.config.floatX)*value
        if self.shared:
            param = theano.shared(value=param, borrow=True, name=name)
        return param

    def orth(self, size, name=None,**kwargs):
        scale = kwargs.pop('scale', 1.)
        rng = kwargs.pop('rng', numpy.random.RandomState(1234))
        if len(size) != 2:
            raise ValueError
        if size[0] == size[1]:
            M = rng.randn(*size).astype(theano.config.floatX)
            Q, R = numpy.linalg.qr(M)
            Q = Q * numpy.sign(numpy.diag(R))
            param = Q*scale
            if self.shared:
                param = theano.shared(value=param, borrow=True, name=name)
            return param
        else:
            M1 = rng.randn(size[0], size[0]).astype(theano.config.floatX)
            M2 = rng.randn(size[1], size[1]).astype(theano.config.floatX)
            Q1, R1 = numpy.linalg.qr(M1)
            Q2, R2 = numpy.linalg.qr(M2)
            Q1 = Q1 * numpy.sign(numpy.diag(R1))
            Q2 = Q2 * numpy.sign(numpy.diag(R2))
            n_min = min(size[0], size[1])
            param = numpy.dot(Q1[:, :n_min], Q2[:n_min, :]) * scale
            if self.shared:
                param = theano.shared(value=param, borrow=True, name=name)
            return param


def create_index2word(word2index_dict):
    index2word = [0]*len(word2index_dict)
    for key, value in word2index_dict.iteritems():
        index2word[value] = key
    return index2word


class mylog(object):

    def __init__(self,dirname):
        filename = dirname+'/logging.log'
        logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=filename,
                        filemode='w')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)-4s: %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    def logging(self,str):
        logging.info("\n"+str)




def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),;.!?\'\`]", " ", string)
    string = re.sub(r"\. \. \.", ".", string)
    string = re.sub(r"\. \.", ".", string)

    string = re.sub(r"\.\.\.\.\.", " . ", string)
    string = re.sub(r"\.\.\.\.", " . ", string)
    string = re.sub(r"\.\.\.", " . ", string)
    string = re.sub(r"\.\.", " . ", string)
    string = re.sub(r"\.", " . ", string)


    string = re.sub(r"\'", " \' ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r";", " ; ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

def cal_RI(list1,list_target):
    assert len(list1)==len(list_target)
    len_l = len(list1)
    num = 0
    for i in range(len(list1)):
        # c, the number of pairs of elements in S that are in the
        #same substs in X and in different subsets in Y
        for j in range(i):
            if list_target[i] == list_target[j] and list1[i] != list1[j]:
                num +=1

            if list_target[i] != list_target[j] and list1[i] == list1[j]:
                num +=1
    C = len_l*(len_l-1)/2.0
    res =(C-num)/(C)
    return res

if __name__ == "__main__":
    list1 = [1,1,2,2,3]
    list2 = [1,2,2,3,3]
    # cal_RI(list1,list2)
    read_word2vec()


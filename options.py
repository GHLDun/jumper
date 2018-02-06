
import argparse
def load_arguments():

    parser = argparse.ArgumentParser(description='Graph Transition Model')
    parser.add_argument('--eval', action='store_true',
                        help='training flag')
    parser.add_argument('--data_dir', type=str, default='data/task_1_train.txt',
                        help='location of the data corpus')
    parser.add_argument('--task', type= str, default='oi-level',
                        help='task name')
    parser.add_argument('--n_sample', type=int, default=1000,
                        help='number of sample')
    parser.add_argument('--embsize', type=int, default=300,
                        help='size of word embeddings')
    parser.add_argument('--n_hids', type=int, default=4,
                        help='number of hidden units per layer')
    parser.add_argument('--n_filter', type=int, default=50,
                        help='number of kernel')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default= 1,
                        help='initial learning rate')
    parser.add_argument('--clip_grad', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=1,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                        help='batch size')
    parser.add_argument('--model', type=int, default=None,
                        help='model type')
    parser.add_argument('--drate', type=float, default= None,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--prate', type=float, default= None,
                        help='exploration applied to layers (0 = noexploration)')
    parser.add_argument('--save', type = str, default = None,
                        help='file path to save model')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--parallel', action='store_true',
                        help='use multi gpus')
    parser.add_argument('--interval', type=int, default=10, metavar='N',
                        help='report interval')
    parser.add_argument('--mark', type=str,  default='mark',
                        help='note to highlignt')
    parser.add_argument('--optim', type=str,  default='adadelta',
                        help='the name of optimizer,adadelta, sgd')
    parser.add_argument('--nofeed', action='store_true',
                        help='use feed back')
    parser.add_argument('--mode', type=int, default=0,
                        help='model mode, 0-train,1-eval,2-cv,3-looptest, 4-casestudy')
    args = parser.parse_args()

    return args


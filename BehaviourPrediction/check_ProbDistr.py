'''
Build a tweet sentiment analyzer
'''

from __future__ import print_function
import six.moves.cPickle as pickle

from collections import OrderedDict
import sys
import random
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import dataset_allinone as dataset

datasets = {'imdb': (dataset.load_data, dataset.prepare_data)}

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def get_dataset(name):
    return datasets[name][0], datasets[name][1]



def unzip(zipped):

    new_params = OrderedDict()
    param_file = open('param_file_MoreHigher_regularized_2e-4.txt', 'w')
    for kk, vv in zipped.items():
        print(kk)
        aa = vv.get_value()
        new_params[kk] = aa
        if kk != "lstm_W":
            param_file.write(kk)
            param_file.write('\n')
        if 'b' not in kk:
            print(len(aa), len(aa[0]))
            for i in range(0,len(aa[0])):
                values = []
                for j in range(0,len(aa)):
                    value = aa[j, i]
                    values.append(str(value))
                param_file.write(' '.join((values)))
                param_file.write('\n')
        else:
            print(len(aa))
            values = []
            for value in aa:
                values.append(str(value))
            param_file.write(' '.join((values)))
            param_file.write('\n')
    param_file.close()

    return new_params

def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)





def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    # embedding we do not need embedding for words!
    # randn = numpy.random.rand(options['n_words'],
    #                           options['dim_proj'])

    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])
    # classifier
    # params['U1'] = 0.01 * numpy.random.randn(options['dim_proj'],
    #                                          options['ydim1']).astype(config.floatX)
    # params['b1'] = numpy.zeros((options['ydim1'],)).astype(config.floatX)
    #
    # params['U2'] = 0.01 * numpy.random.randn(options['dim_proj'],
    #                                          options['ydim2']).astype(config.floatX)
    # params['b2'] = numpy.zeros((options['ydim2'],)).astype(config.floatX)
    #
    # params['U3'] = 0.01 * numpy.random.randn(options['dim_proj']+ options['dim_finalx'],
    #                                         options['ydim3']).astype(config.floatX)
    # params['b3'] = numpy.zeros((options['ydim3'],)).astype(config.floatX)
    #
    # params['U4'] = 0.01 * numpy.random.randn(options['dim_proj'] + options['dim_finalx'],
    #                                         options['ydim4']).astype(config.floatX)
    # params['b4'] = numpy.zeros((options['ydim4'],)).astype(config.floatX)
    #
    # params['U5'] = 0.01 * numpy.random.randn(options['dim_proj']+ options['dim_finalx'],
    #                                         options['ydim5']).astype(config.floatX)
    # params['b5'] = numpy.zeros((options['ydim5'],)).astype(config.floatX)

    params['W1'] = 0.01 * numpy.random.randn(10,
                                             options['ydim1']).astype(config.floatX)
    params['b1'] = numpy.zeros((options['ydim1'],)).astype(config.floatX)

    params['W2'] = 0.01 * numpy.random.randn(10,
                                             options['ydim2']).astype(config.floatX)
    params['b2'] = numpy.zeros((options['ydim2'],)).astype(config.floatX)

    params['W3'] = 0.01 * numpy.random.randn(10 + options['dim_finalx'],
                                             options['ydim3']).astype(config.floatX)
    params['b3'] = numpy.zeros((options['ydim3'],)).astype(config.floatX)

    params['W4'] = 0.01 * numpy.random.randn(10 + options['dim_finalx'],
                                             options['ydim4']).astype(config.floatX)
    params['b4'] = numpy.zeros((options['ydim4'],)).astype(config.floatX)

    params['W5'] = 0.01 * numpy.random.randn(10 + options['dim_finalx'],
                                             options['ydim5']).astype(config.floatX)
    params['b5'] = numpy.zeros((options['ydim5'],)).astype(config.floatX)

    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def get_layer(name):
    fns = layers[name]
    return fns


def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)


def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    # W = numpy.concatenate([ortho_weight(options['dim_proj']),
    #                        ortho_weight(options['dim_proj']),
    #                        ortho_weight(options['dim_proj']),
    #                        ortho_weight(options['dim_proj'])], axis=1)
    # params[_p(prefix, 'W')] = W
    # U = numpy.concatenate([ortho_weight(options['dim_proj']),
    #                        ortho_weight(options['dim_proj']),
    #                        ortho_weight(options['dim_proj']),
    #                        ortho_weight(options['dim_proj'])], axis=1)
    # params[_p(prefix, 'U')] = U

    params[_p(prefix, 'W')] = 0.01 * numpy.random.randn(options['dim_proj'],
                                                        10 * 4).astype(config.floatX)
    params[_p(prefix, 'U')] = 0.01 * numpy.random.randn(10,
                                             10*4).astype(config.floatX)
    # b = numpy.zeros((4 * options['dim_proj'],))
    # params[_p(prefix, 'b')] = b.astype(config.floatX)
    params[_p(prefix, 'b')] = numpy.zeros((4*10, )).astype(config.floatX)

    return params


def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        # i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        # f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        # o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        # c = tensor.tanh(_slice(preact, 3, options['dim_proj']))
        i = tensor.nnet.sigmoid(_slice(preact, 0, 10))
        f = tensor.nnet.sigmoid(_slice(preact, 1, 10))
        o = tensor.nnet.sigmoid(_slice(preact, 2, 10))
        c = tensor.tanh(_slice(preact, 3, 10))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    #dim_proj = options['dim_proj']
    dim_proj=10
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]


# ff: Feed Forward (normal neural net), only useful to put after lstm
#     before the classifier.
layers = {'lstm': (param_init_lstm, lstm_layer)}


def sgd(lr, tparams, grads, x, mask, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, mask, y1, y2, y3, y4, y5,  final_x, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y1, y2, y3, y4, y5, final_x], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, mask, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.items()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def build_model(tparams, options):
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.tensor3('x', dtype='float64')
    mask = tensor.matrix('mask', dtype=config.floatX)
    y1 = tensor.vector('y1', dtype='int64')
    y2 = tensor.vector('y2', dtype='int64')
    y3 = tensor.vector('y3', dtype='int64')
    y4 = tensor.vector('y4', dtype='int64')
    y5 = tensor.vector('y5', dtype='int64')
    final_feature = tensor.matrix('final_feature', dtype='float64')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    #emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples,options['dim_proj']])
    proj = get_layer(options['encoder'])[1](tparams, x, options,
                                            prefix=options['encoder'],
                                            mask=mask)
    if options['encoder'] == 'lstm':
        #proj = (proj * mask[:, :, None]).max(axis=0)
        ##proj = proj / mask.sum(axis=0)[:, None]
        proj = proj[-1]
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)

    pred1 = tensor.nnet.softmax(tensor.dot(proj, tparams['W1']) + tparams['b1'])
    pred2 = tensor.nnet.softmax(tensor.dot(proj, tparams['W2']) + tparams['b2'])

    proj_h = proj
    proj = tensor.concatenate([proj, final_feature], axis=1)

    pred3 = tensor.nnet.softmax(tensor.dot(proj, tparams['W3']) + tparams['b3'])
    pred4 = tensor.nnet.softmax(tensor.dot(proj, tparams['W4']) + tparams['b4'])
    pred5 = tensor.nnet.softmax(tensor.dot(proj, tparams['W5']) + tparams['b5'])

    f_pred_prob1 = theano.function([x, mask], pred1, name='f_pred_prob1')
    f_pred1 = theano.function([x, mask], pred1.argmax(axis=1), name='f_pred1')

    f_pred_prob2 = theano.function([x, mask], pred2, name='f_pred_prob2')
    f_pred2 = theano.function([x, mask], pred2.argmax(axis=1), name='f_pred2')

    f_pred_prob3 = theano.function([x, mask, final_feature], pred3, name='f_pred_prob3')
    f_pred3 = theano.function([x, mask, final_feature], pred3.argmax(axis=1), name='f_pred3')

    f_pred_prob4 = theano.function([x, mask, final_feature], pred4, name='f_pred_prob4')
    f_pred4 = theano.function([x, mask, final_feature], pred4.argmax(axis=1), name='f_pred4')

    f_pred_prob5 = theano.function([x, mask, final_feature], pred5, name='f_pred_prob5')
    f_pred5 = theano.function([x, mask, final_feature], pred5.argmax(axis=1), name='f_pred5')



    off = 1e-8
    if pred3.dtype == 'float16':
        off = 1e-6

    cost1 = -tensor.log(pred1[tensor.arange(n_samples), y1] + off).mean()
    cost2 = -tensor.log(pred2[tensor.arange(n_samples), y2] + off).mean()
    cost3 = -tensor.log(pred3[tensor.arange(n_samples), y3] + off).mean()
    cost4 = -tensor.log(pred4[tensor.arange(n_samples), y4] + off).mean()
    cost5 = -tensor.log(pred5[tensor.arange(n_samples), y5] + off).mean()

    cost = cost1 + cost2 + cost3 + cost4 + cost5
    cost_reward = cost3 + cost4 + cost5
    return use_noise, x, mask, y1, y2, y3, y4, y5, final_feature, \
           f_pred_prob1, f_pred_prob2, f_pred_prob3, f_pred_prob4, f_pred_prob5,\
           f_pred1, f_pred2, f_pred3, f_pred4, f_pred5, cost, cost_reward, proj_h


def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print('%d/%d samples classified' % (n_done, n_samples))

    return probs


#train_file_collect = open('collect_test_file.txt', 'w')
#train_file_cart = open('cart_test_file.txt', 'w')
#train_file_buy = open('buy_test_file.txt', 'w')

def pred_error(f_pred1, f_pred2, f_pred3, f_pred4, f_pred5,
               f_pred_prob1, f_pred_prob2, f_pred_prob3, f_pred_prob4, f_pred_prob5,
               prepare_data, data, iterator,  f_cost, f_h):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err1 = 0
    valid_err2 = 0
    valid_err3 = 0
    valid_err4 = 0
    valid_err5 = 0

    total_num = 0
    total_cost = 0
    for _, valid_index in iterator:
        x, mask, y1, y2, y3, y4, y5, final_x = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        #print(x)
        #print(final_x)
        proj_h = f_h(x, mask)
        for h, x_, y_3, y_4, y_5 in zip(proj_h, final_x, y3, y4, y5):
            train_file_collect.write(str(y_3))
            train_file_collect.write("|")
            train_file_cart.write(str(y_4))
            train_file_cart.write("|")
            train_file_buy.write(str(y_5))
            train_file_buy.write("|")
            for number in h:
                train_file_collect.write(str(number))
                train_file_collect.write(" ")
                train_file_cart.write(str(number))
                train_file_cart.write(" ")
                train_file_buy.write(str(number))
                train_file_buy.write(" ")
            #train_file_collect.write("#")
            #train_file_cart.write("#")
            #train_file_buy.write("#")
            for number in x_:
                train_file_collect.write(str(number))
                train_file_collect.write(" ")
                train_file_cart.write(str(number))
                train_file_cart.write(" ")
                train_file_buy.write(str(number))
                train_file_buy.write(" ")
            train_file_collect.write("\n")
            train_file_cart.write("\n")
            train_file_buy.write("\n")

        total_num += len(valid_index)
        preds1 = f_pred1(x, mask)
        #pred_prob1 = f_pred_prob1(x, mask)
        valid_err1 += (preds1 == y1).sum()

        preds2 = f_pred2(x, mask)
        #pred_prob2 = f_pred_prob2(x, mask)
        valid_err2 += (preds2 == y2).sum()

        preds3 = f_pred3(x, mask, final_x)
        #pred_prob3 = f_pred_prob3(x, mask, final_x)
        valid_err3 += (preds3 == y3).sum()

        preds4 = f_pred4(x, mask, final_x)
        #pred_prob4 = f_pred_prob4(x, mask, final_x)
        valid_err4 += (preds4 == y4).sum()

        preds5 = f_pred5(x, mask, final_x)
        valid_err5 += (preds5 == y5).sum()
        #pred_prob5 = f_pred_prob5(x, mask, final_x)
        total_cost += f_cost(x, mask, y3, y4, y5, final_x)
        #print(proj_h)
        #print(preds1, preds2, preds3, preds4, preds5)
        #print(pred_prob1, pred_prob2, pred_prob3, pred_prob4, pred_prob5)

        ##targets = numpy.array(data[1])[valid_index]

        #print(preds)
        #print(y)

    valid_err1 = 1. - numpy_floatX(valid_err1) / total_num
    valid_err2 = 1. - numpy_floatX(valid_err2) / total_num
    valid_err3 = 1. - numpy_floatX(valid_err3) / total_num
    valid_err4 = 1. - numpy_floatX(valid_err4) / total_num
    valid_err5 = 1. - numpy_floatX(valid_err5) / total_num
    cost = total_cost/len(iterator)


    return (valid_err1, valid_err2, valid_err3, valid_err4, valid_err5, cost)


def train_lstm(
    dim_proj=18,  # word embeding dimension and LSTM number of hidden units.
    dim_finalx=15,
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=50,  # The maximum number of epoch to run
    dispFreq=10000,  # Display to stdout the training progress every N updates
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm',  # TODO: can be removed must be lstm.
    validFreq=10000,  # Compute the validation error after this number of update.
    saveFreq=10000,  # Save the parameters after every saveFreq updates
    maxlen=100,  # Sequence longer then this get ignored
    batch_size=20,  # The batch size during training.
    valid_batch_size=64,  # The batch size used for validation/test set.
    dataset='imdb',

    # Parameter for extra option
    noise_std=0.3,
    use_dropout=False,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=True,  # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
):

    # Model options
    model_options = locals().copy()
    print("model options", model_options)

    load_data, prepare_data = get_dataset(dataset)

    print('Loading data')
    train, valid, test = load_data(valid_portion=0.0,
                                   maxlen=maxlen)


    ##ydim = numpy.max(train[1]) + 1

    model_options['ydim1'] = 7
    model_options['ydim2'] = 6
    model_options['ydim3'] = 2
    model_options['ydim4'] = 2
    model_options['ydim5'] = 2

    print('Building model')
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options)

    if reload_model:
        load_params('lstm_model_MoreHigher_regularized_2e-4.npz', params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, mask, y1, y2, y3, y4, y5, final_x,
     f_pred_prob1,  f_pred_prob2, f_pred_prob3, f_pred_prob4, f_pred_prob5,
     f_pred1, f_pred2, f_pred3, f_pred4, f_pred5,
     cost, cost_reward, proj_h) = build_model(tparams, model_options)



    f_cost_reward = theano.function([x, mask, y3, y4, y5, final_x], cost_reward, name='f_cost_reward')
    f_h = theano.function([x, mask], proj_h, name='f_h')


    prob_file = open("test_prob_distribution_MoreHigher_regularized_2e-4.txt", "w")
    uidx = 0
    kf = get_minibatches_idx(len(test[0]), batch_size, shuffle=False)
    test_size = len(test[0])
    test_correct = 0
    print('uidx begin')
    for _, train_index in kf:
        if uidx%1000==0:
            print(uidx)
        uidx += 1
        use_noise.set_value(1.)

        # Select the random examples for this minibatch
        y = [test[1][t] for t in train_index]
        x = [test[0][t] for t in train_index]

        # Get the data in numpy.ndarray format
        # This swap the axis!
        # Return something of shape (minibatch maxlen, n samples)

        x, mask, y1, y2, y3, y4, y5, final_x = prepare_data(x, y)
        # print(x)
        # print(y)
        #proj_h = f_h(x, mask)

        pred_prob1 = f_pred_prob1(x, mask)
        preds1 = f_pred1(x, mask)
        #pred_prob1 = f_pred_prob1(x, mask)
        test_correct += (preds1 == y1).sum()
        x = x.swapaxes(0, 1)
        mask = mask.swapaxes(0, 1)
        
        if (len(x) != len(y1)):
            print("wrong")
            print(len(x))
            print(len(y1))
        for _prob1, _label, _f, _m  in zip(pred_prob1, y1, x, mask):
            prob_file.write(str(_label+1))
            prob_file.write("\t")
            prob_file.write((','.join([str(i) for i in _prob1])))
            prob_file.write("\t")


            f_str = []
            for step, m_i in zip(_f, _m):
                if m_i == 0.0:
                    break
                step_str = []
                for index, value in enumerate(step):
                    if value ==1 :
                        if index <= 7:
                            step_str.append(str(index))
                        elif index <= 14:
                            step_str.append(str(index-8))
                   
                step_str.append(str(int(step[15])) + str(int(step[16]))  + str(int(step[17])) )
                f_str.append(','.join(step_str))
            prob_file.write('#'.join(f_str) + '\n')
    prob_file.close()
            # train_file_collect.write(str(y_3))
            # train_file_collect.write("|")
            # train_file_cart.write(str(y_4))
            # train_file_cart.write("|")
            # train_file_buy.write(str(y_5))
            # train_file_buy.write("|")
            # for number in h:
            #     train_file_collect.write(str(number))
            #     train_file_collect.write(" ")
            #     train_file_cart.write(str(number))
            #     train_file_cart.write(" ")
            #     train_file_buy.write(str(number))
            #     train_file_buy.write(" ")
            # #train_file_collect.write("#")
            # #train_file_cart.write("#")
            # #train_file_buy.write("#")
            # for number in x_:
            #     train_file_collect.write(str(number))
            #     train_file_collect.write(" ")
            #     train_file_cart.write(str(number))
            #     train_file_cart.write(" ")
            #     train_file_buy.write(str(number))
            #     train_file_buy.write(" ")
            # train_file_collect.write("\n")
            # train_file_cart.write("\n")
            # train_file_buy.write("\n")

    #kf_test = kf_test[0:1]
    unzip(tparams)



    #train_file_buy.close()
    #train_file_collect.close()
    #train_file_cart.close()

    print("precision", float(test_correct)/test_size)
    return  0


if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    train_lstm(
        max_epochs=100,
        test_size=10000,
    )




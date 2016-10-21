import cPickle
import gzip
import os

import numpy
import theano
import Preprocess_reward
import numpy as np




def Feature2emb(single_behavior):
    Final = []

    power_emb = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    power_emb[int(single_behavior[0])-100] = 1.0
    brand_emb = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    brand_emb[int(single_behavior[1])-310] = 1.0
    #xiaoliang_emb = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #xiaoliang_emb[int(single_behavior[8])] = 1.0

    reward = []
    for number in single_behavior[2].split('_'):
        reward.append(float(number))


    #Final.append(float(single_behavior[0]))
    #Final.extend(dt_type2emb(single_behavior[1]))
    #if len(single_behavior[2])==0:
    #    Final.append(0.0)
    #else:
    #    Final.append(float(single_behavior[2]))

    #Final.append(float(single_behavior[3]))
    Final.extend(power_emb)
    Final.extend(brand_emb)
    #Final.append(float(single_behavior[6])-60.0)
    #Final.append(float(single_behavior[7])-10.0)
    #Final.extend(xiaoliang_emb)
    Final.extend(reward)
   # print(single_behavior)
   
    return Final

def final_featureEmb(single_behavior):
    Final = []

    power_emb = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    power_emb[int(single_behavior[0]) - 100] = 1.0
    brand_emb = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    brand_emb[int(single_behavior[1]) - 310] = 1.0

    Final.extend(power_emb)
    Final.extend(brand_emb)
    #Final.extend(xiaoliang_emb)

    return Final


def x2emb(s):
    x = []
    for single_behavior in s:
        x.append(Feature2emb(single_behavior))
    return x

def y2label(s):
    y = []
    y.append(int(s[0]) - 100-1)
    y.append(int(s[1]) - 310-1)
    for number in s[2].split('_'):
        y.append(int(number))

    return y


def prepare_data(seqs, labels, maxlen=40):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:

                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None
    #print(seqs)
    final_labels = []
    final_features = []
    for y in labels:
        final_features.append(final_featureEmb(y))
        final_labels.append(y2label(y))
    y0 = []
    y1 = []
    y2 = []
    y3 = []
    y4 = []

    for trible in final_labels:
        y0.append(trible[0])
        y1.append(trible[1])
        y2.append(trible[2])
        y3.append(trible[3])
        y4.append(trible[4])


    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples, 18)).astype('float64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        
        x[:lengths[idx], idx] = x2emb(s)
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, y0, y1, y2, y3, y4, final_features




def load_data(valid_portion=0.0, maxlen=None):
    all_data = Preprocess_reward.get_dataset()
   # train_set = [all_data[0][0:300000], all_data[1][0:300000]]
   # test_set = [all_data[0][300000:350000], all_data[1][300000:350000]]
    train_set = [[], []]
    test_set = [[], []]

    for i in range(len(all_data[0])):
        if i%10 == 0:
            test_set[0].append(all_data[0][i])
            test_set[1].append(all_data[1][i])
        else:
            train_set[0].append(all_data[0][i])
            train_set[1].append(all_data[1][i])

    ##f.close()
    if maxlen and 0:
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set[0], train_set[1]):
            if len(x) < maxlen+100:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y


    # split training set into validation set
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = numpy.random.permutation(n_samples)
    valid_portion = 0.0
    n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)
    print('the train size is', len(train_set[0]))

    print('the test size is', len(test_set[0]))

    return train_set, valid_set, test_set

train, valid, test = load_data(0.1*0.66, maxlen=100)

#print "the data has been load!"
#print train[0][10]
#print x2emb(train[0][10])
#print train[1][10]
#print Feature2emb(train[1][10])

#print train[0][11]
#print x2emb(train[0][11])
#print train[1][11]
#print Feature2emb(train[1][11])

x, x_mask, label1, label2, label3, label4, label5, final_feature = prepare_data([train[0][10], train[0][11]], [train[1][10], train[1][11]])
print(train[1][10])
print(train[1][11])
print(train[0][10])
print(train[0][11])
print("x")
print x
print("x_mask")
print x_mask
print label1
print label2
print label3
print final_feature


# all_data = PreProcess.get_dataset()
# power_0tags = 0
# brand_0tags = 0
# power_brand_0tags = 0
# all_0tags = 0
# reward_0tags = 0
# for line in all_data[1]:
#     if line[0] == '100' :
#         power_0tags += 1
#     if line[1] == '310':
#         brand_0tags += 1
#         if line[0] == '100':
#             power_brand_0tags += 1
#             if line[2] == '0_0_0':
#                 all_0tags += 1
#     if line[2] == '0_0_0':
#         reward_0tags += 1
#
# print(len(all_data[1]))
#
# print(float(power_0tags)/len(all_data[1]), float(brand_0tags)/len(all_data[1]), float(reward_0tags)/len(all_data[1]),
#       float(power_brand_0tags)/len(all_data[1]), float(all_0tags)/len(all_data[1]))



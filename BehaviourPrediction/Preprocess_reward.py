__author__ = 'cairui'
import cPickle
import gzip
import os

import numpy
import theano
import numpy as np

def get_dataset():
    #All_data = open('alltrain_50w').read().splitlines()
    #All_data_2 = open('alldata_reward').read().splitlines()
    #All_data.extend(All_data_2)
    All_data = open('balance_dataset_MoreHigher.txt').read().splitlines()
    train_samples = []
    train_samples_x = []
    train_samples_y = []
    index = 0
    reward_number = 0
    noreward_number = 0

    level_1 = 0
    total_sum = 0

    for line in All_data:
        index+=1
        train_sample_x = []
        train_sample_y = []

        user_id, power_tags_str, brand_tags_str, rewards_tags_str = line.split('\t')

        power_tags = power_tags_str.split(',')
        brand_tags = brand_tags_str.split(',')
        rewards_tags = rewards_tags_str.split(',')

        ## remove the data point with 2 '0' t
        ## banlance the rewards, no rewards trainning data


        #print rewards_tags[-1]
        for i in range(0, len(power_tags)):
            single_behavior = []
            single_behavior.append(power_tags[i])
            single_behavior.append(brand_tags[i])
            single_behavior.append(rewards_tags[i])

            if i == len(power_tags)-1:
                train_sample_y = single_behavior
                total_sum += 1
                if power_tags[i] == '101':
                    level_1 += 1
            else:
                train_sample_x.append(single_behavior)
        train_samples_x.append(train_sample_x)

        #if len(train_sample_x) > 5:
        #    train_sample_x = train_sample_x[-5:]

        train_samples_y.append(train_sample_y)


    train_samples.append(train_samples_x)

    train_samples.append(train_samples_y)
    print("the reward number is:", reward_number)
    print("the no reward number is", noreward_number)
    print("total power_tags is:", total_sum)
    print("level 1 power_tags is:", level_1)
    print(float(level_1)/total_sum)
    return train_samples


train = get_dataset()
print(len(train))
print(train[0][1])
print(train[1][1])

print(train[0][2])
print(train[1][2])

print(train[0][3])
print(train[1][3])


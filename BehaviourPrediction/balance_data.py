__author__ = 'cairui'
import cPickle
import gzip
import os

import numpy
import theano
import numpy as np

def balance_dataset():


    balance_file = open('balance_dataset_MoreHigher.txt', 'w')

    All_data = open('train_1500w').read().splitlines()
    train_samples = []
    train_samples_x = []
    train_samples_y = []
    index = 0
    reward_number = 0
    noreward_number = 0

    level_1 = 0
    level_num = [0,0,0,0,0,0,0]
    level_limit = [10000,10000,20000,30000,40000,50000,50000]
    level_noReward_num = [0,0,0,0,0,0,0]
    total_sum = 0

    for line in All_data:
        index+=1

        train_sample_x = []
        train_sample_y = []
        user_id,  power_tags_str, brand_tags_str, rewards_tags_str = line.split('\t')
        power_tags = power_tags_str.split(',')
        brand_tags = brand_tags_str.split(',')
        rewards_tags = rewards_tags_str.split(',')

        ## remove the data point with 2 '0' tags
        if power_tags[-1]=='100' or brand_tags[-1]=='310':
            continue

        ## banlance the rewards, no rewards trainning data

        #if noreward_number>200000 and reward_number>200000:
        #    break
        if rewards_tags[-1] == '0_0_0':
            if level_noReward_num[int(power_tags[-1]) - 101] >= level_limit[int(power_tags[-1]) - 101]/2.0 or level_num[int(power_tags[-1]) - 101] >= level_limit[int(power_tags[-1]) - 101] :
                continue
            level_noReward_num[int(power_tags[-1]) - 101] += 1
            level_num[int(power_tags[-1]) - 101] += 1
            balance_file.write(line)
            balance_file.write('\n')

        else:
            if level_num[int(power_tags[-1]) - 101] >= level_limit[int(power_tags[-1]) - 101]:
                continue
            level_num[int(power_tags[-1]) - 101] += 1
            balance_file.write(line)
            balance_file.write('\n')


        #print rewards_tags[-1]
        for i in range(0, len(power_tags)):
            single_behavior = []
            single_behavior.append(power_tags[i])
            single_behavior.append(brand_tags[i])
            single_behavior.append(rewards_tags[i])

            if i == len(power_tags)-1:
                train_sample_y = single_behavior
                total_sum += 1
                if power_tags[i] == '107':
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
    print(level_num)
    print(float(level_1)/total_sum)
    return train_samples


train = balance_dataset()

print(len(train))
print(train[0][1])
print(train[1][1])

print(train[0][2])
print(train[1][2])

print(train[0][3])
print(train[1][3])


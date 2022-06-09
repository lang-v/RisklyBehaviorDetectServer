import os
from random import Random, shuffle

# 从train.list中按照比例随机抽取部分数据作为测试集
def split_list():
    train_file = open('train.list', mode='r')
    train_file_out = open('train0.list', mode='w')
    lines = list(train_file)

    test_file = open('test.list', mode='w')

    count = -1
    temp_list = list()
    for index in range(len(lines)):
        line = lines[index]
        l = line.strip('\n').split()
        if int(l[1]) != count or index == len(lines) - 1:
            count += 1
            shuffle(x=temp_list)
            index = int(len(temp_list) / 2)
            test_file.writelines(temp_list[:index])
            train_file_out.writelines(temp_list[index:])
            temp_list.clear()
        print(line)
        temp_list.append(line)

    train_file_out.flush()
    train_file_out.close()
    test_file.flush()
    test_file.close()


# 打乱list列表
def shuffle_list(path):
    input = open(path)
    lines = list(input)

    output = open(path, 'w')
    shuffle(x=lines)
    output.writelines(lines)
    output.flush()
    output.close()
    input.close()


# split_list()
# shuffle_list('train.list')
# shuffle_list('test.list')

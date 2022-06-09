# 打乱list列表
from random import Random, shuffle



def shuffle_list(path):
    input = open(path)
    lines = list(input)

    output = open(path, 'w')
    shuffle(x=lines)
    output.writelines(lines)
    output.flush()
    output.close()
    input.close()


shuffle_list('train.list')
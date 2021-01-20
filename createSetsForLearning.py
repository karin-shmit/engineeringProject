import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import itertools
import collections

#filtered data
lst1_carmen = np.load('filtered\\area_1_carmen_filtered_25359_signals.npy', allow_pickle=True)
lst2_carmen = np.load('filtered\\area_2_carmen_filtered_52617_signals.npy', allow_pickle=True)
lst3_carmen = np.load('filtered\\area_3_carmen_filtered_12431_signals.npy', allow_pickle=True)
lst4_carmen = np.load('filtered\\area_4_carmen_filtered_7168_signals.npy', allow_pickle=True)

lst1_penny = np.load('filtered\\area_1_penny_filtered_6331_signals.npy', allow_pickle=True)
lst2_penny = np.load('filtered\\area_2_penny_filtered_30472_signals.npy', allow_pickle=True)
lst3_penny = np.load('filtered\\area_3_penny_filtered_2298_signals.npy', allow_pickle=True)
lst4_penny = np.load('filtered\\area_4_penny_filtered_1780_signals.npy', allow_pickle=True)

lst1_menta = np.load('filtered\\area_1_menta_filtered_30408_signals.npy', allow_pickle=True)
lst2_menta = np.load('filtered\\area_2_menta_filtered_47582_signals.npy', allow_pickle=True)
lst3_menta = np.load('filtered\\area_3_menta_filtered_20016_signals.npy', allow_pickle=True)
lst4_menta = np.load('filtered\\area_4_menta_filtered_4109_signals.npy', allow_pickle=True)

carmen_full_table = pd.read_csv('carmen_full_info.csv')
penny_full_table = pd.read_csv('penny_full_info.csv')
menta_full_table = pd.read_csv('menta_full_info.csv')


def data_size(lst1, lst2, lst3, lst4):
    return min(len(lst1), len(lst2), len(lst3), len(lst4))


def shared_dates_level(lst_dates):
    dates_shared_1 = []
    dates_shared_2 = []
    dates_shared_3 = []
    dates_shared_4 = []
    dates_shared = []
    for column in lst_dates:
        y = lst_dates[column]
        y = y.notna()
        m = 0
        for i in y:
            if i is True:
                m += 1
        if len(column) == 5:
            column = str(0) + column
        if m == 1:
            dates_shared_1.append(column)
        elif m == 2:
            dates_shared_2.append(column)
        elif m == 3:
            dates_shared_3.append(column)
        elif m == 4:
            dates_shared_4.append(column)
    return [dates_shared_4, dates_shared_3, dates_shared_2, dates_shared_1]


def find_closese_sum(numbers, targets):
    bestcomb = []
    numbers = numbers[:]
    for t in targets:
        if not numbers:
            break
        combs = sum([list(itertools.combinations(numbers, r))
                     for r in range(1, len(numbers) + 1)], [])
        sums = np.asarray(list(map(sum, combs)))
        index = np.argmin(np.abs(np.asarray(sums) - t))
        if (sums[index] - t) < 0:
            if index < len(combs) - 1:
                index+=1
        bestcomb = combs[index]
        numbers = list(set(numbers).difference(bestcomb))
    return bestcomb


def create_dates_by_len(date_shared):
    def func(x):
        return len(x)

    dates_shared = list(date_shared)
    dates_by_len = map(func, dates_shared)
    dates_by_len = list(dates_by_len)
    dic = dict()
    for i in range(len(dates_by_len)):
        dic[dates_by_len[i]] = dates_shared[i]
    return dic, dates_by_len


def part_of_the_signal(closed, dic, dic_3, last, k, dates_by_len_3, signals):
    for key in closed:
        if key != last:
            signals += dic[key]
        else:
            break
    signals += dic[last][: k]
    temp = dic[last][k:]
    dic.pop(last)
    dates_by_len_3.append(np.abs(k))
    dic_3[np.abs(k)] = temp

    return dic, dic_3, signals, dates_by_len_3


def data_for_big_area(data_size, dates_shared):
    def func(x):
        return len(x)

    dic_0, d_0 = create_dates_by_len(dates_shared[0])
    dic_1, d_1 = create_dates_by_len(dates_shared[1])
    dic_2, d_2 = create_dates_by_len(dates_shared[2])
    dic_3, d_3 = create_dates_by_len(dates_shared[3])

    dates_by_len = [[], [], [], []]
    lst_temp = [d_0, d_1, d_2, d_3]
    for i in range(len(lst_temp)):
        if len(lst_temp[i]) > 10:
            num_iter = int(np.ceil(len(lst_temp[i]) / 10))
            for j in range(num_iter - 1):
                list_i = lst_temp[i][j * 10: (j + 1) * 10]
                dates_by_len[i].append(list_i)
        else:
            dates_by_len[i].append(lst_temp[i])

    dics = [dic_0, dic_1, dic_2, dic_3]

    training_val, validation_val, test_val = int(np.floor(data_size * 0.7)), int(np.floor(data_size * 0.15)), int(
        np.floor(data_size * 0.15))

    training_set, dates_by_len, dics = signals_for_area(training_val, dates_by_len, dics)
    validation_set, dates_by_len, dics = signals_for_area(validation_val, dates_by_len, dics)
    test_set, dates_by_len, dics = signals_for_area(test_val, dates_by_len, dics)
    return training_set, validation_set, test_set


def signals_for_area(k, dates_by_len, dics):
    res = 0
    signals, closed, last = [], list(), -1
    for i in range(len(dates_by_len)):
        for j in range(len(dates_by_len[i])):
            dates_by_len[i][j], k, closed, last = updated_para(dates_by_len[i][j], k)
            res += sum(closed)
            if k < 0:  # there is a need to take part of that list
                res -= last
                dics[i], dics[3], signals, dates_by_len[3][0] = part_of_the_signal(closed, dics[i], dics[3], last, k,
                                                                                   dates_by_len[3][0], signals)
                return signals, dates_by_len, dics
            for key in closed:
                signals += dics[i][key]
            dics[i] = {key: val for key, val in dics[i].items() if key not in closed}
            if k == 0:
                return signals, dates_by_len, dics
    return signals, dates_by_len, dics


def updated_para(dates_by_len, k):
    closed = list(find_closese_sum(dates_by_len, [k]))
    if not closed:
        last = -1
    else:
        last = closed[-1]
    res = last
    for i in closed:
        if closed:
            k -= i
            dates_by_len.remove(i)
            if k < 0:
                res = i
                break
    return dates_by_len, k, closed, res

def add_to_dict(date, x, dic):
    if x[3] in dic:
      dic[x[3]].append(x[0])
    else:
      dic[x[3]] = [x[0]]
    return dic


def extract_signals(dates, monkey):
  dic_1, dic_2, dic_3, dic_4 = {}, {}, {}, {}
  for x in tqdm(monkey):
    if x[3] in dates[0]:
      dic_1 = add_to_dict(dates[0], x, dic_1)
    elif x[3] in dates[1]:
      dic_2 = add_to_dict(dates[1], x, dic_2)
    elif x[3] in dates[2]:
      dic_3 = add_to_dict(dates[2], x, dic_3)
    elif x[3] in dates[3]:
      dic_4 = add_to_dict(dates[3], x, dic_4)

  values_1 = dic_1.values()
  values_2 = dic_2.values()
  values_3 = dic_3.values()
  values_4 = dic_4.values()
  return [values_1 , values_2, values_3, values_4]


def save_data(train, test, valid, monkey, area):

  np.save("train\\"+'area_'+str(area)+'_'+str(monkey)+'_train_'+ str(len(train)) +'_signals.npy', np.array(train))
  np.save("test\\"+'area_'+str(area)+'_'+str(monkey)+'_test_'+ str(len(test)) +'_signals.npy', np.array(test))
  np.save("validation\\"+'area_'+str(area)+'_'+str(monkey)+'_validation_'+ str(len(valid)) +'_signals.npy', np.array(valid))


# def run_data(lst1, lst2, lst3, lst4, full_table, monkey, OUTPUT_PATH):
#     y = data_size(lst1, lst2, lst3, lst4)
#     shared_data = shared_dates_level(full_table)
#
#     lst1_dates = extract_signals(shared_data, lst1)
#     lst2_dates = extract_signals(shared_data, lst2)
#     lst3_dates = extract_signals(shared_data, lst3)
#     lst4_dates = extract_signals(shared_data, lst4)
#
#     data_1 = data_ready(y, lst1_dates)
#     data_2 = data_ready(y, lst2_dates)
#     data_3 = data_ready(y, lst3_dates)
#     data_4 = data_ready(y, lst4_dates)
#
#     # res_1 = divide_data(data_1)
#     # res_2 = divide_data(data_2)
#     # res_3 = divide_data(data_3)
#     # res_4 = divide_data(data_4)
#
#     # save_data(monkey, res_1, 1, OUTPUT_PATH)
#     # save_data(monkey, res_2, 2, OUTPUT_PATH)
#     # save_data(monkey, res_3, 3, OUTPUT_PATH)
#     # save_data(monkey, res_4, 4, OUTPUT_PATH)

y= data_size(lst1_menta, lst2_menta, lst3_menta, lst4_menta)
shared_data = shared_dates_level(menta_full_table)
lst_dates = extract_signals(shared_data, lst1_menta)
train, test, valid = data_for_big_area(y, lst_dates)
# data_for_big_area(y, lst2_dates)

save_data(train, test, valid, "menta", 1)

print(np.shape(train))
print(np.shape(test))
print(np.shape(valid))




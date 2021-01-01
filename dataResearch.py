import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.signal as sp
import collections

SAMPLE_RATE = 1000
SAMPLE_DATE_CARMEN = '011213'
SAMPLE_DATE_PENNY_124 = '180717'
SAMPLE_DATE_PENNY_3 = '130717'
SAMPLE_DATE_MENTA_124 = '100416'
SAMPLE_DATE_MENTA_3 = '230316'
CARMEN = 'carmen'
MENTA = 'menta'
PENNY = 'penny'
THRESHOLD = 32764

def labelMonkeyDF(monkey_area1, monkey_area2, monkey_area3, monkey_area4):
    """
    :return: labeled monkeys data frame
    """
    df1 = pd.DataFrame({'signal': monkey_area1[:, 0], 'brain_area': monkey_area1[:, 1], 'depth': monkey_area1[:, 2],'date': monkey_area1[:, 3]})
    df2 = pd.DataFrame({'signal': monkey_area2[:, 0], 'brain_area': monkey_area2[:, 1], 'depth': monkey_area2[:, 2],'date': monkey_area2[:, 3]})
    df3 = pd.DataFrame({'signal': monkey_area3[:, 0], 'brain_area': monkey_area3[:, 1], 'depth': monkey_area3[:, 2],'date': monkey_area3[:, 3]})
    df4 = pd.DataFrame({'signal': monkey_area4[:, 0], 'brain_area': monkey_area4[:, 1], 'depth': monkey_area4[:, 2],'date': monkey_area4[:, 3]})
    return df1, df2, df3, df4

def getSamples(df, sampleDate):
    """
    :return: 30 first samples of a certain date
    """
    counter = 0
    samples = []
    for x, y in zip(df['signal'], df['date']):
        if counter >= 30:
            break
        if y == sampleDate:
            counter += 1
            samples.append(x)
    return samples

def plotMonkeySamples(monkey_signal ,monkey_name, area, sampleDate, color):
    plt.plot(monkey_signal[0], color=color, label="area "+str(area))
    i=0
    for x in monkey_signal:
        if i==0:
            i+=1
            continue
        plt.plot(x, color=color)
    plt.title(monkey_name+" area "+str(area)+" | date: " + sampleDate)
    plt.show()


def plotAllAreas(df1, df2, df3, df4, monkey, sampleDate):
    """
    plots all the monkeys
    """
    area1_samples = getSamples(df1, sampleDate)
    area2_samples = getSamples(df2, sampleDate)
    area3_samples = getSamples(df3, SAMPLE_DATE_MENTA_3)
    area4_samples = getSamples(df4, sampleDate)

    plotMonkeySamples(area1_samples,monkey,1, sampleDate,"red")
    plotMonkeySamples(area2_samples,monkey,2, sampleDate,"blue")
    plotMonkeySamples(area3_samples,monkey,3, SAMPLE_DATE_MENTA_3,"green")
    plotMonkeySamples(area4_samples,monkey,4, sampleDate,"orange")

    # plt.title("all areas "+monkey)
    # plt.legend()
    # plt.show()


def plotPSDMean(samples, area, monkey, color, color_mean):
    """
    plot the PSD mean and the PSD on the same graph for the samples
    """
    psd_lst = []
    f_axis = []
    psd, f = plt.psd(samples[0], NFFT=1024, Fs=SAMPLE_RATE, noverlap=896, color=color, label="area "+str(area))
    for x in f:
        if x<150:
            f_axis.append(x)
    psd_lst.append(psd[:len(f_axis)])
    i=0
    for x in tqdm(samples):
        if i==0:
            i+=1
            continue
        psd, f = plt.psd(x, NFFT=1024, Fs=SAMPLE_RATE, noverlap=896, color=color)
        psd_lst.append(psd[:len(f_axis)])

    plt.show()

    psd_lst = np.array(psd_lst)
    # for p in psd_lst:
    #     plt.plot(f_axis,10 * np.log10(p), color=color)

    psd_mean = psd_lst.mean(axis=0)
    psd_mean = 10 * np.log10(psd_mean)
    # plt.plot(f_axis, psd_mean, color=color_mean, label="Mean area "+str(area))
    # plt.title(monkey + " area " + str(area) + " | PSD mean and PSD")
    # plt.legend()
    # plt.show()
    return psd_lst, psd_mean, f_axis


def getMeanPSDplot(df1, df2, df3, df4, sampleDate, monkey):
    """
    plot the PSD mean and the PSD for all the areas
    """
    area1_samples = getSamples(df1, sampleDate)
    area2_samples = getSamples(df2, sampleDate)
    area3_samples = getSamples(df3, SAMPLE_DATE_MENTA_3)
    area4_samples = getSamples(df4, sampleDate)

    psd_lst1, psd_mean1, f_axis1 = plotPSDMean(area1_samples,1,monkey, "blue", "red")
    psd_lst2, psd_mean2, f_axis2 = plotPSDMean(area2_samples,2,monkey, "green", "orange")
    psd_lst3, psd_mean3, f_axis3 = plotPSDMean(area3_samples,3,monkey, "black", "yellow")
    psd_lst4, psd_mean4, f_axis4 = plotPSDMean(area4_samples,4,monkey, "grey", "purple")

    for p in psd_lst1:
        plt.plot(f_axis1,10 * np.log10(p), color="blue")
    plt.plot(f_axis1, psd_mean1, color="red", label="Mean area "+str(1))
    for p in psd_lst2:
        plt.plot(f_axis2,10 * np.log10(p), color="green")
    plt.plot(f_axis2, psd_mean2, color="orange", label="Mean area "+str(2))
    for p in psd_lst3:
        plt.plot(f_axis3,10 * np.log10(p), color="black")
    plt.plot(f_axis3, psd_mean3, color="yellow", label="Mean area "+str(3))
    for p in psd_lst4:
        plt.plot(f_axis4,10 * np.log10(p), color="grey")
    plt.plot(f_axis4, psd_mean4, color="purple", label="Mean area "+str(4))

    plt.title("PSD all areas "+monkey)
    plt.legend()
    plt.show()

def notch50(din, Fs):
    """
    notch50 algorithm
    """
    Wp = np.array([46, 51]) / (Fs/2)
    Ws = np.array([49, 50]) / (Fs/2)
    Rp = 0.5
    Rs = 10
    # n,Wn = sp.ellipord( Wp, Ws, Rp, Rs)
    # b,a=ellip(n,Rp,Rs,Wn,'stop')
    n,Wn = sp.cheb2ord( Wp, Ws, Rp, Rs)
    [b,a]= sp.cheby2(n,Rs,Wn,'stop')
    dout = sp.filtfilt(b,a,din)
    return dout

def activateNotch50(df):
    """
    activate the notch50 on the signals in the data frame
    """
    x = df.__deepcopy__()
    for index in x.index:
        x.loc[index,'signal'] = notch50(x.loc[index,'signal'], SAMPLE_RATE)
    return x

def countSaturation(df):
    """
    get the percentage of saturated bins in per signal for all the signals in the area
    """
    saturation = {}
    for x in tqdm(df["signal"]):
        counter = 0
        for i in x:
            if i>31000:
                counter+=1
        percent = (counter/8000) * 100
        if percent != 0:
            if percent not in saturation:
                saturation[percent]=1
            else:
                saturation[percent]+=1
    return saturation

def getSaturated(df):
    """
    find all the signals that have above 5% of saturated bins and the signals that have set of above 20 neighbors that
    are above the threshold
    """
    signals = []
    for x in tqdm(df["signal"]):
        counter = 0
        for i in x:
            if i >= THRESHOLD:
                counter += 1
        percent = (counter / 8000) * 100
        if percent >= 5:
            signals.append(x)
    for x in tqdm(df["signal"]):
        x = np.array(x, dtype=np.int64)
        set_neigboring = set()
        counter = 0
        neighbor = 0
        count_neigbor = 0
        for i in x:
            if np.abs(i) >= THRESHOLD:
                counter += 1
                if np.abs(i - neighbor) < 10:
                    count_neigbor += 1
                else:
                    set_neigboring.add(count_neigbor)
                    count_neigbor = 1
            neighbor = i
            if i == len(x) - 1:
                set_neigboring.add(count_neigbor)
        if len(set_neigboring) > 0:
            if max(set_neigboring) >= 20:
                signals.append(x)

    return signals

def deleteFromDF(df, lst):
    """
    deletes from df signals
    """
    print(lst)
    if len(lst) > 0:
        drop_indexes = df[df['signal'].isin(lst)].index
        new_df = df.drop(index=drop_indexes)
    else:
        new_df = df
    return new_df.to_numpy(), len(new_df["signal"])


lst1_penny = np.load('area_1_penny_divided_6349_signals.npy', allow_pickle=True)
lst2_penny = np.load('area_2_penny_divided_30496_signals.npy', allow_pickle=True)
lst3_penny = np.load('area_3_penny_divided_2301_signals.npy', allow_pickle=True)
lst4_penny = np.load('area_4_penny_divided_1780_signals.npy', allow_pickle=True)

lst1_carmen = np.load('area_1_carmen_divided_25542_signals.npy', allow_pickle=True)
lst2_carmen = np.load('area_2_carmen_divided_52697_signals.npy', allow_pickle=True)
lst3_carmen = np.load('area_3_carmen_divided_12440_signals.npy', allow_pickle=True)
lst4_carmen = np.load('area_4_carmen_divided_7170_signals.npy', allow_pickle=True)

lst1_menta = np.load('area_1_menta_divided_30450_signals.npy', allow_pickle=True)
lst2_menta = np.load('area_2_menta_divided_47614_signals.npy', allow_pickle=True)
lst3_menta = np.load('area_3_menta_divided_20026_signals.npy', allow_pickle=True)
lst4_menta = np.load('area_4_menta_divided_4109_signals.npy', allow_pickle=True)

df1_c, df2_c,df3_c,df4_c = labelMonkeyDF(lst1_carmen,lst2_carmen,lst3_carmen,lst4_carmen)
df1_p, df2_p,df3_p,df4_p = labelMonkeyDF(lst1_penny,lst2_penny,lst3_penny,lst4_penny)
df1_m, df2_m,df3_m,df4_m = labelMonkeyDF(lst1_menta,lst2_menta,lst3_menta,lst4_menta)


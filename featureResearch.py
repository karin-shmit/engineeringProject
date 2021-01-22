from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_classif
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import librosa
import sys
import entropy as en
from pyentrp import entropy as ent

SAMPLE_RATE = 1000
eps = sys.float_info.epsilon

def labelMonkeyDF(monkey_area1, monkey_area2, monkey_area3, monkey_area4):
    """
    :return: labeled monkeys data frame
    """
    df1 = pd.DataFrame({'signal': monkey_area1[:, 0], 'brain_area': monkey_area1[:, 1], 'depth': monkey_area1[:, 2],'date': monkey_area1[:, 3]})
    df2 = pd.DataFrame({'signal': monkey_area2[:, 0], 'brain_area': monkey_area2[:, 1], 'depth': monkey_area2[:, 2],'date': monkey_area2[:, 3]})
    df3 = pd.DataFrame({'signal': monkey_area3[:, 0], 'brain_area': monkey_area3[:, 1], 'depth': monkey_area3[:, 2],'date': monkey_area3[:, 3]})
    df4 = pd.DataFrame({'signal': monkey_area4[:, 0], 'brain_area': monkey_area4[:, 1], 'depth': monkey_area4[:, 2],'date': monkey_area4[:, 3]})
    return df1, df2, df3, df4

def extract_feature(X):
    X = X.astype(float)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=SAMPLE_RATE, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=SAMPLE_RATE).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=SAMPLE_RATE).T, axis=0)
    shannon = [ent.shannon_entropy(X)]
    sample = ent.sample_entropy(X, 1)
    spectral = [np.round(spectral_entropy(X, SAMPLE_RATE), 2)]
    per = [ent.permutation_entropy(X)]
    energy_ent = [energy_entropy(X)]
    energy_sig = [energy(X)]
    zero_cross = [zero_crossing_rate(X)]
    f, psd = welch(X, nfft=1024, fs=SAMPLE_RATE, noverlap=896, nperseg=1024)
    fft = np.fft.fft(X) / len(X)
    fft = np.abs(fft[: len(X) // 7])
    fft = [fft[0]]

    return np.concatenate((mfccs, chroma, mel, shannon, sample, spectral,per,energy_ent, energy_sig, zero_cross, psd, chroma, fft))

def energy_entropy(frame, n_short_blocks=10):
    """Computes entropy of energy"""
    # total frame energy
    frame_energy = np.sum(frame ** 2)
    frame_length = len(frame)
    sub_win_len = int(np.floor(frame_length / n_short_blocks))
    if frame_length != sub_win_len * n_short_blocks:
        frame = frame[0:sub_win_len * n_short_blocks]

    # sub_wins is of size [n_short_blocks x L]
    sub_wins = frame.reshape(sub_win_len, n_short_blocks, order='F').copy()

    # Compute normalized sub-frame energies:
    s = np.sum(sub_wins ** 2, axis=0) / (frame_energy + eps)

    # Compute entropy of the normalized sub-frame energies:
    entropy = -np.sum(s * np.log2(s + eps))
    return entropy

def spectral_entropy(signal, n_short_blocks=10):
    """Computes the spectral entropy"""
    # number of frame samples
    num_frames = len(signal)

    # total spectral energy
    total_energy = np.sum(signal ** 2)

    # length of sub-frame
    sub_win_len = int(np.floor(num_frames / n_short_blocks))
    if num_frames != sub_win_len * n_short_blocks:
        signal = signal[0:sub_win_len * n_short_blocks]

    # define sub-frames (using matrix reshape)
    sub_wins = signal.reshape(sub_win_len, n_short_blocks, order='F').copy()

    # compute spectral sub-energies
    s = np.sum(sub_wins ** 2, axis=0) / (total_energy + eps)

    # compute spectral entropy
    entropy = -np.sum(s * np.log2(s + eps))

    return entropy


def energy(frame):
    """Computes signal energy of frame"""
    return np.sum(frame ** 2) / np.float64(len(frame))

def zero_crossing_rate(frame):
    """Computes zero crossing rate of frame"""
    count = len(frame)
    count_zero = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
    return np.float64(count_zero) / np.float64(count - 1.0)


def dc_normalize(signal):
    """Removes DC and normalizes to -1, 1 range"""
    norm = signal.copy().astype(np.float64)
    norm -= norm.mean()
    norm /= abs(norm).max() + 1e-10
    return norm

def getAllFeaturesVector(area):
    """
    length of features:
    mfccs : 40 [0:39]
    chroma : 12 [40:51]
    mel : 128 [52:179]
    shannon : 1 [180:180]
    sample : 1 [181:181]
    per : 1 [182:182]
    fft : 1142 [183:1324]
    """
    features = []
    for x in tqdm(area):
        mfccs, chroma, mel, shannon, sample, per, fft = extract_feature(x)
        feature = np.concatenate((mfccs,chroma,mel,shannon, sample, per, fft))
        features.append(feature)
    return features


def getFeaturesFromArea(area):
    feature_mfcc = []
    feature_chroma = []
    feature_mel = []
    feature_shanon = []
    feature_sample = []
    feature_per = []
    feature_fft = []
    for x in tqdm(area):
        mfccs, chroma, mel, shanon, sample, per, fft = extract_feature(x)
        feature_mfcc.append(mfccs)
        feature_chroma.append(chroma)
        feature_mel.append(mel)
        feature_shanon.append(shanon)
        feature_sample.append(sample)
        feature_per.append(per)
        feature_fft.append(fft)
    return feature_mfcc, feature_chroma, feature_mel, feature_shanon, feature_sample, feature_per

def getTSNE2D(lst):
    tsne = TSNE(n_components=2)
    comp = tsne.fit_transform(lst)
    x = [i[0] for i in comp]
    y = [i[1] for i in comp]
    return x, y

def getTSNE3D(lst):
    tsne = TSNE(n_components=3)
    comp = tsne.fit_transform(lst)
    x = [i[0] for i in comp]
    y = [i[1] for i in comp]
    z = [i[0] for i in comp]
    return x, y, z

def getPCA2D(lst):
    lst = StandardScaler().fit_transform(lst)
    pca = PCA(n_components=2)
    comp =  pca.fit_transform(lst)
    x = [i[0] for i in comp]
    y = [i[1] for i in comp]
    return x,y

def getPCA3D(lst):
    # lst = StandardScaler().fit_transform(lst)
    pca_mfcc = PCA(n_components=3)
    comp = pca_mfcc.fit_transform(lst)
    x = [i[0] for i in comp]
    y = [i[1] for i in comp]
    z = [i[2] for i in comp]
    return x, y, z

def plotFeatureMonkey2D(area1, area2, area3, area4, monkey, feature):
    x_1, y_1 = getPCA2D(area1)
    plt.scatter(x_1, y_1, color='maroon', label="area_1")
    x_2, y_2 = getPCA2D(area2)
    plt.scatter(x_2, y_2, color='navy', label="area_2")
    x_3, y_3 = getPCA2D(area3)
    plt.scatter(x_3, y_3, color='green', label="area_3")
    x_4, y_4 = getPCA2D(area4)
    plt.scatter(x_4, y_4, color='purple', label="area_4")
    plt.title(feature+ ' - ' + monkey)
    plt.legend()
    plt.show()

def plotAllFeaturesMonkeyArea2D(mfcc, chroma, mel,shanon, sample, per, area, color):
    x_1, y_1 = getPCA2D(mfcc)
    plt.scatter(x_1, y_1, color=color, label="mfcc_area_"+str(area), marker=".")
    x_2, y_2 = getPCA2D(chroma)
    plt.scatter(x_2, y_2, color=color, label="chroma_area_"+str(area), marker="^")
    x_3, y_3 = getPCA2D(mel)
    plt.scatter(x_3, y_3, color=color, label="mel_area_"+str(area), marker="x")

def plotAllVectorFeatures2Dpca(vector, area, color):
    x_1, y_1 = getPCA2D(vector)
    plt.scatter(x_1, y_1, color=color, label=str(area), marker=".")

def plotAllVectorFeatures2Dtsne(vector, area, color):
    x_1, y_1 = getTSNE2D(vector)
    plt.scatter(x_1, y_1, color=color, label=str(area), marker=".")

def plotAllVectorFeatures3Dpca(vector1, vector2, vector3, vector4, monkey):
    x_1, y_1, z_1 = getPCA3D(vector1)
    x_2, y_2, z_2 = getPCA3D(vector2)
    x_3, y_3, z_3 = getPCA3D(vector3)
    x_4, y_4, z_4 = getPCA3D(vector4)
    ax = plt.axes(projection="3d")
    ax.scatter3D(x_1, y_1, z_1 , color="maroon", label="area_" + str(1))
    ax.scatter3D(x_2, y_2, z_2, color="navy", label="area_" + str(2))
    ax.scatter3D(x_3, y_3, z_3, color="green", label="area_" + str(3))
    ax.scatter3D(x_4, y_4, z_4, color="purple", label="area_" + str(4))
    plt.title('PCA for all features together | ' + monkey + "  ")
    plt.legend()
    plt.show()

def plotAllVectorFeatures3Dtsne(vector1, vector2, vector3, vector4, monkey):
    x_1, y_1, z_1 = getTSNE3D(vector1)
    x_2, y_2, z_2 = getTSNE3D(vector2)
    x_3, y_3, z_3 = getTSNE3D(vector3)
    x_4, y_4, z_4 = getTSNE3D(vector4)
    ax = plt.axes(projection="3d")
    ax.scatter3D(x_1, y_1, z_1 , color="maroon", label="area_" + str(1))
    ax.scatter3D(x_2, y_2, z_2, color="navy", label="area_" + str(2))
    ax.scatter3D(x_3, y_3, z_3, color="green", label="area_" + str(3))
    ax.scatter3D(x_4, y_4, z_4, color="purple", label="area_" + str(4))
    plt.title('PCA for all features together | ' + monkey + "  ")
    plt.legend()
    plt.show()

def plotPCAfeature(area1, area2, area3, area4, monkey):
    area1_mfcc, area1_chroma, area1_mel, area1_shanon, area1_sample, area1_per = getFeaturesFromArea(area1)
    area2_mfcc, area2_chroma, area2_mel, area2_shanon, area2_sample, area2_per = getFeaturesFromArea(area2)
    area3_mfcc, area3_chroma, area3_mel, area3_shanon, area3_sample, area3_per = getFeaturesFromArea(area3)
    area4_mfcc, area4_chroma, area4_mel, area4_shanon, area4_sample, area4_per = getFeaturesFromArea(area4)

    plotFeatureMonkey2D(area1_mfcc, area2_mfcc, area3_mfcc, area4_mfcc, monkey, "mfcc")
    plotFeatureMonkey2D(area1_chroma, area2_chroma, area3_chroma, area4_chroma, monkey, "chroma")
    plotFeatureMonkey2D(area1_mel, area2_mel, area3_mel, area4_mel, monkey, "mel")
    plotFeatureMonkey2D(area1_shanon, area2_shanon, area3_shanon, area4_shanon, monkey, "shanon entropy")
    plotFeatureMonkey2D(area1_sample, area2_sample, area3_sample, area4_sample, monkey, "sample entropy")
    plotFeatureMonkey2D(area1_per, area2_per, area3_per, area4_per, monkey, "permutation entropy")

    # plotAllFeaturesMonkeyArea2D(area1_mfcc, area1_chroma, area1_mel, area1_shanon, area1_sample, area1_per, 1, "maroon")
    # plotAllFeaturesMonkeyArea2D(area2_mfcc, area2_chroma, area2_mel, area2_shanon, area2_sample, area2_per, 2, "navy")
    # plotAllFeaturesMonkeyArea2D(area3_mfcc, area3_chroma, area3_mel, area3_shanon, area3_sample, area3_per, 3, "green")
    # plotAllFeaturesMonkeyArea2D(area4_mfcc, area4_chroma, area4_mel, area4_shanon, area4_sample, area4_per, 4, 'purple')

    plt.title('all features | ' + monkey + " | PCA normalized")
    plt.legend()
    plt.show()

def plotPCAallFeaturesVector(area1, area2, area3, area4, monkey):
    features1 = getAllFeaturesVector(area1)
    features2 = getAllFeaturesVector(area2)
    features3 = getAllFeaturesVector(area3)
    features4 = getAllFeaturesVector(area4)

    plotAllVectorFeatures3Dpca(features1, features2, features3, features4, monkey)
    # plotAllVectorFeatures2Dpca(features1, 1, "maroon")
    # plotAllVectorFeatures2Dpca(features2, 2, "navy")
    # plotAllVectorFeatures2Dpca(features3, 3, "green")
    # plotAllVectorFeatures2Dpca(features4, 4, "purple")

    # plt.title('PCA for all features together | ' + monkey + " | PCA normalized ")
    # plt.legend()
    # plt.show()

def plotTSNEallFeaturesVector(area1, area2, area3, area4, monkey):
    features1 = getAllFeaturesVector(area1)
    features2 = getAllFeaturesVector(area2)
    features3 = getAllFeaturesVector(area3)
    features4 = getAllFeaturesVector(area4)

    # plotAllVectorFeatures3Dtsne(features1, features2, features3, features4, monkey)
    plotAllVectorFeatures2Dtsne(features1, 1, "maroon")
    plotAllVectorFeatures2Dtsne(features2, 2, "navy")
    plotAllVectorFeatures2Dtsne(features3, 3, "green")
    plotAllVectorFeatures2Dtsne(features4, 4, "purple")

    plt.title('TSNE for all features together | ' + monkey + " ")
    plt.legend()
    plt.show()

def getMutualInfo(area1, area2, area3, area4, monkey):
    feature1 = getAllFeaturesVector(area1)
    feature2 = getAllFeaturesVector(area2)
    feature3 = getAllFeaturesVector(area3)
    feature4 = getAllFeaturesVector(area4)
    feature_matrix = np.concatenate((feature1, feature2, feature3, feature4))

    area1_y = [0] * len(area1)
    area2_y = [1] * len(area2)
    area3_y = [2] * len(area3)
    area4_y = [3] * len(area4)
    areas = area1_y + area2_y + area3_y + area4_y

    mi = mutual_info_classif(feature_matrix, areas, discrete_features=False)
    index1 = np.argmax(mi)
    old = mi[index1]
    mi[index1] = -1
    index2 = np.argmax(mi)

    mi[index1] = old

    plt.scatter(range(len(mi)), mi)
    plt.xlabel("feature index")
    plt.ylabel("mutual info")
    plt.title(monkey)
    plt.show()

    print(index1)
    print(index2)

def getMonkeyBestFeaturesPlot(area1,area2,area3,area4, first, second,xlab, ylab, monkey):
    feature1 = np.array(getAllFeaturesVector(area1))
    best1 = np.log10(feature1[:, first])
    secBest1 = feature1[:, second]
    feature2 = np.array(getAllFeaturesVector(area2))
    best2 = np.log10(feature2[:, first])
    secBest2 = feature2[:, second]
    feature3 = np.array(getAllFeaturesVector(area3))
    best3 = np.log10(feature3[:, first])
    secBest3 = feature3[:, second]
    feature4 = np.array(getAllFeaturesVector(area4))
    best4 = np.log10(feature4[:, first])
    secBest4 = feature4[:, second]
    plt.scatter(best1, secBest1, color="maroon", label="area1")
    plt.scatter(best2, secBest2, color='navy', label="area2")
    plt.scatter(best3, secBest3, color='green', label="area3")
    plt.scatter(best4, secBest4, color='purple', label="area4")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(monkey)
    plt.legend()
    plt.show()



lst1_penny = np.load('random\\area_1_penny_random_samples.npy', allow_pickle=True)
lst2_penny = np.load('random\\area_2_penny_random_samples.npy', allow_pickle=True)
lst3_penny = np.load('random\\area_3_penny_random_samples.npy', allow_pickle=True)
lst4_penny = np.load('random\\area_4_penny_random_samples.npy', allow_pickle=True)

# lst1_penny = np.load('filtered\\area_1_penny_filtered_6331_signals.npy', allow_pickle=True)
# lst2_penny = np.load('filtered\\area_2_penny_filtered_30472_signals.npy', allow_pickle=True)
# lst3_penny = np.load('filtered\\area_3_penny_filtered_2298_signals.npy', allow_pickle=True)
# lst4_penny = np.load('filtered\\area_4_penny_filtered_1780_signals.npy', allow_pickle=True)

lst1_carmen = np.load('random\\area_1_carmen_random_samples.npy', allow_pickle=True)
lst2_carmen = np.load('random\\area_2_carmen_random_samples.npy', allow_pickle=True)
lst3_carmen = np.load('random\\area_3_carmen_random_samples.npy', allow_pickle=True)
lst4_carmen = np.load('random\\area_4_carmen_random_samples.npy', allow_pickle=True)

# lst1_carmen = np.load('filtered\\area_1_carmen_filtered_25359_signals.npy',allow_pickle=True)
# lst2_carmen = np.load('filtered\\area_2_carmen_filtered_52617_signals.npy',allow_pickle=True)
# lst3_carmen = np.load('filtered\\area_3_carmen_filtered_12431_signals.npy',allow_pickle=True)
# lst4_carmen = np.load('filtered\\area_4_carmen_filtered_7168_signals.npy',allow_pickle=True)

lst1_menta = np.load('random\\area_1_menta_random_samples.npy', allow_pickle=True)
lst2_menta = np.load('random\\area_2_menta_random_samples.npy', allow_pickle=True)
lst3_menta = np.load('random\\area_3_menta_random_samples.npy', allow_pickle=True)
lst4_menta = np.load('random\\area_4_menta_random_samples.npy', allow_pickle=True)

# lst1_menta = np.load('filtered\\area_1_menta_filtered_30408_signals.npy', allow_pickle=True)
# lst2_menta = np.load('filtered\\area_2_menta_filtered_47582_signals.npy', allow_pickle=True)
# lst3_menta = np.load('filtered\\area_3_menta_filtered_20016_signals.npy', allow_pickle=True)
# lst4_menta = np.load('filtered\\area_4_menta_filtered_4109_signals.npy', allow_pickle=True)

# df1_m, df2_m,df3_m,df4_m = labelMonkeyDF(lst1_menta,lst2_menta,lst3_menta,lst4_menta)
# df1_p, df2_p,df3_p,df4_p = labelMonkeyDF(lst1_penny,lst2_penny,lst3_penny,lst4_penny)
# df1_c, df2_c,df3_c,df4_c = labelMonkeyDF(lst1_carmen,lst2_carmen,lst3_carmen,lst4_carmen)


# plotPCAfeature(lst1_penny, lst2_penny, lst3_penny, lst4_penny,'penny')
# plotPCAfeature(lst1_carmen, lst2_carmen, lst3_carmen, lst4_carmen,'carmen')
# plotPCAfeature(lst1_menta, lst2_menta, lst3_menta, lst4_menta,'menta')

# | PCA normalized

# plotPCAallFeaturesVector(lst1_carmen, lst2_carmen, lst3_carmen, lst4_carmen, "carmen")
# plotPCAallFeaturesVector(lst1_penny, lst2_penny, lst3_penny, lst4_penny, "penny")
# plotPCAallFeaturesVector(lst1_menta, lst2_menta, lst3_menta, lst4_menta, "menta")
#
# plotTSNEallFeaturesVector(lst1_carmen, lst2_carmen, lst3_carmen, lst4_carmen, "carmen")
# plotTSNEallFeaturesVector(lst1_menta, lst2_menta, lst3_menta, lst4_menta, "menta")
# plotTSNEallFeaturesVector(lst1_penny, lst2_penny, lst3_penny, lst4_penny, "penny")


# getMutualInfo(df1_c["signal"], df2_c["signal"], df3_c["signal"], df4_c["signal"], "carmen") # 183 55
# getMutualInfo(df1_m["signal"], df2_m["signal"], df3_m["signal"], df4_m["signal"], "menta") # 183, 57
# getMutualInfo(df1_p["signal"], df2_p["signal"], df3_p["signal"], df4_p["signal"], "penny") # 64 183

# getMonkeyBestFeaturesPlot(lst1_carmen, lst2_carmen, lst3_carmen, lst4_carmen, 183, 55 ,"1st feature of fft", "55th feature of mel", "carmen")
# getMonkeyBestFeaturesPlot(lst1_menta, lst2_menta, lst3_menta, lst4_menta, 183, 57, "1st feature of fft" , "57th feature of mel","menta")
# getMonkeyBestFeaturesPlot(lst1_penny, lst2_penny, lst3_penny, lst4_penny, 64, 183, "13 feature of mel", "1st feature of fft", "penny")






import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import librosa
from pyentrp import entropy as ent
from sklearn import svm
from sklearn import metrics
import carmen_files as f
from scipy.signal import welch
import matplotlib.pyplot as plt
import featureResearch as feat
import pandas as pd
import neurokit2 as nk

SAMPLE_RATE = 1000

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, \
    f1_score


def extract_mel_chroma_first_fft_feature(X):
    X = X.astype(float)
    stft = np.abs(librosa.stft(X))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=SAMPLE_RATE).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=SAMPLE_RATE).T, axis=0)
    fft = np.fft.fft(X) / len(X)
    fft = np.abs(fft[: len(X) // 7])
    fft = [fft[0]]
    welch = nk.signal_psd(X, method="welch", min_frequency=1, max_frequency=20, show=True)["Power"]
    multitaper = nk.signal_psd(X, method="multitapers", max_frequency=20, show=True)["Power"]
    lomb = nk.signal_psd(X, method="lomb", min_frequency=1, max_frequency=20, show=True)["Power"]
    # burg = nk.signal_psd(X, method="burg", min_frequency=1, max_frequency=20, order=10, show=True)["Power"]
    welch = np.array(welch)
    multitaper = np.array(multitaper)
    # lomb = np.array(lomb)
    # burg = np.array(burg)
    # print("chroma:", len(chroma))
    # print("mel:", len(mel))
    # print("fft:", len(fft))
    return chroma, mel, fft, welch, multitaper
    # return chroma, mel, fft, welch, multitaper, lomb, burg


def extract_additional_feature(X):
    X = X.astype(float)
    # burg = nk.signal_psd(X, method="burg", min_frequency=1, max_frequency=20, order=10, show=True)["Power"]
    sample = nk.entropy_sample(X)
    entropy = nk.entropy_approximate(X)
    # burg = np.array(burg)
    y = np.array([sample, entropy])
    return y


def extract_feature(X):
    X = X.astype(float)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=SAMPLE_RATE, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=SAMPLE_RATE).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=SAMPLE_RATE).T, axis=0)
    shannon = [ent.shannon_entropy(X)]
    sample = ent.sample_entropy(X, 1)
    per = [ent.permutation_entropy(X)]
    fft = np.fft.fft(X) / len(X)
    fft = np.abs(fft[: len(X) // 7])
    fft = [fft[0]]
    return mfccs, chroma, mel, shannon, sample, per, fft


def get_mel_chroma_first_fft_features_vector(area):
    """
    this method takes only mel, chroma and the first fft features
    """
    features = []
    for x in tqdm(area):
        chroma, mel, fft, welch, multitaper= extract_mel_chroma_first_fft_feature(x)
        feature = np.concatenate((chroma, mel, fft, welch, multitaper))
        features.append(feature)
    return np.array(features)


def get_additional_features(area):
    features = []
    for x in tqdm(area):
        y = extract_additional_feature(x)
        features.append(y)
    return np.array(features)


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
        feature = np.concatenate((mfccs, chroma, mel, shannon, sample, per, fft))
        features.append(feature)
    return np.array(features)


def save_data(res1, res2, res3, res4, path, monkey):
    np.save(
        path + 'area_' + str(1) + '_' + str(monkey) + '_features_extra_validation_' + str(len(res1)) + '_signals.npy',
        np.array(res1))
    np.save(
        path + 'area_' + str(2) + '_' + str(monkey) + '_features_extra_validation_' + str(len(res2)) + '_signals.npy',
        np.array(res2))
    np.save(
        path + 'area_' + str(3) + '_' + str(monkey) + '_features_extra_validation_' + str(len(res3)) + '_signals.npy',
        np.array(res3))
    np.save(
        path + 'area_' + str(4) + '_' + str(monkey) + '_features_extra_validation_' + str(len(res4)) + '_signals.npy',
        np.array(res4))


def getY(area1, area2, area3, area4):
    y_1 = [0] * len(area1)
    y_2 = [1] * len(area2)
    y_3 = [2] * len(area3)
    y_4 = [3] * len(area4)
    y = y_1 + y_2 + y_3 + y_4
    return y


def getX(area1, area2, area3, area4):
    x = np.concatenate((area1, area2, area3, area4))
    return x


def SVM(x_train, y_train, x_test, y_test):
    clf = svm.SVC(C=0.1, kernel='linear')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("SVM Accuracy:", metrics.accuracy_score(y_test, y_pred))


def SVMovr(x_train, y_train, x_test, y_test):
    clf = svm.SVC(decision_function_shape='ovr')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("SVC ONE VS REST Accuracy:", metrics.accuracy_score(y_test, y_pred))


def linearSVC(x_train, y_train, x_test, y_test):
    clf = svm.LinearSVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Linear SVC Accuracy:", metrics.accuracy_score(y_test, y_pred))


def randomForest(x_train, y_train, x_test, y_test):
    """Random Forest Accuracy: 0.6495348837209303"""
    # clf = RandomForestClassifier(max_depth=13)
    clf = RandomForestClassifier(
        min_samples_leaf=50,
        n_estimators=150,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=50,
        max_features='auto')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(y_pred)
    print("Random Forest Accuracy:", metrics.accuracy_score(y_test, y_pred))


def GradientBoosting(x_train, y_train, x_test, y_test):
    clf = GradientBoostingClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Gradient Boosting Accuracy:", metrics.accuracy_score(y_test, y_pred))


def KNeighbors(x_train, y_train, x_test, y_test):
    clf = KNeighborsClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("KNeighbors Accuracy:", metrics.accuracy_score(y_test, y_pred))


def mutualInfo(x, y, monkey):
    mi = mutual_info_classif(x, y, discrete_features=False)
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


def addFeatures(old, new):
    res = []
    for i in tqdm(range(len(old))):
        res.append(np.concatenate((old[i], new[i])))
    return np.array(res)


# x_train = getX(f.train1_carmen, f.train2_carmen, f.train3_carmen, f.train4_carmen)
# y_train = getY(f.train1_carmen, f.train2_carmen, f.train3_carmen, f.train4_carmen)
#
# x_test = getX(f.test1_carmen, f.test2_carmen, f.test3_carmen, f.test4_carmen)
# y_test = getY(f.test1_carmen, f.test2_carmen, f.test3_carmen, f.test4_carmen)

# SVM(x_train,y_train, x_test,y_test)
# randomForest(x_train,y_train, x_test,y_test)
# GradientBoosting(x_train,y_train, x_test,y_test)
# SVMovr(x_train,y_train, x_test,y_test)
# linearSVC(x_train,y_train, x_test,y_test)
# KNeighbors(x_train, y_train, x_test, y_test)

# randomForest(x_train,y_train, x_test,y_test)
# mutualInfo(x_test,y_test,"carmen")
# mutualInfo(x_train,y_train,"carmen")

def getFeaturesAndNormalize(data):
    """
    Doesn't help. The results are not good with this one
    """
    features = []
    for sig in tqdm(data):
        feature = feat.extract_feature(feat.dc_normalize(sig))
        features.append(feature)
    return np.array(features)


def resize_features_vector(data1, data2):
    resize_data = np.concatenate((data1, data2), axis=1)
    np.save(
        "train_short\\carmen\\" + 'area_' + str(1) + '_' + "carmen" + '_features_extra_validation_' + str(
            len(resize_data)) + '_signals.npy',
        np.array(resize_data))



#

# #
# x_train1 = get_additional_features(f.sig_train_carmen1)
# np.save(
#     "train_short\\carmen\\entropy\\" + 'area_' + str(1) + '_' + "carmen"+ '_features_extra_validation_' + str(len(x_train1)) + '_signals.npy',
#     np.array(x_train1))
# x_train2 = get_additional_features(f.sig_train_carmen2)
# np.save(
#     "train_short\\carmen\\entropy\\" + 'area_' + str(2) + '_' + "carmen"+ '_features_extra_validation_' + str(len(x_train2)) + '_signals.npy',
#     np.array(x_train2))
# x_train3 = get_additional_features(f.sig_train_carmen3)
# np.save(
#     "train_short\\carmen\\entropy\\" + 'area_' + str(3) + '_' + "carmen"+ '_features_extra_validation_' + str(len(x_train3)) + '_signals.npy',
#     np.array(x_train3))
# x_train4 = get_additional_features(f.sig_train_carmen4)
# np.save(
#     "train_short\\carmen\\entropy\\" + 'area_' + str(4) + '_' + "carmen"+ '_features_extra_validation_' + str(len(x_train4)) + '_signals.npy',
#     np.array(x_train4))
# save_data(x_train1, x_train2, x_train3, x_train4, "train_short\\carmen\\", "carmen")
x_test1 = get_additional_features(f.sig_test_carmen1)
x_test2 = get_additional_features(f.sig_test_carmen2)
x_test3 = get_additional_features(f.sig_test_carmen3)
x_test4 = get_additional_features(f.sig_test_carmen4)
save_data(x_test1, x_test2, x_test3, x_test4, "test_short\\carmen\\entropy\\", "carmen")
#
# #
# if __name__ == "__main__":
# #     signal = f.sig_valid_carmen_1[0]
# #     # parameters = nk.complexity_optimize(signal, show=True)
# #     # print(parameters)
# #     ci_min, ci_max = nk.hdi(signal, ci=0.95, show=True)
# #     # print(ci_min)
# #     # print(ci_max)
# #     print(np.array([ci_max, ci_min]))
#     train_1 = np.concatenate((f.short_train1_carmen, f.add_short_train1_carmen), axis=1)
#     train_2 = np.concatenate((f.short_train2_carmen, f.add_short_train2_carmen), axis=1)
#     train_3 = np.concatenate((f.short_train3_carmen, f.add_short_train3_carmen), axis=1)
#     train_4 = np.concatenate((f.short_train4_carmen, f.add_short_train4_carmen), axis=1)
#
#     test_1 = np.concatenate((f.short_test1_carmen, f.add_short_test1_carmen), axis=1)
#     test_2 = np.concatenate((f.short_test2_carmen, f.add_short_test2_carmen), axis=1)
#     test_3 = np.concatenate((f.short_test3_carmen, f.add_short_test3_carmen), axis=1)
#     test_4 = np.concatenate((f.short_test4_carmen, f.add_short_test4_carmen), axis=1)
#
#     # x_train = getX(f.short_train1_carmen, f.short_train2_carmen, f.short_train3_carmen, f.short_train4_carmen)
#     # y_train = getY(f.short_train1_carmen, f.short_train2_carmen, f.short_train3_carmen, f.short_train4_carmen)
#     x_train = getX(train_1, train_2, train_3, train_4)
#     y_train = getY(train_1, train_2, train_3, train_4)
#
#     x_test = getX(test_1, test_2, test_3, test_4)
#     y_test = getY(test_1, test_2, test_3, test_4)
# # #     # #     #
# #     x_test = getX(f.short_test1_carmen, f.short_test2_carmen, f.short_test3_carmen, f.short_test4_carmen)
# #     y_test = getY(f.short_test1_carmen, f.short_test2_carmen, f.short_test3_carmen, f.short_test4_carmen)
# #     # # #
# #     # x_train = getX(f.train1_carmen, f.train2_carmen, f.train3_carmen, f.train4_carmen)
# #     # y_train = getY(f.train1_carmen, f.train2_carmen, f.train3_carmen, f.train4_carmen)
# #     #
# #     # x_test = getX(f.test1_carmen, f.test2_carmen, f.test3_carmen, f.test4_carmen)
# #     # y_test = getY(f.test1_carmen, f.test2_carmen, f.test3_carmen, f.test4_carmen)
#
#     # SVM(x_train,y_train, x_test,y_test)
#     randomForest(x_train,y_train, x_test,y_test)
#     # GradientBoosting(x_train,y_train, x_test,y_test)
#     # SVMovr(x_train,y_train, x_test,y_test)
#     # linearSVC(x_train,y_train, x_test,y_test)
#     # KNeighbors(x_train, y_train, x_test, y_test)
#     mutualInfo(x_test, y_test, "carmen - test")
#     mutualInfo(x_train, y_train, "carmen - train")
#
# # #     print(f.short_test1_carmen)
# # #     print("--------------")
# # #     print(f.short_test2_carmen)

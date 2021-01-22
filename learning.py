import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import librosa
from pyentrp import entropy as ent
from sklearn import svm
from sklearn import metrics
import files as f
from scipy.signal import welch
import matplotlib.pyplot as plt
import featureResearch as feat
import pandas as pd

SAMPLE_RATE = 1000





# def getAllFeaturesVector(area):
#     """
#     length of features:
#     mfccs : 40 [0:39]
#     chroma : 12 [40:51]
#     mel : 128 [52:179]
#     shannon : 1 [180:180]
#     sample : 1 [181:181]
#     per : 1 [182:182]
#     fft : 1142 [183:1324]
#     """
#     features = []
#     for x in tqdm(area):
#         mfccs, chroma, mel, shannon, sample, per, fft = extract_feature(x)
#         feature = np.concatenate((mfccs,chroma,mel,shannon, sample, per, fft))
#         features.append(feature)
#     return np.array(features)

def save_data(res1,res2,res3,res4,path,monkey):
    np.save(path + 'area_' + str(1) + '_' + str(monkey) + '_features_extra_validation_' + str(len(res1)) + '_signals.npy',
            np.array(res1))
    np.save(path + 'area_' + str(2) + '_' + str(monkey) + '_features_extra_validation_' + str(len(res2)) + '_signals.npy',
            np.array(res1))
    np.save(path + 'area_' + str(3) + '_' + str(monkey) + '_features_extra_validation_' + str(len(res3)) + '_signals.npy',
            np.array(res1))
    np.save(path + 'area_' + str(4) + '_' + str(monkey) + '_features_extra_validation_' + str(len(res4)) + '_signals.npy',
            np.array(res1))

def getY(area1,area2,area3,area4):
    y_1 = [0] * len(area1)
    y_2 = [1] * len(area2)
    y_3 = [2] * len(area3)
    y_4 = [3] * len(area4)
    y = y_1 + y_2 + y_3 + y_4
    return y

def getX(area1,area2,area3,area4):
    x = np.concatenate((area1,area2,area3,area4))
    return x

def SVM(x_train, y_train, x_test, y_test):
    clf = svm.SVC(C=0.1,kernel='linear')
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
    """Random Forest Accuracy: 0.6265116279069768"""
    clf = RandomForestClassifier(max_depth=13)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    print(y_pred)
    print("Random Forest Accuracy:", metrics.accuracy_score(y_test, y_pred))

def GradientBoosting(x_train, y_train, x_test, y_test):
    clf = GradientBoostingClassifier()
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    print("Gradient Boosting Accuracy:", metrics.accuracy_score(y_test,y_pred))

def KNeighbors(x_train, y_train, x_test, y_test):
    clf = KNeighborsClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("KNeighbors Accuracy:", metrics.accuracy_score(y_test, y_pred))


def mutualInfo(x,y,monkey):
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
        res.append(np.concatenate((old[i],new[i])))
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
    features = []
    for sig in tqdm(data):
        feature = feat.extract_feature(feat.dc_normalize(sig))
        features.append(feature)
    return np.array(features)

x_train1 = getFeaturesAndNormalize(f.sig_train_carmen1)
x_train2 = getFeaturesAndNormalize(f.sig_train_carmen2)
x_train3 = getFeaturesAndNormalize(f.sig_train_carmen3)
x_train4 = getFeaturesAndNormalize(f.sig_train_carmen4)

save_data(x_train1,x_train2,x_train3,x_train4, "train\\", "carmen")

x_test1 = getFeaturesAndNormalize(f.sig_test_carmen1)
x_test2 = getFeaturesAndNormalize(f.sig_test_carmen2)
x_test3 = getFeaturesAndNormalize(f.sig_test_carmen3)
x_test4 = getFeaturesAndNormalize(f.sig_test_carmen4)

save_data(x_test1,x_test2,x_test3 ,x_test4, "test\\", "carmen")

x_train = getX(f.train1_carmen, f.train2_carmen, f.train3_carmen, f.train4_carmen)
y_train = getY(f.train1_carmen, f.train2_carmen, f.train3_carmen, f.train4_carmen)

x_test = getX(f.test1_carmen, f.test2_carmen, f.test3_carmen, f.test4_carmen)
y_test = getY(f.test1_carmen, f.test2_carmen, f.test3_carmen, f.test4_carmen)

randomForest(x_train,y_train, x_test,y_test)
mutualInfo(x_test,y_test,"carmen - test")
mutualInfo(x_train,y_train,"carmen - train")





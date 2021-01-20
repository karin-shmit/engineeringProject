import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import librosa
from pyentrp import entropy as ent
from sklearn import svm
from sklearn import metrics
import pandas as pd

SAMPLE_RATE =    1000

# 433269002

611
7207


train1_carmen = np.load('train\\carmen\\area_1_carmen_1FFT_train_features_5017_signals.npy', allow_pickle=True)
train2_carmen = np.load('train\\carmen\\area_2_carmen_1FFT_train_features_5017_signals.npy', allow_pickle=True)
train3_carmen = np.load('train\\carmen\\area_3_carmen_1FFT_train_features_5017_signals.npy', allow_pickle=True)
train4_carmen = np.load('train\\carmen\\area_4_carmen_1FFT_train_features_5017_signals.npy', allow_pickle=True)

test1_carmen = np.load('test\\carmen\\area_1_carmen_1FFT_test_features_1075_signals.npy', allow_pickle=True)
test2_carmen = np.load('test\\carmen\\area_2_carmen_1FFT_test_features_1075_signals.npy', allow_pickle=True)
test3_carmen = np.load('test\\carmen\\area_3_carmen_1FFT_test_features_1075_signals.npy', allow_pickle=True)
test4_carmen = np.load('test\\carmen\\area_4_carmen_1FFT_test_features_1075_signals.npy', allow_pickle=True)

valid1_carmen = np.load('validation\\carmen\\area_1_carmen_1FFT_validation_features_1075_signals.npy', allow_pickle=True)
valid2_carmen = np.load('validation\\carmen\\area_2_carmen_1FFT_validation_features_1075_signals.npy', allow_pickle=True)
valid3_carmen = np.load('validation\\carmen\\area_3_carmen_1FFT_validation_features_1075_signals.npy', allow_pickle=True)
valid4_carmen = np.load('validation\\carmen\\area_4_carmen_1FFT_validation_features_1075_signals.npy', allow_pickle=True)

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

    # print("mfcc:",len(mfccs))
    # print("chroma:",len(chroma))
    # print("mel:",len(mel))
    # print("shannon:",len(shannon))
    # print("sample:",len(sample))
    # print("per:",len(per))
    # print("fft:",len(fft))
    return mfccs, chroma, mel, shannon, sample, per, fft

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
    return np.array(features)

def save_data(res1,res2,res3,res4,monkey):
    np.save("validation\\" + 'area_' + str(1) + '_' + str(monkey) + '_features_validation_' + str(len(res1)) + '_signals.npy',
            np.array(res1))
    np.save("validation\\" + 'area_' + str(2) + '_' + str(monkey) + '_features_validation_' + str(len(res2)) + '_signals.npy',
            np.array(res1))
    np.save("validation\\" + 'area_' + str(3) + '_' + str(monkey) + '_features_validation_' + str(len(res3)) + '_signals.npy',
            np.array(res1))
    np.save("validation\\" + 'area_' + str(4) + '_' + str(monkey) + '_features_validation_' + str(len(res4)) + '_signals.npy',
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

x_train = getX(train1_carmen, train2_carmen, train3_carmen, train4_carmen)
y_train = getY(train1_carmen, train2_carmen, train3_carmen, train4_carmen)

x_test = getX(test1_carmen, test2_carmen, test3_carmen, test4_carmen)
y_test = getY(test1_carmen, test2_carmen, test3_carmen, test4_carmen)

# SVM(x_train,y_train, x_test,y_test)
# randomForest(x_train,y_train, x_test,y_test)
GradientBoosting(x_train,y_train, x_test,y_test)
# SVMovr(x_train,y_train, x_test,y_test)
# linearSVC(x_train,y_train, x_test,y_test)
# KNeighbors(x_train, y_train, x_test, y_test)

# extract_feature(train1_carmen[0])

# print(len(train1_carmen[0]))
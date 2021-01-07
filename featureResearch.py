from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import librosa
from pyentrp import entropy as ent

SAMPLE_RATE = 1000

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

    return mfccs, chroma, mel, shannon, sample, per,fft

def getAllFeaturesVector(area):
    features = []
    for x in tqdm(area):
        feature = np.array([])
        mfccs, chroma, mel, shannon, sample, per, fft = extract_feature(x)
        feature = np.concatenate((feature,mfccs,chroma,mel,shannon, sample, per, fft))
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
    x_1, y_1, z_1 = getTNSE3D(vector1)
    x_2, y_2, z_2 = getTNSE3D(vector2)
    x_3, y_3, z_3 = getTNSE3D(vector3)
    x_4, y_4, z_4 = getTNSE3D(vector4)
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


lst1_penny = np.load('area_1_penny_random_samples.npy', allow_pickle=True)
lst2_penny = np.load('area_2_penny_random_samples.npy', allow_pickle=True)
lst3_penny = np.load('area_3_penny_random_samples.npy', allow_pickle=True)
lst4_penny = np.load('area_4_penny_random_samples.npy', allow_pickle=True)

lst1_carmen = np.load('area_1_carmen_random_samples.npy', allow_pickle=True)
lst2_carmen = np.load('area_2_carmen_random_samples.npy', allow_pickle=True)
lst3_carmen = np.load('area_3_carmen_random_samples.npy', allow_pickle=True)
lst4_carmen = np.load('area_4_carmen_random_samples.npy', allow_pickle=True)

lst1_menta = np.load('area_1_menta_random_samples.npy', allow_pickle=True)
lst2_menta = np.load('area_2_menta_random_samples.npy', allow_pickle=True)
lst3_menta = np.load('area_3_menta_random_samples.npy', allow_pickle=True)
lst4_menta = np.load('area_4_menta_random_samples.npy', allow_pickle=True)

# plotPCAfeature(lst1_penny, lst2_penny, lst3_penny, lst4_penny,'penny')
# plotPCAfeature(lst1_carmen, lst2_carmen, lst3_carmen, lst4_carmen,'carmen')
# plotPCAfeature(lst1_menta, lst2_menta, lst3_menta, lst4_menta,'menta')

# | PCA normalized

# plotPCAallFeaturesVector(lst1_carmen, lst2_carmen, lst3_carmen, lst4_carmen, "carmen")
# plotPCAallFeaturesVector(lst1_penny, lst2_penny, lst3_penny, lst4_penny, "penny")
# plotPCAallFeaturesVector(lst1_menta, lst2_menta, lst3_menta, lst4_menta, "menta")
#
plotTSNEallFeaturesVector(lst1_carmen, lst2_carmen, lst3_carmen, lst4_carmen, "carmen")
plotTSNEallFeaturesVector(lst1_menta, lst2_menta, lst3_menta, lst4_menta, "menta")
plotTSNEallFeaturesVector(lst1_penny, lst2_penny, lst3_penny, lst4_penny, "penny")

import numpy as np

# train1_carmen = np.load('train\\carmen\\area_1_carmen_1FFT_train_features_5017_signals.npy', allow_pickle=True)
# train2_carmen = np.load('train\\carmen\\area_2_carmen_1FFT_train_features_5017_signals.npy', allow_pickle=True)
# train3_carmen = np.load('train\\carmen\\area_3_carmen_1FFT_train_features_5017_signals.npy', allow_pickle=True)
# train4_carmen = np.load('train\\carmen\\area_4_carmen_1FFT_train_features_5017_signals.npy', allow_pickle=True)
#
# test1_carmen = np.load('test\\carmen\\area_1_carmen_1FFT_test_features_1075_signals.npy', allow_pickle=True)
# test2_carmen = np.load('test\\carmen\\area_2_carmen_1FFT_test_features_1075_signals.npy', allow_pickle=True)
# test3_carmen = np.load('test\\carmen\\area_3_carmen_1FFT_test_features_1075_signals.npy', allow_pickle=True)
# test4_carmen = np.load('test\\carmen\\area_4_carmen_1FFT_test_features_1075_signals.npy', allow_pickle=True)
#
# valid1_carmen = np.load('validation\\carmen\\area_1_carmen_1FFT_validation_features_1075_signals.npy', allow_pickle=True)
# valid2_carmen = np.load('validation\\carmen\\area_2_carmen_1FFT_validation_features_1075_signals.npy', allow_pickle=True)
# valid3_carmen = np.load('validation\\carmen\\area_3_carmen_1FFT_validation_features_1075_signals.npy', allow_pickle=True)
# valid4_carmen = np.load('validation\\carmen\\area_4_carmen_1FFT_validation_features_1075_signals.npy', allow_pickle=True)
#
# short_train1_carmen = np.load('train_short\\area_1_carmen_features_extra_validation_5017_signals.npy', allow_pickle=True)
# short_train2_carmen = np.load('train_short\\area_2_carmen_features_extra_validation_5017_signals.npy', allow_pickle=True)
# short_train3_carmen = np.load('train_short\\area_3_carmen_features_extra_validation_5017_signals.npy', allow_pickle=True)
# short_train4_carmen = np.load('train_short\\area_4_carmen_features_extra_validation_5017_signals.npy', allow_pickle=True)
#
# short_test1_carmen = np.load('test_short\\area_1_carmen_features_extra_validation_1075_signals.npy', allow_pickle=True)
# short_test2_carmen = np.load('test_short\\area_2_carmen_features_extra_validation_1075_signals.npy', allow_pickle=True)
# short_test3_carmen = np.load('test_short\\area_3_carmen_features_extra_validation_1075_signals.npy', allow_pickle=True)
# short_test4_carmen = np.load('test_short\\area_4_carmen_features_extra_validation_1075_signals.npy', allow_pickle=True)
#

short_train1_carmen = np.load('train_short\\carmen\\area_1_carmen_features_extra_validation_5017_signals.npy', allow_pickle=True)
short_train2_carmen = np.load('train_short\\carmen\\area_2_carmen_features_extra_validation_5017_signals.npy', allow_pickle=True)
short_train3_carmen = np.load('train_short\\carmen\\area_3_carmen_features_extra_validation_5017_signals.npy', allow_pickle=True)
short_train4_carmen = np.load('train_short\\carmen\\area_4_carmen_features_extra_validation_5017_signals.npy', allow_pickle=True)

short_test1_carmen = np.load('test_short\\carmen\\area_1_carmen_features_extra_validation_1075_signals.npy', allow_pickle=True)
short_test2_carmen = np.load('test_short\\carmen\\area_2_carmen_features_extra_validation_1075_signals.npy', allow_pickle=True)
short_test3_carmen = np.load('test_short\\carmen\\area_3_carmen_features_extra_validation_1075_signals.npy', allow_pickle=True)
short_test4_carmen = np.load('test_short\\carmen\\area_4_carmen_features_extra_validation_1075_signals.npy', allow_pickle=True)


add_short_train1_carmen = np.load('train_short\\carmen\\HDI\\area_1_carmen_features_extra_validation_5017_signals.npy', allow_pickle=True)
add_short_train2_carmen = np.load('train_short\\carmen\\HDI\\area_2_carmen_features_extra_validation_5017_signals.npy', allow_pickle=True)
add_short_train3_carmen = np.load('train_short\\carmen\\HDI\\area_3_carmen_features_extra_validation_5017_signals.npy', allow_pickle=True)
add_short_train4_carmen = np.load('train_short\\carmen\\HDI\\area_4_carmen_features_extra_validation_5017_signals.npy', allow_pickle=True)


add_short_test1_carmen = np.load('test_short\\carmen\\HDI\\area_1_carmen_features_extra_validation_1075_signals.npy', allow_pickle=True)
add_short_test2_carmen = np.load('test_short\\carmen\\HDI\\area_2_carmen_features_extra_validation_1075_signals.npy', allow_pickle=True)
add_short_test3_carmen = np.load('test_short\\carmen\\HDI\\area_3_carmen_features_extra_validation_1075_signals.npy', allow_pickle=True)
add_short_test4_carmen = np.load('test_short\\carmen\\HDI\\area_4_carmen_features_extra_validation_1075_signals.npy', allow_pickle=True)


sig_train_carmen1 = np.load('train\\carmen\\area_1_carmen_train_5017_signals.npy',allow_pickle=True)
sig_train_carmen2 = np.load('train\\carmen\\area_2_carmen_train_5017_signals.npy',allow_pickle=True)
sig_train_carmen3 = np.load('train\\carmen\\area_3_carmen_train_5017_signals.npy',allow_pickle=True)
sig_train_carmen4 = np.load('train\\carmen\\area_4_carmen_train_5017_signals.npy',allow_pickle=True)

sig_test_carmen1 = np.load('test\\carmen\\area_1_carmen_test_1075_signals.npy',allow_pickle=True)
sig_test_carmen2 = np.load('test\\carmen\\area_2_carmen_test_1075_signals.npy',allow_pickle=True)
sig_test_carmen3 = np.load('test\\carmen\\area_3_carmen_test_1075_signals.npy',allow_pickle=True)
sig_test_carmen4 = np.load('test\\carmen\\area_4_carmen_test_1075_signals.npy',allow_pickle=True)

sig_valid_carmen_1 = np.load('validation\\carmen\\area_1_carmen_validation_1075_signals.npy',allow_pickle=True)
sig_valid_carmen_2 = np.load('validation\\carmen\\area_2_carmen_validation_1075_signals.npy',allow_pickle=True)
sig_valid_carmen_3 = np.load('validation\\carmen\\area_3_carmen_validation_1075_signals.npy',allow_pickle=True)
sig_valid_carmen_4 = np.load('validation\\carmen\\area_4_carmen_validation_1075_signals.npy',allow_pickle=True)



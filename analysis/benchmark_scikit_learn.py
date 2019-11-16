import sklearn
import h5py  # needed for MATLAB -v7.3 files
from scipy.io import loadmat
from time import process_time

import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.kernel_ridge import KernelRidge
from sklearn import svm
from sklearn.model_selection import train_test_split

# Directories
wakeman_rootdir = '/Users/matthiastreder/OneDrive - Cardiff University/data/'
#rootdir     = '/home/matthias/WakemanHensonMEEG/'
preprocdir  = wakeman_rootdir + 'preprocessed/'
resultsdir  = wakeman_rootdir + 'results/'

haxby_rootdir = '/Users/matthiastreder/OneDrive - Cardiff University/data/'
#haxby_rootdir     = '/data/neuroimaging/Haxby2001_fMRI/'
haxby_resultsdir  = haxby_rootdir + 'results/'

# which benchmarking analysis to do
do_benchmark_single_subjects = 1
do_benchmark_supersubject = 0
do_benchmark_haxby = 0

# Define models
lda             = LinearDiscriminantAnalysis()
logreg          =  LogisticRegression()
nb              =  GaussianNB()
svm_linear      = svm.SVC(kernel='linear')
svm_rbf         = svm.SVC(kernel='rbf')

ridge           = Ridge(alpha=0.1)
kernel_ridge    = KernelRidge(kernel='rbf')
svr_linear      = svm.LinearSVR()
svr_rbf         = svm.SVR(kernel='rbf')

def get_acc(model, X_tr, y_tr, X_te, y_te):
    '''Fit model and calculate accuracy on test set'''
    return np.mean(model.fit(X_tr, y_tr).predict(X_te) == y_te)

if do_benchmark_single_subjects:
    ################################################
    ##           SINGLE SUBJECT ANALYSIS          ##
    ################################################
    print('Loading MEG single subjects')

    p          = 0.2   # fraction for test set
    ntime      = 265   # n time points
    nsubjects  = 16    # n of subjects
    Z = np.zeros((nsubjects, ntime));

    # Accuracy -- classifiers
    acc_lda        = Z.copy()
    acc_logreg     = Z.copy()
    acc_nb         = Z.copy()
    acc_svm_linear = Z.copy()
    acc_svm_rbf    = Z.copy()

    # Time -- classifiers
    time_lda    = Z.copy()
    time_logreg = Z.copy()
    time_nb     = Z.copy()
    time_svm_linear = Z.copy()
    time_svm_rbf    = Z.copy()

    # Time -- regression models
    time_ridge        = Z.copy()
    time_kernel_ridge = Z.copy()
    time_svr_linear   = Z.copy()
    time_svr_rbf      = Z.copy()

    for n in range(0,nsubjects):   # --- loop across subjects ---

        print(f'**** processing subject #{n+1}')

        filename = f'benchmarking_data_single_subject{n+1}.mat'
        filepath = resultsdir + filename

        # Read class labels and data X
        f = loadmat(filepath)
        clabel = f['clabel'][:,0]
        X  = f['X']
        print(X.shape)

        # Randomly split into train and test set
        X_train, X_test, y_train, y_test = \
                 train_test_split(X, clabel, test_size=p)

        # Create a numerical regression target which roughly represents time-in-experiment
        y = np.array(range(len(clabel)), dtype='float')

        for t in range(ntime):   # --- loop across time points

            if (t % 50 == 0): print(t, '..', end='')
            X_t         = X[:,:,t]

            dat_acc = (X_train[:,:,t], y_train, X_test[:,:,t], y_test)

            ## --- CLASSIFICATION ACCURACY ---
            acc_lda[n,t]        = get_acc(lda, *dat_acc)
            acc_logreg[n,t]     = get_acc(logreg, *dat_acc)
            acc_nb[n,t]         = get_acc(nb, *dat_acc)
            acc_svm_linear[n,t] = get_acc(svm_linear, *dat_acc)
            acc_svm_rbf[n,t]    = get_acc(svm_rbf, *dat_acc)


            ## --- TIMING of classification models ---
            tic = process_time(); lda.fit(X_t, clabel)
            time_lda[n,t] = process_time() - tic
            tic = process_time(); logreg.fit(X_t, clabel)
            time_logreg[n,t] = process_time() - tic
            tic = process_time(); nb.fit(X_t, clabel)
            time_nb[n,t] = process_time() - tic
            tic = process_time(); svm_linear.fit(X_t, clabel)
            time_svm_linear[n,t] = process_time() - tic
            tic = process_time(); svm_rbf.fit(X_t, clabel)
            time_svm_rbf[n,t] = process_time() - tic

            ## --- TIMING of regression models ---
            tic = process_time(); ridge.fit(X_t, clabel)
            time_ridge[n,t] = process_time() - tic
            tic = process_time(); kernel_ridge.fit(X_t, clabel)
            time_kernel_ridge[n,t] = process_time() - tic
            tic = process_time(); svr_linear.fit(X_t, clabel)
            time_svr_linear[n,t] = process_time() - tic
            tic = process_time(); svr_rbf.fit(X_t, clabel)
            time_svr_rbf[n,t] = process_time() - tic

    # Turn average into dataframe and save
    print('Saving data as CSV')
    acc = pd.DataFrame(data={'LDA' : acc_lda.mean(axis=1), 'LogReg' : acc_logreg.mean(axis=1), \
       'Naive Bayes':acc_nb.mean(axis=1), 'SVM (linear)':acc_svm_linear.mean(axis=1), 'SVM (RBF)':acc_svm_rbf.mean(axis=1)})
    acc.to_csv(resultsdir + 'benchmarking_results_single_subjects_scikit_learn_acc.csv')

    time = pd.DataFrame(data={'LDA' : time_lda.mean(axis=1), 'LogReg' : time_logreg.mean(axis=1), \
       'Naive Bayes':time_nb.mean(axis=1), 'SVM (linear)':time_svm_linear.mean(axis=1), 'SVM (RBF)':time_svm_rbf.mean(axis=1), \
       'Ridge':time_ridge.mean(axis=1), 'Kernel Ridge':time_kernel_ridge.mean(axis=1), 'SVR (linear)':time_svr_linear.mean(axis=1), 'SVR (RBF)':time_svr_rbf.mean(axis=1)})
    time.to_csv(resultsdir + 'benchmarking_results_single_subjects_scikit_learn_time.csv')

    print('Finished MEG single-subjects')


if do_benchmark_supersubject:
    ################################################
    ##            SUPER-SUBJECT ANALYSIS          ##
    ################################################
    print('Loading MEG super-subject')

    filepath = resultsdir + 'benchmarking_data_supersubject.mat'

    # Read class labels and data X
    # Read class labels and data X
    f = h5py.File(filepath, mode='r')
    clabel = np.array(f['clabel'], dtype='int')
    X  = np.array(f['M'])
    time  = np.array(f['time'])
    print(X.shape)

    n_time      = len(time)
    Z = np.zeros(n_time);

    # Time -- classifiers
    time_lda    = Z.copy()
    time_logreg = Z.copy()
    time_nb     = Z.copy()
    time_svm_linear = Z.copy()
    time_svm_rbf    = Z.copy()

    # Time -- regression models
    time_ridge        = Z.copy()
    time_kernel_ridge = Z.copy()
    time_svr_linear   = Z.copy()
    time_svr_rbf      = Z.copy()

    # Create a numerical regression target which roughly represents time-in-experiment
    y = np.array(range(n_time), dtype='float')

    for t in range(n_time):   # --- loop across subjects ---

        if t % 10 == 0: print(t, '..')

        X_t         = X[:,:,t]

        ## --- TIMING of classification models ---
        tic = process_time(); lda.fit(X_t, clabel)
        time_lda[t] = process_time() - tic
        tic = process_time(); logreg.fit(X_t, clabel)
        time_logreg[t] = process_time() - tic
        tic = process_time(); nb.fit(X_t, clabel)
        time_nb[t] = process_time() - tic
        tic = process_time(); svm_linear.fit(X_t, clabel)
        time_svm_linear[t] = process_time() - tic
        tic = process_time(); svm_rbf.fit(X_t, clabel)
        time_svm_rbf[t] = process_time() - tic

        ## --- TIMING of regression models ---
        tic = process_time(); ridge.fit(X_t, clabel)
        time_ridge[t] = process_time() - tic
        tic = process_time(); kernel_ridge.fit(X_t, clabel)
        time_kernel_ridge[t] = process_time() - tic
        tic = process_time(); svr_linear.fit(X_t, clabel)
        time_svr_linear[t] = process_time() - tic
        tic = process_time(); svr_rbf.fit(X_t, clabel)
        time_svr_rbf[t] = process_time() - tic

    # Turn average into dataframe and save
    print('Saving data as CSV')
    time = pd.DataFrame(data={'LDA' : time_lda, 'LogReg' : time_logreg, \
      'Naive Bayes':time_nb, 'SVM (linear)':time_svm_linear, 'SVM (RBF)':time_svm_rbf, \
      'Ridge':time_ridge, 'Kernel Ridge':time_kernel_ridge, 'SVR (linear)':time_svr_linear, 'SVR (RBF)':time_svr_rbf})
    time.to_csv(resultsdir + 'benchmarking_results_supersubject_scikit_learn.csv')

    print('Finished super-subject')

if do_benchmark_haxby:
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%               fMRI ANALYSIS                %%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    print('Loading Haxby fMRI')

    nsubjects = 6
    ntime = 216
    Z = np.zeros(nsubjects);

    # Time -- classifiers
    time_lda    = Z.copy()
    time_logreg = Z.copy()
    time_nb     = Z.copy()
    time_svm_linear = Z.copy()
    time_svm_rbf    = Z.copy()

    # Time -- regression models
    time_ridge        = Z.copy()
    time_kernel_ridge = Z.copy()
    time_svr_linear   = Z.copy()
    time_svr_rbf      = Z.copy()


    for n in range(nsubjects):   # --- loop across subjects ---

        print('Subject', n+1)
        filepath = haxby_resultsdir + f'benchmarking_data_haxby_subject{n+1}.mat'

        # Read class labels and data X
        f = h5py.File(filepath, mode='r')
        print(f.keys())
        clabel = np.array(f['clabel'], dtype='int')[0,:]
        X  = np.array(f['X']).transpose()
        print(X.shape)

        # Create a dummy numerical regression target which roughly represents time-in-experiment
        y = np.array(range(ntime), dtype='float')

        ## --- TIMING of classification models ---
        tic = process_time(); lda.fit(X, clabel)
        time_lda[n] = process_time() - tic
        tic = process_time(); logreg.fit(X, clabel)
        time_logreg[n] = process_time() - tic
        tic = process_time(); nb.fit(X, clabel)
        time_nb[n] = process_time() - tic
        tic = process_time(); svm_linear.fit(X, clabel)
        time_svm_linear[n] = process_time() - tic
        tic = process_time(); svm_rbf.fit(X, clabel)
        time_svm_rbf[n] = process_time() - tic

        ## --- TIMING of regression models ---
        tic = process_time(); ridge.fit(X, clabel)
        time_ridge[n] = process_time() - tic
        tic = process_time(); kernel_ridge.fit(X, clabel)
        time_kernel_ridge[n] = process_time() - tic
        tic = process_time(); svr_linear.fit(X, clabel)
        time_svr_linear[n] = process_time() - tic
        tic = process_time(); svr_rbf.fit(X, clabel)
        time_svr_rbf[n] = process_time() - tic

    # Turn average into dataframe and save
    print('Saving data as CSV')
    time = pd.DataFrame(data={'LDA' : time_lda, 'LogReg' : time_logreg, \
      'Naive Bayes':time_nb, 'SVM (linear)':time_svm_linear, 'SVM (RBF)':time_svm_rbf, \
      'Ridge':time_ridge, 'Kernel Ridge':time_kernel_ridge, 'SVR (linear)':time_svr_linear, 'SVR (RBF)':time_svr_rbf})
    time.to_csv(resultsdir + 'benchmarking_results_fmri_scikit_learn.csv')

    print('Finished fMRI')

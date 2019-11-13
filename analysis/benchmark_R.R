# Performs benchmarking on MEG and fMRI data
library(rhdf5)
library(MASS)      # regularized LDA, used in caret
library(glmnet)    # L2 logistic regression
library(e1071)      # Naive Bayes and SVM
library(listdtr)   # Kernel ridge regression

# define directories
rootdir     = '/home/matthias/WakemanHensonMEEG/';
preprocdir  = paste(rootdir, 'preprocessed/', sep='')
resultsdir  = paste(rootdir, 'results/', sep='')
figdir      = paste(rootdir, 'figures/', sep='')

haxby_rootdir     = '/data/neuroimaging/Haxby2001_fMRI/';
haxby_resultsdir  = paste(haxby_rootdir, 'results/', sep='')

# which benchmarking analysis to do
do_benchmark_single_subjects = 0
do_benchmark_supersubject = 1
do_benchmark_haxby = 1

# Logistic regression : glmnet(data,label,family="binomial", alpha=0, lambda=1)


if (do_benchmark_single_subjects) {
  ################################################
  ##           SINGLE SUBJECT ANALYSIS          ##
  ################################################
  library(R.matlab
          )
  print('Loading MEG single subjects')
  
  p          = 0.2   # fraction for test set
  ntime      = 265   # n time points
  nsubjects  = 16    # n of subjects
  Z = matrix(0, nsubjects, ntime);
  
  # Accuracy -- classifiers
  acc_lda        = Z
  acc_logreg     = Z
  acc_nb         = Z
  acc_svm_linear = Z
  acc_svm_rbf    = Z
  
  # Time -- classifiers
  time_lda    = Z
  time_logreg = Z
  time_nb     = Z
  time_svm_linear = Z
  time_svm_rbf    = Z

  # Time -- regression models
  time_ridge        = Z;
  time_kernel_ridge = Z;
  time_svr_linear   = Z;
  time_svr_rbf      = Z;

  
  for (n in 1:nsubjects) {   # --- loop across subjects ---
    
    print(sprintf('**** processing subject #%d', n))
    
    filename = sprintf('benchmarking_data_single_subject%d.mat', n)
    filepath = paste(resultsdir,filename,sep='')
    
    # Read class labels and data X
    M = readMat('/home/matthias/WakemanHensonMEEG/results/benchmarking_data_single_subject1.mat')
    clabel = factor(M$clabel)
    X  = M$X
    shape = attributes(X)$dim
    print(shape)
    
    # Randomly split into train and test set
    test_size = round(p * shape[1])
    test_ind <- sample(shape[1], size = test_size)
    
    X.train = X[-test_ind,, ]
    X.test  = X[test_ind,, ]
    y.train = clabel[-test_ind] 
    y.test  = clabel[test_ind] 
    
    # Create a numerical regression target which roughly represents time-in-experiment
    y = as.numeric(1:length(clabel))
    
    
    for (t in 1:ntime) {     # --- loop across time points
      
      if (t %% 50 == 0) { print(sprintf('%d .. ', t))}
      X.t         = X[, , t]
      X.train.t   = X.train[, , t]
      X.test.t    = X.test[, , t]
      
      ## --- CLASSIFICATION ACCURACY ---
      #gamma = 0.01 # shrinkage parameter 
      #lambda = 1  # use LDA not QDA
      # Somehow regularized LDA gies low performance even when leaving lambda/gamma out?
      #cf_lda    = rda(x = X.train.t, grouping = y.train, lambda = lambda, gamma = gamma) 
      # Use standard LDA instead
      cf_lda    = lda(x = X.train.t, grouping = y.train, CV=F, prior=c(0.5,0.5))
      cf_logreg = glmnet(X.train.t, y.train, family="binomial", alpha=0, lambda=1)
      cf_nb     = naiveBayes(X.train.t, y.train)
      cf_svm_linear = svm(X.train.t, y.train, kernel='linear', cost=1)
      cf_svm_rbf    = svm(X.train.t, y.train, kernel='radial', cost=1)
      
      acc_lda[n,t]      = mean(predict(cf_lda, X.test.t)$class == y.test)
      acc_logreg[n,t]   = mean(predict(cf_logreg, X.test.t, type='class') == y.test)
      acc_nb[n,t]       = mean(predict(cf_nb, X.test.t, type = 'class') == y.test)
      acc_svm_linear[n,t]   = mean(predict(cf_svm_linear, X.test.t) == y.test)
      acc_svm_rbf[n,t]      = mean(predict(cf_svm_rbf, X.test.t) == y.test)
      
      ## --- TIMING of classification models ---
      time_lda[n,t]    = system.time(lda(x = X.t, grouping = clabel, CV=F, prior=c(0.5,0.5)))[3]
      time_logreg[n,t] = system.time(glmnet(X.t, clabel, family="binomial", alpha=0, lambda=1))[3]
      time_nb[n,t]     = system.time(naiveBayes(X.t, clabel))[3]
      time_svm_linear[n,t] = system.time(svm(X.t, clabel, kernel='linear', cost=1))[3]
      time_svm_rbf[n,t]    = system.time(svm(X.t, clabel, kernel='radial', cost=1))[3]
      
      ## --- TIMING of regression models ---
      
      time_ridge[n,t]        = system.time(glmnet(X.t, y, alpha=0, lambda=1))[3]
      # kernel ridge (krr) in the listdtr package has no interface for setting hyperparameters
      # but uses LOOCV to find them - does not seem to finish in reasonable time though so skip it
      #time_kernel_ridge[n,t] = system.time(krr(X.t, y));
      time_svr_linear[n,t]   = system.time(svm(X.t, y, type='eps', kernel='linear', cost=1))[3]
      time_svr_rbf[n,t]      = system.time(svm(X.t, y, type='eps', kernel='radial', cost=1))[3]

    }
  }

  
  print('Saving data as RData')
  save(acc_lda, acc_logreg, acc_nb, acc_svm_linear, acc_svm_rbf,time_lda, 
       time_logreg, time_nb, time_svm_linear, time_svm_rbf, time_ridge, time_kernel_ridge, 
       time_svr_linear, time_svr_rbf, 
       file = paste(resultsdir,'benchmarking_results_single_subjects.RData',sep=''))
  
  # Turn average into dataframe and save
  load(file = paste(resultsdir,'benchmarking_results_single_subjects.RData',sep=''))
  
  print('Saving data as CSV')
  acc = data.frame(colMeans(acc_lda), colMeans(acc_logreg), colMeans(acc_nb), colMeans(acc_svm_linear), colMeans(acc_svm_rbf))
  colnames(acc) <- c('lda','logreg','naive bayes','SVM (linear)', 'SVM (RBF)')
  write.csv(acc, paste(resultsdir,'benchmarking_results_single_subjects_R_acc.csv',sep=''))
  
  time  = data.frame(colMeans(time_kernel_ridge), colMeans(time_lda), colMeans(time_logreg), 
                     colMeans(time_nb), colMeans(time_ridge), colMeans(time_svm_linear), 
                     colMeans(time_svm_rbf),colMeans(time_svr_linear), colMeans(time_svr_rbf))
  colnames(time) <- c('kernel_ridge','lda','logreg','naive bayes','ridge','SVM (linear)', 'SVM (RBF)', 'SVR (linear)', 'SVR (RBF)')
  write.csv(time, paste(resultsdir,'benchmarking_results_single_subjects_R_time.csv',sep=''))
  
  print('Finished single subjects')
  
}


if (do_benchmark_supersubject) {
  ################################################
  ##            SUPER-SUBJECT ANALYSIS          ##
  ################################################
  print('Loading MEG super-subject')
  
  filepath = paste(resultsdir,'benchmarking_data_supersubject.mat',sep='')
  
  # Read class labels and data X
  clabel = factor(h5read(filepath,'clabel'))
  time = h5read(filepath,'time')
  X  = h5read(filepath,'M')
  shape = attributes(X)$dim
  print(shape)
  
  ntime      = length(time)   # n time points
  Z = matrix(0, ntime);
  
  # Time -- classifiers
  time_lda    = Z
  time_logreg = Z
  time_nb     = Z
  time_svm_linear = Z
  time_svm_rbf    = Z
  
  # Time -- regression models
  time_ridge        = Z;
  time_svr_linear   = Z;
  time_svr_rbf      = Z;
  
  
  for (t in 1:ntime) {   # --- loop across subjects ---
    
    # Create a numerical regression target which roughly represents time-in-experiment
    y = as.numeric(1:length(clabel))
    
    
    if (t %% 10 == 0) { print(sprintf('%d .. ', t))}
    X.t         = X[, , t]
    
    ## --- TIMING of classification models ---
    time_lda[t]    = system.time(lda(x = X.t, grouping = clabel, CV=F, prior=c(0.5,0.5)))[3]
    time_logreg[t] = system.time(glmnet(X.t, clabel, family="binomial", alpha=0, lambda=1))[3]
    time_nb[t]     = system.time(naiveBayes(X.t, clabel))[3]
    time_svm_linear[t] = system.time(svm(X.t, clabel, kernel='linear', cost=1))[3]
    time_svm_rbf[t]    = system.time(svm(X.t, clabel, kernel='radial', cost=1))[3]
    
    ## --- TIMING of regression models ---
    
    time_ridge[t]        = system.time(glmnet(X.t, y, alpha=0, lambda=1))[3]
    time_svr_linear[t]   = system.time(svm(X.t, y, type='eps', kernel='linear', cost=1))[3]
    time_svr_rbf[t]      = system.time(svm(X.t, y, type='eps', kernel='radial', cost=1))[3]
    
    
  }
  
  
  print('Saving supersubject data as RData')
  save(time_lda, time_logreg, time_nb, time_svm_linear, time_svm_rbf, time_ridge, 
       time_svr_linear, time_svr_rbf, 
       file = paste(resultsdir,'benchmarking_results_supersubject.RData',sep=''))
  
  # Turn average into dataframe and save
  print('Saving data as CSV')
  time  = data.frame(time_lda, time_logreg, 
                     time_nb, time_ridge, time_svm_linear, 
                     time_svm_rbf,time_svr_linear, time_svr_rbf)
  colnames(time) <- c('lda','logreg','naive bayes','ridge','SVM (linear)', 'SVM (RBF)', 'SVR (linear)', 'SVR (RBF)')
  write.csv(time, paste(resultsdir,'benchmarking_results_supersubject_R_time.csv',sep=''))
  
  print('Finished super-subject')
  
}



if (do_benchmark_haxby) {
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%               fMRI ANALYSIS                %%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  print('Loading Haxby fMRI')
  
  nsubjects = 6
  Z = matrix(0, nsubjects);
  
  # Time -- classifiers
  time_lda    = Z
  time_logreg = Z
  time_nb     = Z
  time_svm_linear = Z
  time_svm_rbf    = Z
  
  # Time -- regression models
  time_ridge        = Z;
  time_svr_linear   = Z;
  time_svr_rbf      = Z;
  
  
  for (n in 1:nsubjects) {   # --- loop across subjects ---
    
    filepath = paste(haxby_resultsdir, sprintf('benchmarking_data_haxby_subject%d.mat',n),sep='')
    
    # Read class labels and data X
    clabel = factor(h5read(filepath,'clabel'))
    X = h5read(filepath,'X')
    shape = attributes(X)$dim
    print(shape)
    
    # Create a dummy numerical regression target which roughly represents time-in-experiment
    y = as.numeric(1:length(clabel))

    ## --- TIMING of classification models ---
    ## doesnt work - OOM for LDA
    #time_lda[n]    = system.time(lda(x = X, grouping = clabel, CV=F, prior=c(0.5,0.5)))[3] 
    time_logreg[n] = system.time(glmnet(X, clabel, family="binomial", alpha=0, lambda=1))[3]
    time_nb[n]     = system.time(naiveBayes(X, clabel))[3]
    time_svm_linear[n] = system.time(svm(X, clabel, kernel='linear', cost=1))[3]
    time_svm_rbf[n]    = system.time(svm(X, clabel, kernel='radial', cost=1))[3]
    
    ## --- TIMING of regression models ---
    time_ridge[n]        = system.time(glmnet(X, y, family = 'gaussian',alpha=0, lambda=1))[3]
    time_svr_linear[n]   = system.time(svm(X, y, type='eps', kernel='linear', cost=1))[3]
    time_svr_rbf[n]      = system.time(svm(X, y, type='eps', kernel='radial', cost=1))[3]
  }
  
  print('Saving fMRI results data as RData')
  save(time_lda, time_logreg, time_nb, time_svm_linear, time_svm_rbf, time_ridge, 
       time_svr_linear, time_svr_rbf, 
       file = paste(haxby_resultsdir,'benchmarking_results_fMRI.RData',sep=''))
  
  # Turn average into dataframe and save
  print('Saving data as CSV')
  time  = data.frame(time_lda, time_logreg, 
                     time_nb, time_ridge, time_svm_linear, 
                     time_svm_rbf,time_svr_linear, time_svr_rbf)
  colnames(time) <- c('lda','logreg','naive bayes','ridge','SVM (linear)', 'SVM (RBF)', 'SVR (linear)', 'SVR (RBF)')
  write.csv(time, paste(haxby_resultsdir,'benchmarking_results_fMRI_R_time.csv',sep=''))
  
  print('Finished fMRI')
  
}


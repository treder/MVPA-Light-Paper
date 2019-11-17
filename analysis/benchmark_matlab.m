% Performs benchmarking on Wakeman & Henson and fMRI data:
% - out of the box classification performance (using classifiers with their
%   default settings)
% - computational speed: how long does it take to train the classifier 
%
% Tested with:
% Dataset: Wakeman and Henson 1.0.3, downloaded from https://openneuro.org/datasets/ds000117/versions/1.0.3
% MATLAB: R2019a
% FieldTrip: revision r8588-7599-g94c95e995 (August 2019)
clear
close all

% define directories
rootdir     = '/home/matthias/WakemanHensonMEEG/';
preprocdir  = [rootdir 'preprocessed/'];
resultsdir  = [rootdir 'results/'];
figdir      = [rootdir 'figures/'];

haxby_rootdir     = '/data/neuroimaging/Haxby2001_fMRI/';
haxby_resultsdir  = [haxby_rootdir 'results/'];

% which benchmarking analysis to do
do_benchmark_single_subjects = 0;
do_benchmark_supersubject = 0;
do_benchmark_haxby = 1;

if do_benchmark_single_subjects
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%           SINGLE SUBJECT ANALYSIS          %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Loading MEG single subjects\n')
load([resultsdir 'benchmarking_data_single_subjects.mat'],'clabels','meg','nsubjects')
dim             = [];
p               = 0.2;

% clabels features three types of clabels per subject
% 1: three classes
% 2: famous vs scrambled faces
% 3: famous vs familiar faces

meg_acc = cell(nsubjects, 1);
meg_time = cell(nsubjects, 1);

metric = 'accuracy';

for n=1:nsubjects     % --- loop across subjects
    
    fprintf('**** processing subject #%d\n', n)
    dat = meg{n};
    
    %% Randomly split into train/test set
    n_time = length(dat.time);
    clabel = clabels{n};
    [X_train, y_train, X_test, y_test] = split_into_train_and_test_set(dat.trial, clabel, p);
    
    %% init acc and time variables
    Z = zeros(n_time, 1);
    
    acc = struct();
    % MVPA Light - accuracy
    acc.mv_lda        = Z;
    acc.mv_logreg     = Z;
    acc.mv_nb         = Z;
    acc.mv_svm_linear = Z;
    acc.mv_svm_rbf    = Z;
    acc.mv_libsvm_linear    = Z;
    acc.mv_libsvm_rbf       = Z;
    acc.mv_liblinear  = Z;
    
    % MATLAB - accuracy
    acc.mat_lda        = Z;
    acc.mat_logreg     = Z;
    acc.mat_nb         = Z;
    acc.mat_svm_linear = Z;
    acc.mat_svm_rbf    = Z;
    
    time = struct();
    % MVPA Light - classification (time)
    time.mv_lda        = Z;
    time.mv_logreg     = Z;
    time.mv_nb         = Z;
    time.mv_svm_linear = Z;
    time.mv_svm_rbf    = Z;
    time.mv_libsvm_linear    = Z;
    time.mv_libsvm_rbf       = Z;
    time.mv_liblinear_svm_primal = Z;
    time.mv_liblinear_svm_dual   = Z;
    time.mv_liblinear_logreg_primal = Z;
    time.mv_liblinear_logreg_dual   = Z;
    
    % MATLAB - classification (time)
    time.mat_lda        = Z;
    time.mat_logreg     = Z;
    time.mat_nb         = Z;
    time.mat_svm_linear = Z;
    time.mat_svm_rbf    = Z;
    
    % MVPA Light - regression (time)
    time.mv_ridge        = Z;
    time.mv_kernel_ridge = Z;
    time.mv_svr_linear   = Z;
    time.mv_svr_rbf      = Z;
    
    % MATLAB - regression (time)
    time.mat_ridge        = Z;
    time.mat_svr_linear = Z;
    time.mat_svr_rbf    = Z;

    
    
    %% Out of the box classification
    for t=1:n_time     % --- loop across time points
        
        if mod(t,50)==0, fprintf('%d .. ', t); end
        X_t         = dat.trial(:,:,t);
        X_train_t   = X_train(:,:,t);
        X_test_t    = X_test(:,:,t);

        %% --- CLASSIFICATION ACCURACY ---

        %% Get default hyperparameters
        param_mv_lda        = mv_get_hyperparameter('lda');
        param_mv_logreg     = mv_get_hyperparameter('logreg');
        param_mv_nb         = mv_get_hyperparameter('naive_bayes');
        param_mv_svm_linear = mv_get_hyperparameter('svm');
        param_mv_svm_rbf    = mv_get_hyperparameter('svm');
        param_mv_libsvm_linear      = mv_get_hyperparameter('libsvm');
        param_mv_libsvm_rbf         = mv_get_hyperparameter('libsvm');
        param_mv_liblinear          = mv_get_hyperparameter('liblinear');
        
        
        param_mv_svm_linear.kernel = 'linear';
        param_mv_svm_rbf.kernel    = 'rbf';
        param_mv_libsvm_linear.kernel   = 'linear';
        param_mv_libsvm_rbf.kernel      = 'rbf';
        
        %% MVPA-Light classifiers
        
        cf_mv_lda       = train_lda(param_mv_lda, X_train_t, y_train);
        cf_mv_logreg    = train_logreg(param_mv_logreg, X_train_t, y_train);
        cf_mv_nb        = train_naive_bayes(param_mv_nb, X_train_t, y_train);
        cf_mv_svm_linear = train_svm(param_mv_svm_linear, X_train_t, y_train);
        cf_mv_svm_rbf   = train_svm(param_mv_svm_rbf, X_train_t, y_train);
        cf_mv_libsvm_linear     = train_libsvm(param_mv_libsvm_linear, X_train_t, y_train);
        cf_mv_libsvm_rbf        = train_libsvm(param_mv_libsvm_rbf, X_train_t, y_train);
        cf_mv_liblinear         = train_liblinear(param_mv_liblinear, X_train_t, y_train);
        
        acc.mv_lda(t)       = mv_calculate_performance(metric, 'clabel', test_lda(cf_mv_lda, X_test_t), y_test, dim);
        acc.mv_logreg(t)    = mv_calculate_performance(metric, 'clabel', test_logreg(cf_mv_logreg, X_test_t), y_test, dim);
        acc.mv_nb(t)        = mv_calculate_performance(metric, 'clabel', test_naive_bayes(cf_mv_nb, X_test_t), y_test, dim);
        acc.mv_svm_linear(t)    = mv_calculate_performance(metric, 'clabel', test_svm(cf_mv_svm_linear, X_test_t), y_test, dim);
        acc.mv_svm_rbf(t)       = mv_calculate_performance(metric, 'clabel', test_svm(cf_mv_svm_rbf, X_test_t), y_test, dim);
        acc.mv_libsvm_linear(t) = mv_calculate_performance(metric, 'clabel', test_libsvm(cf_mv_libsvm_linear, X_test_t), y_test, dim);
        acc.mv_libsvm_rbf(t)    = mv_calculate_performance(metric, 'clabel', test_libsvm(cf_mv_libsvm_rbf, X_test_t), y_test, dim);
        acc.mv_liblinear(t) = mv_calculate_performance(metric, 'clabel', test_liblinear(cf_mv_liblinear, X_test_t), y_test, dim);

        %% MATLAB classifier default parameters
        param_mat_lda           = {'DiscrimType' 'linear'};
        param_mat_logreg        = {'binomial' 'Link' 'logit' 'Alpha' 0.01 'lambda', 1};
        param_mat_nb            = {'DistributionNames' 'normal'};
        param_mat_svm_linear    = {'KernelFunction' 'linear'};
        param_mat_svm_rbf       = {'KernelFunction' 'RBF'};

        %% MATLAB classifiers
        cf_mat_lda      = fitcdiscr(X_train_t ,y_train, param_mat_lda{:});
        [cf_mat_logreg, FitInfo]   = lassoglm(X_train_t, y_train==1, param_mat_logreg{:});
        yhat_logreg = double(glmval([cf_mat_logreg; FitInfo.Intercept], X_test_t,'logit')>0.5);
        yhat_logreg(yhat_logreg==0)= 2;
        
        cf_mat_nb           = fitcnb(X_train_t, y_train, param_mat_nb{:});
        cf_mat_svm_linear   = fitcsvm(X_train_t, y_train, param_mat_svm_linear{:});
        cf_mat_svm_rbf      = fitcsvm(X_train_t, y_train, param_mat_svm_rbf{:});
        
        acc.mat_lda(t)       = mv_calculate_performance(metric, 'clabel', cf_mat_lda.predict(X_test_t), y_test, dim);
        acc.mat_logreg(t)    = mv_calculate_performance(metric, 'clabel', yhat_logreg, y_test, dim);
        acc.mat_nb(t)           = mv_calculate_performance(metric, 'clabel', cf_mat_nb.predict(X_test_t), y_test, dim);
        acc.mat_svm_linear(t)   = mv_calculate_performance(metric, 'clabel', cf_mat_svm_linear.predict(X_test_t), y_test, dim);
        acc.mat_svm_rbf(t)      = mv_calculate_performance(metric, 'clabel', cf_mat_svm_rbf.predict(X_test_t), y_test, dim);

        
        %% --- TIMING of classification models ---
                
        %% MVPA-Light -- classifiers
        % disable hyperparameter optimization to measure pure training time
        param_mv_lda.lambda     = 0.01;
        param_mv_logreg.reg     = 'l2';
        param_mv_logreg.lambda  = 1;
        
        param_mv_liblinear_svm_primal   = mv_get_hyperparameter('liblinear');
        param_mv_liblinear_svm_dual     = mv_get_hyperparameter('liblinear');
        param_mv_liblinear_svm_primal.type = 2;
        param_mv_liblinear_svm_dual.type   = 1;
        
        param_mv_liblinear_logreg_primal   = mv_get_hyperparameter('liblinear');
        param_mv_liblinear_logreg_dual     = mv_get_hyperparameter('liblinear');
        param_mv_liblinear_logreg_primal.type  = 0;        % primal logistic regression
        param_mv_liblinear_logreg_dual.type    = 7;        % dual logistic regression
        
    
        %% Time it - MVPA-Light -- classifiers
        tic; train_lda(param_mv_lda, X_train_t, y_train);
        time.mv_lda(t) = toc;
        tic; train_logreg(param_mv_logreg, X_train_t, y_train);
        time.mv_logreg(t) = toc;
        tic; train_naive_bayes(param_mv_nb, X_train_t, y_train);
        time.mv_nb(t) = toc;
        
        tic; train_svm(param_mv_svm_linear, X_train_t, y_train);
        time.mv_svm_linear(t) = toc;
        tic; train_svm(param_mv_svm_rbf, X_train_t, y_train);
        time.mv_svm_rbf(t) = toc;
        
        tic; train_libsvm(param_mv_libsvm_linear, X_train_t, y_train);
        time.mv_libsvm_linear(t) = toc;
        tic; train_libsvm(param_mv_libsvm_rbf, X_train_t, y_train);
        time.mv_libsvm_rbf(t) = toc;
        
        tic; train_liblinear(param_mv_liblinear_svm_primal, X_train_t, y_train);
        time.mv_liblinear_svm_primal(t) = toc;
        tic; train_liblinear(param_mv_liblinear_svm_dual, X_train_t, y_train);
        time.mv_liblinear_svm_dual(t) = toc;
        tic; train_liblinear(param_mv_liblinear_logreg_primal, X_train_t, y_train);
        time.mv_liblinear_logreg_primal(t) = toc;
        tic; train_liblinear(param_mv_liblinear_logreg_dual, X_train_t, y_train);
        time.mv_liblinear_logreg_dual(t) = toc;

        %% Time it - MATLAB -- classifiers
        % disable hyperparameter optimization to measure pure training time
        param_mat_lda           = {'DiscrimType' 'linear' 'Gamma' 0.01};

        tic; fitcdiscr(X_train_t ,y_train, param_mat_lda{:});
        time.mat_lda(t) = toc;
        tic; lassoglm(X_train_t, y_train==1, param_mat_logreg{:});
        time.mat_logreg(t) = toc;
        tic; fitcnb(X_train_t, y_train, param_mat_nb{:});
        time.mat_nb(t) = toc;
        tic; fitcsvm(X_train_t, y_train, param_mat_svm_linear{:});
        time.mat_svm_linear(t) = toc;
        tic; fitcsvm(X_train_t, y_train, param_mat_svm_rbf{:});
        time.mat_svm_rbf(t) = toc;
        
        %% --- TIMING of regression models ---
        % Since we do not have a real regression target, we will simply 
        % take the trial number to predict 'time-in-experiment'.
        % Furthermore, the test set is not needed here since we only train,
        % so use full data for training.
        y = (1:size(X_t,1))';
        
        %% Time it - MVPA-Light -- regression
        param_mv_ridge = mv_get_hyperparameter('ridge');
        param_mv_ridge.lambda = 0.1;
        param_mv_kernel_ridge = mv_get_hyperparameter('kernel_ridge');
        param_mv_svr_linear = mv_get_hyperparameter('svr');
        param_mv_svr_rbf    = mv_get_hyperparameter('svr');
        param_mv_svr_linear.kernel  = 'linear';
        param_mv_svr_rbf.kernel     = 'rbf';
        
        tic; train_ridge(param_mv_ridge, X_t, y);
        time.mv_ridge(t) = toc;
        tic; train_kernel_ridge(param_mv_kernel_ridge, X_t, y);
        time.mv_kernel_ridge(t) = toc;
        tic; train_svr(param_mv_svr_linear, X_t, y);
        time.mv_svr_linear(t) = toc;
        tic; train_svr(param_mv_svr_rbf, X_t, y);
        time.mv_svr_rbf(t) = toc;
        
        %% Time it - MATLAB -- regression
        param_mat_svr_linear    = {'KernelFunction' 'linear'};
        param_mat_svr_rbf       = {'KernelFunction' 'RBF'};

        tic; ridge(y, X_t, 0.1);
        time.mat_ridge(t) = toc;
        tic; fitrsvm(X_t, y, param_mat_svr_linear{:});
        time.mat_svr_linear(t) = toc;
        tic; fitrsvm(X_t, y, param_mat_svr_rbf{:});
        time.mat_svr_rbf(t) = toc;

    end
    fprintf('\n')

    %% save
    meg_acc{n} = acc;
    meg_time{n} = time;
end

save([resultsdir 'benchmarking_results_single_subjects'],'nsubjects',...
    'meg_acc','meg_time')

fprintf('Finished single subjects.\n')

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%           SUPER-SUBJECT ANALYSIS          %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if do_benchmark_supersubject
% We are only doing the timing here
clear meg clabels
fprintf('Loading MEG super-subject\n')
load([resultsdir 'benchmarking_data_supersubject.mat'],'M', 'clabel','time')

% init time
n_time = length(time);
Z = zeros(n_time, 1);

time = struct();
% MVPA Light - classification
time.mv_lda        = Z;
time.mv_logreg     = Z;
time.mv_nb         = Z;
time.mv_svm_linear = Z;
time.mv_svm_rbf    = Z;
time.mv_libsvm_linear    = Z;
time.mv_libsvm_rbf       = Z;
time.mv_liblinear_svm_primal = Z;
time.mv_liblinear_svm_dual   = Z;
time.mv_liblinear_logreg_primal = Z;
time.mv_liblinear_logreg_dual   = Z;

% MATLAB - classification
time.mat_lda        = Z;
time.mat_logreg     = Z;
time.mat_nb         = Z;
time.mat_svm_linear = Z;
time.mat_svm_rbf    = Z;

% MVPA Light - regression
time.mv_ridge        = Z;
time.mv_kernel_ridge = Z;
time.mv_svr_linear   = Z;
time.mv_svr_rbf      = Z;

% MATLAB - regression
time.mat_ridge        = Z;
time.mat_svr_linear = Z;
time.mat_svr_rbf    = Z;



for t=1:n_time     % --- loop across time points
    if mod(t,50)==0, fprintf('%d ..', t); end
    
    X_t         = M(:,:,t);
    
    %% --- TIMING of classification models ---
    %% MVPA-Light -- classifiers
    % disable hyperparameter optimization to measure pure training time
    param_mv_lda        = mv_get_hyperparameter('lda');
    param_mv_lda.lambda     = 0.01;

    param_mv_logreg     = mv_get_hyperparameter('logreg');
    param_mv_logreg.reg     = 'l2';
    param_mv_logreg.lambda  = 1;

    param_mv_nb         = mv_get_hyperparameter('naive_bayes');

    param_mv_svm_linear = mv_get_hyperparameter('svm');
    param_mv_svm_rbf    = mv_get_hyperparameter('svm');
    param_mv_svm_linear.kernel = 'linear';
    param_mv_svm_rbf.kernel    = 'rbf';
    
    param_mv_libsvm_linear      = mv_get_hyperparameter('libsvm');
    param_mv_libsvm_rbf         = mv_get_hyperparameter('libsvm');
    param_mv_libsvm_linear.kernel   = 'linear';
    param_mv_libsvm_rbf.kernel      = 'rbf';

    param_mv_liblinear_svm_primal   = mv_get_hyperparameter('liblinear');
    param_mv_liblinear_svm_dual     = mv_get_hyperparameter('liblinear');
    param_mv_liblinear_svm_primal.type = 2;         % primal linear svm
    param_mv_liblinear_svm_dual.type   = 1;         % dual linear svm
    
    param_mv_liblinear_logreg_primal   = mv_get_hyperparameter('liblinear');
    param_mv_liblinear_logreg_dual     = mv_get_hyperparameter('liblinear');
    param_mv_liblinear_logreg_primal.type  = 0;        % primal logistic regression
    param_mv_liblinear_logreg_dual.type    = 7;        % dual logistic regression

    %% Time it - MVPA-Light -- classifiers
    tic; train_lda(param_mv_lda, X_t, clabel);
    time.mv_lda(t) = toc;
    tic; train_logreg(param_mv_logreg, X_t, clabel);
    time.mv_logreg(t) = toc;
    tic; train_naive_bayes(param_mv_nb, X_t, clabel);
    time.mv_nb(t) = toc;
    tic; train_svm(param_mv_svm_linear, X_t, clabel);
    time.mv_svm_linear(t) = toc;
    tic; train_svm(param_mv_svm_rbf, X_t, clabel);
    time.mv_svm_rbf(t) = toc;
    tic; train_libsvm(param_mv_libsvm_linear, X_t, clabel);
    time.mv_libsvm_linear(t) = toc;
    tic; train_libsvm(param_mv_libsvm_rbf, X_t, clabel);
    time.mv_libsvm_rbf(t) = toc;
    tic; train_liblinear(param_mv_liblinear_svm_primal, X_t, clabel);
    time.mv_liblinear_svm_primal(t) = toc;
    tic; train_liblinear(param_mv_liblinear_svm_dual, X_t, clabel);
    time.mv_liblinear_svm_dual(t) = toc;
    tic; train_liblinear(param_mv_liblinear_logreg_primal, X_t, clabel);
    time.mv_liblinear_logreg_primal(t) = toc;
    tic; train_liblinear(param_mv_liblinear_logreg_dual, X_t, clabel);
    time.mv_liblinear_logreg_dual(t) = toc;
    
    %% MATLAB classifier default parameters
    % disable hyperparameter optimization to measure pure training time
    param_mat_lda           = {'DiscrimType' 'linear' 'Gamma' 0.01};
    param_mat_logreg        = {'binomial' 'Link' 'logit' 'Alpha' 0.01 'lambda', 1};
    param_mat_nb            = {'DistributionNames' 'normal'};
    param_mat_svm_linear    = {'KernelFunction' 'linear'};
    param_mat_svm_rbf       = {'KernelFunction' 'RBF'};

    %% Time it - MATLAB -- classifiers
    
    tic; fitcdiscr(X_t ,clabel, param_mat_lda{:});
    time.mat_lda(t) = toc;
    tic; lassoglm(X_t, clabel==1, param_mat_logreg{:});
    time.mat_logreg(t) = toc;
    tic; fitcnb(X_t, clabel, param_mat_nb{:});
    time.mat_nb(t) = toc;
    tic; fitcsvm(X_t, clabel, param_mat_svm_linear{:});
    time.mat_svm_linear(t) = toc;
    tic; fitcsvm(X_t, clabel, param_mat_svm_rbf{:});
    time.mat_svm_rbf(t) = toc;
    
    %% --- TIMING of regression models ---
    % Since we do not have a real regression target, we will simply
    % take the trial number to predict 'time-in-experiment'.
    % Furthermore, the test set is not needed here since we only train,
    % so use full data for training.
    y = (1:size(X_t,1))';
        
    %% Time it - MVPA-Light -- regression
    
    param_mv_ridge = mv_get_hyperparameter('ridge');
    param_mv_ridge.lambda = 0.1;
    param_mv_kernel_ridge = mv_get_hyperparameter('kernel_ridge');
    param_mv_svr_linear = mv_get_hyperparameter('svr');
    param_mv_svr_rbf    = mv_get_hyperparameter('svr');
    param_mv_svr_linear.kernel  = 'linear';
    param_mv_svr_rbf.kernel     = 'rbf';
    
    tic; train_ridge(param_mv_ridge, X_t, y);
    time.mv_ridge(t) = toc;
    tic; train_kernel_ridge(param_mv_kernel_ridge, X_t, y);
    time.mv_kernel_ridge(t) = toc;
    tic; train_svr(param_mv_svr_linear, X_t, y);
    time.mv_svr_linear(t) = toc;
    tic; train_svr(param_mv_svr_rbf, X_t, y);
    time.mv_svr_rbf(t) = toc;
    
    %% Time it - MATLAB -- regression
    param_mat_svr_linear    = {'KernelFunction' 'linear'};
    param_mat_svr_rbf       = {'KernelFunction' 'RBF'};
    
    tic; ridge(y, X_t, 0.1);
    time.mat_ridge(t) = toc;
    tic; fitrsvm(X_t, y, param_mat_svr_linear{:});
    time.mat_svr_linear(t) = toc;
    tic; fitrsvm(X_t, y, param_mat_svr_rbf{:});
    time.mat_svr_rbf(t) = toc;
    
end

fprintf('\nSaving results for supersubject\n')
save([resultsdir 'benchmarking_results_supersubject'],'time')

fprintf('Finished supersubject.\n')
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%               fMRI ANALYSIS                %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if do_benchmark_haxby
% We are only doing the timing here
clear M clabels
fprintf('Loading fMRI\n')
load([haxby_resultsdir 'benchmarking_data_haxby.mat'],'fmri', 'clabels')

n_subjects = length(fmri);

Z = zeros(n_subjects, 1);
n_voxel = Z;
time = struct();

% MVPA Light -- classifier
time.mv_lda        = Z;
time.mv_logreg     = Z;
time.mv_nb         = Z;
time.mv_svm_linear = Z;
time.mv_svm_rbf    = Z;
time.mv_libsvm_linear    = Z;
time.mv_libsvm_rbf       = Z;
time.mv_liblinear_svm_primal = Z;
time.mv_liblinear_svm_dual   = Z;
time.mv_liblinear_logreg_primal = Z;
time.mv_liblinear_logreg_dual   = Z;

% MATLAB -- classifier
time.mat_lda        = Z;
time.mat_logreg     = Z;
time.mat_nb         = Z;
time.mat_svm_linear = Z;
time.mat_svm_rbf    = Z;

% MVPA Light -- regression
time.mv_ridge        = Z;
time.mv_kernel_ridge = Z;
time.mv_svr_linear   = Z;
time.mv_svr_rbf      = Z;

% MATLAB -- regression
time.mat_ridge        = Z;
time.mat_svr_linear = Z;
time.mat_svr_rbf    = Z;

for n=1:n_subjects     % --- loop across time points
    
    fprintf('*** Subject %d ***\n', n)
    
    X = fmri{n};   % trials x voxels
    clabel = clabels{n};
    n_voxel(n) = size(X,2);

    %% --- TIMING of classification models ---
    
    %% MVPA-Light -- classifiers
    % disable hyperparameter optimization to measure pure training time
    param_mv_lda        = mv_get_hyperparameter('lda');
    param_mv_lda.lambda     = 0.01;

    param_mv_logreg     = mv_get_hyperparameter('logreg');
    param_mv_logreg.reg     = 'l2';
    param_mv_logreg.lambda  = 1;

    param_mv_nb         = mv_get_hyperparameter('naive_bayes');

    param_mv_svm_linear = mv_get_hyperparameter('svm');
    param_mv_svm_rbf    = mv_get_hyperparameter('svm');
    param_mv_svm_linear.kernel = 'linear';
    param_mv_svm_rbf.kernel    = 'rbf';
    
    param_mv_libsvm_linear      = mv_get_hyperparameter('libsvm');
    param_mv_libsvm_rbf         = mv_get_hyperparameter('libsvm');
    param_mv_libsvm_linear.kernel   = 'linear';
    param_mv_libsvm_rbf.kernel      = 'rbf';

    param_mv_liblinear_svm_primal   = mv_get_hyperparameter('liblinear');
    param_mv_liblinear_svm_dual     = mv_get_hyperparameter('liblinear');
    param_mv_liblinear_svm_primal.type = 2;         % primal linear svm
    param_mv_liblinear_svm_dual.type   = 1;         % dual linear svm
    
    param_mv_liblinear_logreg_primal   = mv_get_hyperparameter('liblinear');
    param_mv_liblinear_logreg_dual     = mv_get_hyperparameter('liblinear');
    param_mv_liblinear_logreg_primal.type  = 0;        % primal logistic regression
    param_mv_liblinear_logreg_dual.type    = 7;        % dual logistic regression

    %% Time it - MVPA-Light -- classifiers
    tic; train_lda(param_mv_lda, X, clabel);
    time.mv_lda(n) = toc;
%     tic; train_logreg(param_mv_logreg, X, clabel); % OUT OF MEMORY ~ 200 GB required
%     time.mv_logreg(n) = toc;
    tic; train_naive_bayes(param_mv_nb, X, clabel);
    time.mv_nb(n) = toc;
    tic; train_svm(param_mv_svm_linear, X, clabel);
    time.mv_svm_linear(n) = toc;
    tic; train_svm(param_mv_svm_rbf, X, clabel);
    time.mv_svm_rbf(n) = toc;
    tic; train_libsvm(param_mv_libsvm_linear, X, clabel);
    time.mv_libsvm_linear(n) = toc;
    tic; train_libsvm(param_mv_libsvm_rbf, X, clabel);
    time.mv_libsvm_rbf(n) = toc;
    tic; train_liblinear(param_mv_liblinear_svm_primal, X, clabel);
    time.mv_liblinear_svm_primal(n) = toc;
    tic; train_liblinear(param_mv_liblinear_svm_dual, X, clabel);
    time.mv_liblinear_svm_dual(n) = toc;
    tic; train_liblinear(param_mv_liblinear_logreg_primal, X, clabel);
    time.mv_liblinear_logreg_primal(n) = toc;
    tic; train_liblinear(param_mv_liblinear_logreg_dual, X, clabel);
    time.mv_liblinear_logreg_dual(n) = toc;
    
    %% MATLAB classifier default parameters
    % disable hyperparameter optimization to measure pure training time
    param_mat_lda           = {'DiscrimType' 'linear' 'Gamma' 0.01};
    param_mat_logreg        = {'binomial' 'Link' 'logit' 'Alpha' 0.01 'lambda', 1};
    param_mat_nb            = {'DistributionNames' 'normal'};
    param_mat_svm_linear    = {'KernelFunction' 'linear'};
    param_mat_svm_rbf       = {'KernelFunction' 'RBF'};

    %% Time it - MATLAB -- classifiers
%     tic; fitcdiscr(X ,clabel, param_mat_lda{:});  % OUT OF MEMORY ~ 200 GB required
%     time.mat_lda(n) = toc;
    tic; lassoglm(X, clabel==1, param_mat_logreg{:});
    time.mat_logreg(n) = toc;
    tic; fitcnb(X, clabel, param_mat_nb{:});
    time.mat_nb(n) = toc;
    tic; fitcsvm(X, clabel, param_mat_svm_linear{:});
    time.mat_svm_linear(n) = toc;
    tic; fitcsvm(X, clabel, param_mat_svm_rbf{:});
    time.mat_svm_rbf(n) = toc;
    
    %% --- TIMING of regression models ---
    % Since we do not have a real regression target, we will simply
    % take the trial number to predict 'time-in-experiment'.
    % Furthermore, the test set is not needed here since we only train,
    % so use full data for training.
    y = (1:size(X,1))';

    %% Time it - MVPA-Light -- regression
    param_mv_ridge = mv_get_hyperparameter('ridge');
    param_mv_ridge.lambda = 0.1;
    param_mv_kernel_ridge = mv_get_hyperparameter('kernel_ridge');
    param_mv_svr_linear = mv_get_hyperparameter('svr');
    param_mv_svr_rbf    = mv_get_hyperparameter('svr');
    param_mv_svr_linear.kernel  = 'linear';
    param_mv_svr_rbf.kernel     = 'rbf';
    
    tic; train_ridge(param_mv_ridge, X, y);
    time.mv_ridge(n) = toc;
    tic; train_kernel_ridge(param_mv_kernel_ridge, X, y);
    time.mv_kernel_ridge(n) = toc;
    tic; train_svr(param_mv_svr_linear, X, y);
    time.mv_svr_linear(n) = toc;
    tic; train_svr(param_mv_svr_rbf, X, y);
    time.mv_svr_rbf(n) = toc;
    
    %% Time it - MATLAB -- regression
    param_mat_svr_linear    = {'KernelFunction' 'linear'};
    param_mat_svr_rbf       = {'KernelFunction' 'RBF'};
    
%     tic; ridge(y, X, 0.1);          % OUT OF MEMORY
%     time.mat_ridge(n) = toc;
    tic; fitrsvm(X, y, param_mat_svr_linear{:});
    time.mat_svr_linear(n) = toc;
    tic; fitrsvm(X, y, param_mat_svr_rbf{:});
    time.mat_svr_rbf(n) = toc;
    
end


save([resultsdir 'benchmarking_results_Haxby'],'time','n_voxel')

fprintf('Finished fMRI.\n')
end
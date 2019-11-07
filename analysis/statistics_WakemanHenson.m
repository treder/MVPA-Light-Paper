% Performs statistical analysis of the classification results in
% classify_WakemanHenson.m
%
clear
close all

% define directories
rootdir     = '/home/matthias/WakemanHensonMEEG/';
resultsdir  = [rootdir 'results/'];
figdir      = [rootdir 'figures/'];

% visualization settings
opt_title = {'FontSize' 14};

%% --- Level 1 (subject-level) statistics --- 
% Here we will perform level 1 statisical analyses on the first subject
% only. We will compare binomial test, permutation test, and cluster
% permutation test for the FAMOUS vs SCRAMBLED time x time generalization
% using the EEG data.

% Load classification results for subject 1
load([resultsdir 'classification_analysis_subject1'])

cfg = [];
cfg.classifier  = 'lda';
cfg.metric      = 'acc';
cfg.k           = 5;
cfg.repeat      = 1;

[perf, result] = mv_classify_timextime(cfg, eeg.trial, clabels{1,2});

%% binomial
cfg = [];
cfg.test    = 'binomial';
stat_binomial = mv_statistics(cfg, result);

mv_plot_result(result, erp_time, erp_time, 'mask', stat_binomial.mask);
title('Binomial test', opt_title{:})

figuresize(11,8,'centimeters')
saveas(gcf,[figdir 'level1_binomial.pdf'])

%% permutation
cfg.metric          = 'acc';
cfg.test            = 'permutation';
cfg.n_permutations  = 500;

stat_permutation = mv_statistics(cfg, result, eeg.trial, clabels{1,2});

mv_plot_result(result, erp_time, erp_time, 'mask', stat_permutation.mask)
title('Permutation test', opt_title{:})

figuresize(11,8,'centimeters')
saveas(gcf,[figdir 'level1_permutation.pdf'])

%% cluster permutation
cfg.test            = 'permutation';
cfg.correctm        = 'cluster';
cfg.clustercritval  = 0.55;

stat_cluster = mv_statistics(cfg, result, eeg.trial, clabels{1,2});

mv_plot_result(result, erp_time, erp_time, 'mask', stat_cluster.mask)
title('Cluster permutation', opt_title{:})

figuresize(11,8,'centimeters')
saveas(gcf,[figdir 'level1_cluster_permutation.pdf'])

%% save level 1 stats
% save([resultsdir 'statistics_subject1'], 'stat_binomial', 'stat_permutation', 'stat_cluster')

%%
load([resultsdir 'statistics_subject1'])

%% --- Level 2 (group-level) statistics using AUC values
% Load classification results for all subjects
load([resultsdir 'classification_analysis'])

freqs = 6:1:30;

%% Perform a cluster permutation test on the time-frequency classification
cond = 1;
results = result_freq(:,cond);

% we calculated multiple metrics but we only need AUC here
results = mv_select_result(results, 'auc');

% Let us first plot the average result
mean_result = mv_combine_results(results, 'average');
mv_plot_result(mean_result, freq_time, freqs)

cfg = [];
cfg.metric          = 'auc';
cfg.test            = 'permutation';
cfg.correctm        = 'cluster';
cfg.n_permutations  = 1000;
cfg.clustercritval  = 2.58;
% for alpha 0.05 = t-val is 1.96
% for alpha 0.01 = t-val is 2.58

% Level 2 stats settings
cfg.design          = 'within';
cfg.statistic       = 'ttest';   
cfg.null            = 0.5;

stat_cluster = mv_statistics(cfg, results);

% mask the mean AUC to only show the significant cluster
mv_plot_result(mean_result, freq_time, freqs, 'mask', stat_cluster.mask)
title('')

figuresize(9, 8,'centimeters')
saveas(gcf,[figdir sprintf('level2_cluster_permutation_cond%d.pdf', cond)])



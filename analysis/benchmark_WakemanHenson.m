% Performs benchmarking, i.e. comparing the speed of training different
% classifiers.
%
% Tested with:
% Dataset: Wakeman and Henson 1.0.3, downloaded from https://openneuro.org/datasets/ds000117/versions/1.0.3
% MATLAB: R2019a
% FieldTrip: revision r8588-7599-g94c95e995 (August 2019)
close all

% define directories
rootdir     = '/data/neuroimaging/WakemanHensonMEEG/';
preprocdir  = [rootdir 'preprocessed/'];
resultsdir  = [rootdir 'results/'];
figdir      = [rootdir 'figures/'];

nsubjects  = 16;

% Cell arrays for collecting all results across subjects
cf = cell(nsubjects, 2);   % keeps the classifier 
Xs = cell(nsubjects, 2);   
do_plot = 0;

classifiers = {'lda' 'logreg' 'naive_bayes' 'svm' 'kernel_fda' 'libsvm' ...
    'liblinear'};

meeg_time = cell(numel(classifiers), 1);
eeg_time = cell(numel(classifiers), 1);
meg_time = cell(numel(classifiers), 1);

%% TODO ---

for nn=1:nsubjects     % --- loop across subjects
    
    %% Load data
    fprintf('**** loading subject #%d\n', nn)
    load([preprocdir 'sbj-' num2str(nn)],'dat')
    
    % bring trials from cell array into matrix form
    cfg = [];
    cfg.keeptrials  = 'yes';
    cfg.covariance  = 'yes';
	meeg = ft_timelockanalysis(cfg, meeg);

    meeg = scale_MEEG(meeg);

    % MEG only
    cfg = [];
    cfg.channel = 'MEG';
	meg= ft_selectdata(cfg, meeg);
    
    % EEG only
    cfg = [];
    cfg.channel = 'EEG';
	eeg= ft_selectdata(cfg, meeg);
    
    %% Define class labels
    clabel = meg.trialinfo;
    % recode the classes by collapsing initial/immediates/delayed
    % triggers to one class such that
    % 1 = FAMOUS
    % 2 = UNFAMILIAR
    % 3 = SCRAMBLED
    clabel(ismember(clabel,[5,6,7])) = 1;
    clabel(ismember(clabel,[13,14,15])) = 2;
    clabel(ismember(clabel,[17,18,19])) = 3;
    
    %% also create labels for binary classification (two classes)
    % famous vs unfamiliar and famous vs scrambled faces
    ix_famous_scrambled = (ismember(clabel, [1, 3]));
    
    cfg = [];
    cfg.trials  = ix_famous_scrambled;
    cfg.latency = [-0.1, 0.9];
    meg_famous_scrambled = ft_selectdata(cfg, meg);
    
    clabel_famous_scrambled = clabel(ix_famous_scrambled);
    
    % for binary classification, recode scrambled (class 3) to class 2
    clabel_famous_scrambled(clabel_famous_scrambled==3) = 2;
    
    erp_time = meg_famous_scrambled.time;
    
    %% Cross-validation for N170 and sustained ERP component 400-800 ms
    N170_times = find( (meg.time > 0.15) & (meg.time < 0.2));
    sustained_times = find( (meg.time > 0.4) & (meg.time < 0.8));

    Xs{nn,1} = squeeze(mean(meg.trial(:,:,N170_times), 3));
    Xs{nn,2} = squeeze(mean(meg.trial(:,:,sustained_times), 3));
    
    % MVPA-Light
    cfg = [];
    cfg.classifier      = 'multiclass_lda';
    cfg.metric          = {'accuracy' 'confusion'};

    [cf_cv{nn,1}, result_N170] = mv_crossvalidate(cfg, Xs{nn,1}, clabel);
    [cf_cv{nn,2}, result_sus] = mv_crossvalidate(cfg, Xs{nn,2}, clabel);
    
    % call mv_plot_result for a quick visualisation of the results
    if do_plot
        mv_plot_result(result_N170)
        mv_plot_result(result_sus)
    end
    
    %%% Also keep the classifier trained on the full data on famouse vs
    %%% scrambled
    param = mv_get_hyperparameter('lda');
    cf{nn,1} = train_lda(param, Xs{nn,1}(ix_famous_scrambled,:), clabel_famous_scrambled);
    cf{nn,2} = train_lda(param, Xs{nn,2}(ix_famous_scrambled,:), clabel_famous_scrambled);

    %% Classify time x time [time generalisation]
    meg_famous_scrambled.trial = zscore(meg_famous_scrambled.trial);
    
    cfg = [];
    cfg.classifier      = 'lda';
    cfg.metric          = 'none';

    [cf_time{nn}, result] = mv_classify_across_time(cfg, meg_famous_scrambled.trial, clabel_famous_scrambled);

    if do_plot, mv_plot_result(result, meg_famous_scrambled.time, meg_famous_scrambled.time); end
    
    %% Perform time-frequency analysis using FieldTrip
    cfg              = [];
    cfg.output       = 'pow';
    cfg.method       = 'mtmconvol';
    cfg.taper        = 'hanning';
    cfg.keeptrials   = 'yes';
    cfg.foi          = 6:1:30;
    cfg.t_ftimwin    = 5./cfg.foi;  % 5 cycles per time window
    cfg.toi          = -0.2:0.02:1.1;
    
    ft_warning off
    freq = ft_freqanalysis(cfg, meg);
    
    % reduce to famous vs scrambled
    cfg = [];
    cfg.trials  = ix_famous_scrambled;
    cfg.latency = [-0.1, 1];
    freq_fs= ft_selectdata(cfg, freq);
    
    cfg.trials  = ix_famous_familiar;
    cfg.latency = [-0.1, 1];
    freq_ff= ft_selectdata(cfg, freq);

    freq_time = freq_fs.time;

    %% Classification for each time-frequency point separately
    % (the feature vector consists of the power at every MEG channel
    % for a given time-frequency point)
    cfg = [];
    cfg.classifier      = 'lda';
    cfg.metric          = 'auc';
    cfg.repeat          = 2;
    
    % We will use the function mv_classify for classification of the
    % time-frequency data. The function does not assume any specific order
    % of the data dimensions. Hence, the user needs to specify which
    % dimension codes for samples and which codes for the features. 
    cfg.sample_dimension = 1;
    cfg.feature_dimension  = 2;
    % Dimensions 3 (frequency) and 4 (time) are not specified and will
    % automatically be used as search/loop dimenions, so the result will be
    % a [frequencies x times] matrix of AUC values
    
    % optional: provide the names of the dimensions for nice output
    cfg.dimension_names = {'samples','channels','frequencies','time points'};
    
    % perform classification for famous vs scrambled and famous vs unfamiliar
    cf_freq{nn,1} = mv_classify(cfg, freq_fs.powspctrm, clabel_famous_scrambled);
    cf_freq{nn,2} = mv_classify(cfg, freq_ff.powspctrm, clabel_famous_familiar);
    
    % perf is now a 2-D [frequencies x times] matrix of AUC values
    if do_plot
        figure
        mv_plot_2D(cf_freq{nn,1}, 'x', freq_fs.time, 'y', freq_fs.freq)
        xlabel('Time'), ylabel('Frequency')
        title('AUC for famous vs scrambled faces at each time-frequency point')
        
        figure
        mv_plot_2D(cf_freq{nn,2}, 'x', freq_fs.time, 'y', freq_fs.freq)
        xlabel('Time'), ylabel('Frequency')
        title('AUC for famous vs unfamiliar faces at each time-frequency point')
    end
    
    %% Frequency generalization [freq x freq classification]
    % Here we will train on a specific frequency and test on another
    % frequency. To this end, all channels x all time points will serve as
    % features. 
    % The result is a frequency x frequency plot of classification
    % performance.

    % Select just subset of the time points
    cfg = [];
    cfg.latency         = [0.7, 1];
    freq_fs2 = ft_selectdata(cfg, freq_fs);

    % Z-score to bring all time-frequency points on equal footing
    freq_fs2.powspctrm = zscore(freq_fs2.powspctrm,[],1);
    
    cfg = [];
    cfg.classifier              = 'lda';
    cfg.metric                  = 'auc';
    cfg.dimension_names         = {'samples','channels','frequencies','time points'};
    
    % Samples are still coded in dimension 1
    cfg.sample_dimension  = 1;
    % Now we want to use both dimension 2 (channels) and dimension 4 (time
    % points) as features
    cfg.feature_dimension  = [2, 4];
    % Generalization is performed across dimension 3 (frequencies)
    cfg.generalization_dimension = 3;
    
    % to be a bit faster, use 10-fold cross-validation with no extra repetitions
    cfg.k               = 10;
    cfg.repeat          = 1;

    cf_freqxfreq{nn} = mv_classify(cfg, freq_fs2.powspctrm, clabel_famous_scrambled);
    
    if do_plot
        figure
        F = freq_fs2.freq;
        mv_plot_2D(cf_freqxfreq{nn}, 'x', F, 'y', F)
        xlabel('Test frequency [Hz]'), ylabel('Train frequency [Hz]')
        title('Frequency generalization using channels-x-times as features')
    end
    
    %% Perform analysis separately for EEG, MEG, and EEG+MEG
    cfg = [];
    cfg.trials  = ix_famous_scrambled;
    cfg.latency = [-0.1, 0.9];
    meeg = ft_selectdata(cfg, meeg);
    
    % Split into EEG and MEG channels
    cfg = [];
    cfg.channel  = 'EEG';
    eeg = ft_selectdata(cfg, meeg);

    cfg.channel  = 'MEG';
    meg = ft_selectdata(cfg, meeg);
    
    cfg = [];
    cfg.classifier      = 'lda';
    cfg.metric          = 'auc';
    
    % to be a bit faster, use 10-fold cross-validation with no extra repetitions
    cfg.k               = 10;
    cfg.repeat          = 1;

    [cf_meeg{nn,1}, result1] = mv_classify_across_time(cfg, eeg.trial, clabel_famous_scrambled);
    [cf_meeg{nn,2}, result2] = mv_classify_across_time(cfg, meg.trial, clabel_famous_scrambled);
    [cf_meeg{nn,3}, result3] = mv_classify_across_time(cfg, meeg.trial, clabel_famous_scrambled);

    % Plot results together
    if do_plot
        mv_plot_result({result1, result2, result3}, meeg.time)
        legend({'EEG', 'MEG', 'EEG+MEG'})
    end
end

save([resultsdir 'classification_analysis'],'nsubjects',...
    'erp_time','freq_time','F', 'Xs', 'cf', 'cf_*')

fprintf('Finished all.\n')
return

%% --- CALCULATE GRAND AVERAGE AND PLOT RESULTS --
load([resultsdir 'classification_analysis'])
err = sqrt(nsubjects);   % normalisation for standard error

acc = cellfun(@(c) c{1}, cf_cv);
confusion = cellfun(@(c) c{2}, cf_cv, 'UniformOutput', false);

% For each classification analysis, calculate grand average
% and standard error across subjects
av_acc = mean(acc,1);
se_acc = std(acc) / err;
av_confusion = cat(3, mean(cat(3, confusion{:,1}),3), mean(cat(3, confusion{:,2}),3) );
se_confusion = cat(3, std(cat(3, confusion{:,1}),[],3)/err, std(cat(3, confusion{:,2}),[],3)/err );
av_time = mean(cat(3, cf_time{:}),3);
se_time = std(cat(3, cf_time{:}),[],3)/err;
av_freq= cat(3, mean(cat(3, cf_freq{:,1}),3), mean(cat(3, cf_freq{:,2}),3) );
se_freq= cat(3, std(cat(3, cf_freq{:,1}),[],3), std(cat(3, cf_freq{:,2}),[],3))/err;
av_freqxfreq= mean(cat(3, cf_freqxfreq{:}),3);
se_freqxfreq= std(cat(3, cf_freqxfreq{:}),[],3)/err;
av_meeg= cat(2, mean(cat(2, cf_meeg{:,1}),2), mean(cat(2, cf_meeg{:,2}),2), mean(cat(2, cf_meeg{:,3}),2) );
se_meeg= cat(2, std(cat(2, cf_meeg{:,1}),[],2), std(cat(2, cf_meeg{:,2}),[],2), std(cat(2, cf_meeg{:,3}),[],2) )/err;

%% PLOT
ylab_opt = {'Fontweight' 'bold' };
xlab_opt = ylab_opt;
title_opt = {'Fontweight' 'bold' 'Fontsize' 12};

erp_labels = {'N170' 'Sustained ERP'};

%% plot cross-validation: accuracy
figure
% mark chance level
plot([-0.2000    3.2000], [1 1]*0.33, '--', 'LineWidth', 2,'Color',[1,1,1]*0.5)
text(2.5, 0.33, sprintf('Chance\nlevel'),'Color',[1,1,1]*0.5)
hold on, grid on
bar(1:2, av_acc);
set(gca,'XTick',1:2, 'XTickLabel', erp_labels)
errorbar(1:2, av_acc, se_acc,'.','Color','k','LineWidth',2)
ylabel('Accuracy', ylab_opt{:})
title('Classification accuracy [3 classes]', title_opt{:})

figuresize(12,8,'centimeters')
saveas(gcf,[figdir 'crossval_accuracy.pdf'])

%% plot cross-validation: confusion
classes = {'Famous' 'Unfamiliar' 'Scrambled'};
figure
nclasses = 3;
for ii=1:2
    subplot(1,2,ii)
    av = av_confusion(:,:,ii);
    imagesc(av)
    colorbar
    xlabel('Predicted class',xlab_opt{:});
    if ii==1, ylabel('True class',ylab_opt{:}); end
    set(gca,'Xtick',1:nclasses,'Ytick',1:nclasses,'XTickLabel',classes,'YTickLabel',classes)
    for rr=1:nclasses
        for cc=1:nclasses
            text(cc,rr, sprintf('%0.2f',av(rr,cc)), opt_txt{:})
        end
    end
    title([erp_labels{ii} ' [confusion]'])
end

figuresize(24,8,'centimeters')
saveas(gcf,[figdir 'crossval_confusion.pdf'])

%% plot time generalization
figure

h = mv_plot_2D(av_time, 'x', erp_time, 'y', erp_time)
set(get(h.colorbar,'label'),'String','AUC','FontSize',12)
title('Time generalization (ERF)', title_opt{:})

figuresize(12,8,'centimeters')
saveas(gcf,[figdir 'timextime.pdf'])

%% plot freq classification
figure,%colormap(redblue)

titles = {'Famous vs scrambled' 'Famous vs unfamiliar'};
for ii=1:2
    subplot(1,2,ii)
    av = av_freq(:,:,ii);
    h = mv_plot_2D(av, 'x', freq_time, 'y', F)
    xlabel('Time',xlab_opt{:}), ylabel('Frequency',ylab_opt{:})
    title(titles{ii}, title_opt{:})
end
ylabel('')
set(get(h.colorbar,'label'),'String','AUC','FontSize',12)

figuresize(22,8,'centimeters')
saveas(gcf,[figdir 'freq.pdf'])

%% freq x freq generalization
figure,%colormap(redblue)

h = mv_plot_2D(av_freqxfreq, 'x', F, 'y', F)
xlabel('Testing frequency', xlab_opt{:}), ylabel('Training frequency', ylab_opt{:})

title('Frequency generalization', title_opt{:})

set(get(h.colorbar,'label'),'String','AUC','FontSize',12)

figuresize(12,8,'centimeters')
saveas(gcf,[figdir 'freqxfreq.pdf'])

%% EEG vs MEG vs EEG/MEG
figure,%colormap(redblue)

leg = {'EEG' 'MEG' 'EEG+MEG'};

h = mv_plot_1D(erp_time, av_meeg, se_meeg)
legend(leg, 'Location', 'SouthEast')
xlabel('Time',xlab_opt{:}), ylabel('AUC',ylab_opt{:})
title('AUC for different channel sets', title_opt{:})

figuresize(12,8,'centimeters')
saveas(gcf,[figdir 'EEG_vs_MEG.pdf'])

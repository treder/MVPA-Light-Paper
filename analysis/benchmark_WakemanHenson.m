% Performs benchmarking on Wakeman & Henson data:
% - out of the box classification performance (using classifiers with their
%   default settings)
% - computational speed: how long does it take to train the classifier 
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

% Load class labels (saved in classify_WakemanHenson.m)
load([resultsdir 'classification_analysis'],'nsubjects','clabels')

classifiers_light = {'lda' 'logreg' 'naive_bayes' 'svm' 'kernel_fda' 'libsvm' ...
    'liblinear'};

classifiers_matlab = {'lda' 'logreg' 'naive_bayes' 'svm' 'kernel_fda' 'libsvm' ...
    'liblinear'};

% meeg_accuracy = 
meeg_acc = cell(numel(classifiers), 1);
meeg_time = cell(numel(classifiers), 1);

for n=1:nsubjects     % --- loop across subjects
    
    
    %% Load MEEG data
    fprintf('**** loading subject #%d\n', n)
    load([preprocdir 'sbj-' num2str(n)],'dat')
    
    % bring trials from cell array into matrix form
    cfg = [];
    cfg.keeptrials  = 'yes';
    cfg.covariance  = 'yes';
	dat = ft_timelockanalysis(cfg, dat);

    dat = scale_MEEG(dat);
    
    % MEG
    cfg = [];
    cfg.channel = 'MEG';
	meg= ft_selectdata(cfg, dat);
    
    %% Out of the box classification
    
    
    %% Timing 
   
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

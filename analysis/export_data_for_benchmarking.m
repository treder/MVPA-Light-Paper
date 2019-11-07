% Exports Wakeman & Henson and Haxby et al. data for the benchmarking
% analysis. In particular
% - cell array ...
% - create "super-subject" by concatenating all trials. This produces a
%   N>>P dataset where trials outnumber features
% - export full brain MRI voxels. This produces a P>>N dataset where 
%   features vastly outnumber trials
close all

% define directories
wakeman_rootdir     = '/data/neuroimaging/WakemanHensonMEEG/';
wakeman_preprocdir  = [wakeman_rootdir 'preprocessed/'];
wakeman_resultsdir  = [wakeman_rootdir 'results/'];

haxby_rootdir     = '/data/neuroimaging/Haxby2001_fMRI/';
haxby_resultsdir  = [haxby_rootdir 'results/'];

% Load class labels (saved in classify_WakemanHenson.m)
load([resultsdir 'classification_analysis'],'nsubjects','clabels')

%% Wakeman & Henson data
meg = cell(16, 1);
for n=1:nsubjects     % --- loop across subjects
    
    
    %% Load MEEG data
    fprintf('**** loading subject #%d\n', n)
    load([wakeman_preprocdir 'sbj-' num2str(n)],'dat')
    
    % bring trials from cell array into matrix form
    cfg = [];
    cfg.keeptrials  = 'yes';
    cfg.covariance  = 'yes';
	dat = ft_timelockanalysis(cfg, dat);

    dat = scale_MEEG(dat);
    
    % MEG
    cfg = [];
    cfg.channel = 'MEG';
	meg{n}= ft_selectdata(cfg, dat);
   
end

% 

save([wakeman_resultsdir 'benchmarking_data_single_subjects'],'nsubjects','meg','clabels')

%% Create super-subject ( N>>P data)
M = ft_appenddata([], meg{:});
save([wakeman_resultsdir 'benchmarking_data_supersubject'],'meg','clabels')

%% Haxby et al. data
fmri = cell(6, 1);
clabel = cell(6, 1);
for n=1:6     % --- loop across subjects
   
    % Load BOLD time series
    tmp = ft_read_mri([haxby_rootdir sprintf('subj%d', n) '/bold.nii']);
    tmp.anatomy    = double(tmp.anatomy);
    tmp.bold       = reshape(tmp.anatomy, prod(tmp.dim(1:3)), []);

%% Highpass filter >0.01 Hz
    tr = 2.5;
    Fs =  1/tr; 
    freq = 0.01;
    dir = 'twopass';
    tmp.bold = ft_preproc_highpassfilter(tmp.anatomy, Fs, freq, [], [], dir);
    
     %% Load labels and select classes
    y = readtable([sbj_dir 'labels.txt']);

    classes = {'face' 'house'};

    ix = false(length(y.labels),1);
    for c=1:numel(classes)
        ix = ix | strcmp(y.labels, classes{c});
    end
    
    %% Build clabel
    y = y(ix, :);
    y = y.labels;
    % RECODING class labels as numbers
    for c=1:numel(classes)
        y(ismember(y, classes{c})) = {num2str(c)};
    end
    y = cellfun( @str2num, y);
    
    %% Extract time indices corresponding to the selected classes
    tmp.bold = tmp.bold(:, ix);
    
    %% save data
    fmri{n} = tmp;
    clabel{n} = y;
end

save([haxby_resultsdir 'benchmarking_data'],'meg','clabels')

fprintf('Finished all.\n')


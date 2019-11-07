% Preprocess and classify Haxby 2001 fMRI data.
%
% Tested with:
% DATASET: Haxby et al. (2001): Faces and Objects in Ventral Temporal Cortex (fMRI)
% downloaded from http://www.pymvpa.org/datadb/haxby2001.html
% MATLAB: R2019a
% FIELDTRIP: revision r8588-7599-g94c95e995 (August 2019)
close all
clear

% define directories
rootdir     = '/data/neuroimaging/Haxby2001_fMRI/';
preprocdir  = [rootdir 'preprocessed/'];
figdir      = [rootdir 'figures/'];
resultsdir  = [rootdir 'results/'];

%% save the result structs and the neighbours matrix for each subject
nsubjects  = 6;

res_confusion = cell(1,6);
res_time = cell(6,3);
acc_searchlight = cell(1,6);
res_searchlight = cell(1,6);
nb = cell(1,6);
clabel_face_house = cell(1,6);
face_house_bold = cell(1,6);

%% create brain mask
% see end of this script for how the mask was created
brain_mask = zeros(40, 64, 64);
mask_x = 3:38;
mask_y = 10:58;
mask_z = 15:55;
brain_mask(mask_x, mask_y, mask_z) = 1;
brain_mask = logical(brain_mask);
fprintf('Fraction voxels selected for brain mask: %2.3f\n', sum(brain_mask(:))/numel(brain_mask))


%%
for n=1:nsubjects     % --- loop across subjects
    
    fprintf('\n***************************\n*** Processing subject #%d\n***************************\n', n)
    sbj_label = sprintf('subj%d', n);
    sbj_dir = [rootdir  sbj_label '/'];
   
    % Load BOLD time series
    fmri = ft_read_mri([sbj_dir 'bold.nii']);
    fmri.anatomy = double(fmri.anatomy);
    
    n_time = size(fmri.anatomy, 4);
    fmri.bold   = reshape(fmri.anatomy, prod(fmri.dim(1:3)), []);
    
    % Load mask
    mask_vt = ft_read_mri([sbj_dir 'mask4_vt.nii']);
    mask_face = ft_read_mri([sbj_dir 'mask8_face_vt.nii']);
    mask_house = ft_read_mri([sbj_dir 'mask8_house_vt.nii']);

    % Apply mask to get BOLD time series for ROIs
    fmri.vt        = fmri.anatomy(logical(repmat(mask_vt.anatomy, [1 1 1 n_time])));
    fmri.face_roi  = fmri.anatomy(logical(repmat(mask_face.anatomy, [1 1 1 n_time])));
    fmri.house_roi = fmri.anatomy(logical(repmat(mask_house.anatomy, [1 1 1 n_time])));

    % Reshape back into 2d
    fmri.vt        = reshape(fmri.vt, sum(mask_vt.anatomy(:)), []);
    fmri.face_roi  = reshape(fmri.face_roi, sum(mask_face.anatomy(:)), []);
    fmri.house_roi = reshape(fmri.house_roi, sum(mask_house.anatomy(:)), []);

    %% Highpass filter >0.01 Hz
    tr = 2.5;
    Fs =  1/tr; 
    freq = 0.01;
    dir = 'twopass';
    fmri.bold_hp = ft_preproc_highpassfilter(fmri.bold, Fs, freq, [], [], dir);
    fmri.face_hp = ft_preproc_highpassfilter(fmri.face_roi, Fs, freq, [], [], dir);
    fmri.house_hp = ft_preproc_highpassfilter(fmri.house_roi, Fs, freq, [], [], dir);
    fmri.vt_hp = ft_preproc_highpassfilter(fmri.vt, Fs, freq, [], [], dir);

    %% Load labels and select classes
    y = readtable([sbj_dir 'labels.txt']);

    classes = {'face' 'cat' 'house' 'bottle' 'scissors' 'shoe' 'chair' 'scrambledpix'};

    ix = false(length(y.labels),1);
    for c=1:numel(classes)
        ix = ix | strcmp(y.labels, classes{c});
    end

    %% Extract time indices corresponding to the selected classes
    fmri.bold = fmri.bold(:, ix);
    fmri.bold_hp = fmri.bold_hp(:, ix);

    fmri.vt = fmri.vt(:, ix);
    fmri.house_roi= fmri.house_roi(:, ix);
    fmri.face_roi = fmri.face_roi(:, ix);
    
    % high-pass filtered version
    fmri.vt_hp = fmri.vt_hp(:, ix);
    fmri.house_hp= fmri.house_hp(:, ix);
    fmri.face_hp = fmri.face_hp(:, ix);
    
    %% Define class labels
    y = y(ix, :);
    
    clabel = y.labels;
    % RECODING class labels as numbers:
    for c=1:numel(classes)
        clabel(ismember(clabel, classes{c})) = {num2str(c)};
    end
    clabel = cellfun( @str2num, clabel);
    
    % After data is reshaped into 3d trials x voxels x times, we must pick
    % only every 9th element from clabel
    clabel9 = clabel(1:9:end);

    
    %% Reshape to samples x voxels
   
    % use hp filtered data
    fmri.trial = reshape(fmri.bold_hp, [], 9, numel(clabel)/9);
    fmri.trial_vt = reshape(fmri.vt_hp, [], 9, numel(clabel)/9);
    fmri.trial_face = reshape(fmri.face_hp, [], 9, numel(clabel)/9);
    fmri.trial_house = reshape(fmri.house_hp, [], 9, numel(clabel)/9);
    
    fmri.trial = permute(fmri.trial, [3 1 2]);
    fmri.trial_vt = permute(fmri.trial_vt, [3 1 2]);
    fmri.trial_face = permute(fmri.trial_face, [3 1 2]);
    fmri.trial_house = permute(fmri.trial_house, [3 1 2]);

    %% Classify / confusion matrix
    % Instead of randomized k-fold cross-validation, we will exploit the
    % fact that the data has been separated into 12 runs, so we will treat
    % each run as a separate fold. This can be done by setting
    % cv='leavegroupout'
    cfg = [];
    cfg.cv      = 'leavegroupout';
    cfg.group   = y.chunks;
    cfg.repeat  = 1;
    cfg.metric  = 'confusion';
    cfg.classifier  = 'multiclass_lda';
    
    [acc_vt_hp, res_confusion{n}] = mv_crossvalidate(cfg, fmri.vt_hp', clabel);
    
    %% Classify across time
    cfg.group   = y.chunks(1:9:end);
    cfg.metric  = 'accuracy';
    [acc_time_face_roi, res_time{n,1}]  = mv_classify_across_time(cfg, fmri.trial_face, clabel9);
    [acc_time_house_roi, res_time{n,2}] = mv_classify_across_time(cfg, fmri.trial_house, clabel9);
    [acc_time_vt_roi, res_time{n,3}]    = mv_classify_across_time(cfg, fmri.trial_vt, clabel9);
    
    %% --- Searchlight classification ---
    % focus on faces vs houses here
    ix_face_house = ismember(clabel, [1 3]); % 1-face, 3-house
    y_face_house = y(ix_face_house, :);
    
    cl = clabel(ix_face_house);
    cl(cl==3) = 2;  % recode houses as 2
    clabel_face_house{n} = cl;
    
    %% apply brain mask
    
%     brain_voxels = find(brain_mask(:));
%     face_house_bold{n} = fmri.bold_hp(brain_voxels, ix)';

%     %% for searchlight, we need to define which voxels are neighbours of
%     % each other. For this we need to translate the neighbourhood in a 3D
%     % volume into a sparse voxel x voxel neighbourhood matrix. We need the
%     % dimensions of the brain_mask/cube:
%     vox_dim = [36, 49, 41];
%     n_voxel = prod(vox_dim);
%     neighbours = sparse(n_voxel,n_voxel);
%     
%     r = 1;  % "radius" of cube
%     for vox_ix=1:n_voxel
%         if mod(vox_ix,10000)==0, fprintf('%d\n',vox_ix),end
%         % Initialise volumn of 0's
%         vol = zeros(vox_dim);
%         % find center voxel coordinates corresponding to current index
%         [x1,x2,x3] = ind2sub(vox_dim, vox_ix);
%         % Set all voxels in neighbourhood to 1
%         vol(max(1,x1-r):min(vox_dim(1),x1+r), ...
%             max(1,x2-r):min(vox_dim(2),x2+r), ...
%             max(1,x3-r):min(vox_dim(3),x3+r)) = 1;
%         % Extract linear index of these voxels and set the neighbours to 1
%         neighbours_ix = find(vol==1);
%         neighbours(vox_ix, neighbours_ix) = 1;
%         neighbours(neighbours_ix, vox_ix) = 1; 
%     end
%     nb{n} = neighbours;
    
    %% Back to 3D voxels
    face_house_bold{n} = fmri.bold_hp(:, ix_face_house);
    face_house_bold{n} = reshape(face_house_bold{n}, [fmri.dim(1:3), numel(clabel_face_house{n})]);

    % Create neighbours for x, y, z axis: each voxel and its direct
    % neighbour
    neighbours= cell(1,3);
    for i=1:3
        O = ones(fmri.dim(i));
        neighbours{i} = O - triu(O,2) - tril(O,-2);
    end
    
    %% Searchlight classification
    cfg = [];
    cfg.classifier  = 'lda';
    cfg.metric      = 'accuracy';
    cfg.feedback    = 1;

    % cross-validation settings
    cfg.cv          = 'leavegroupout';
    cfg.group       = y_face_house.chunks;
    cfg.repeat      = 1;

    cfg.dimension_names     = {'voxel x' 'voxel y' 'voxel z' 'samples'};
    cfg.sample_dimension    = 4;    % dimension 1 in the data represents the samples/different time points
    cfg.neighbours          = neighbours;
    
    tic
    [acc_searchlight{n}, res_searchlight{n}] = mv_classify(cfg, face_house_bold{n}, clabel_face_house{n});
    toc
    
%     cfg = [];
%     cfg.classifier  = 'lda';
%     cfg.metric      = 'accuracy';
%     cfg.feedback    = 1;
% 
%     % cross-validation settings
%     cfg.cv          = 'leavegroupout';
%     cfg.group       = y_face_house.chunks;
%     cfg.repeat      = 1;
% 
%     cfg.dimension_names     = {'samples' 'voxels'};
%     cfg.sample_dimension    = 1;    % dimension 1 in the data represents the samples/different time points
%     cfg.neighbours          = neighbours;
%     
%     tic
%     [acc_searchlight{n}, res_searchlight{n}] = mv_classify(cfg, face_house_bold{n}, clabel_face_house{n});
%     toc
end

save([resultsdir 'Haxby_fmri_classification_results'], 'res*', 'nb','acc_searchlight', ...
    'nsubjects', 'classes', 'face_house_bold', 'clabel_face_house','vox_dim')

fprintf('Finished all.\n')
return

%% Load data
load([resultsdir 'Haxby_fmri_classification_results'])

%% Reshape searchlight data back into 3D
% for n=1:6
%     res_searchlight{n}.perf = reshape(res_searchlight{n}.perf, vox_dim);
%     res_searchlight{n}.perf_std = reshape(res_searchlight{n}.perf_std, vox_dim);
% end
% 
% mean_searchlight = mv_combine_results(res_searchlight, 'average');

%% Perform a cluster permutation test on the searchlight analysis
n=1;

cfg = [];
cfg.test            = 'permutation';
cfg.correctm        = 'cluster';
cfg.n_permutations  = 100;
cfg.clustercritval  = 0.6;

stat_cluster = mv_statistics(cfg, res_searchlight{n}, face_house_bold{n}, clabel_face_house{n});

fprintf('Found %d significant clusters\n', sum(stat_cluster.p < 0.05))

save([resultsdir 'Haxby_stat_cluster'], 'stat_cluster')


%% Calculate mean anatomical image for displaying results
anat = cell(1,5);
for n=1:numel(anat)
    tmp = ft_read_mri([rootdir sprintf('subj%d', n) '/anat.nii']);
    anat{n} = double(tmp.anatomy);
end

anat = mean(cat(4, anat{:}),4);

%% Plot cluster permutation test results
n =  1;
sbj_label = sprintf('subj%d', n);
sbj_dir = [rootdir  sbj_label '/'];
% Load one fMRI
fmri = ft_read_mri([sbj_dir 'bold.nii']);
fmri.anatomy = double(fmri.anatomy);
fmri.anatomy = squeeze(mean(fmri.anatomy,4));  % Fake anatomy as mean fmri signal
fmri.dim(4) = [];

% Get performance measure
fmri.perf = mean_searchlight.perf;

% Load MRI
mri.anatomy = anat;

% Interpolate performance measure
cfg = [];
cfg.parameter = 'perf';
interp = ft_sourceinterpolate(cfg, fmri, mri)

% Interpolate cluster permutation mask
fmri.mask = stat_cluster.mask;
cfg.parameter = 'mask';
tmp = ft_sourceinterpolate(cfg, fmri, mri)
interp.mask = tmp.mask;

% Plot everything
cfg = [];
cfg.anaparameter    = 'anatomy';
cfg.funparameter    = 'perf';
cfg.maskparameter   = 'mask';
cfg.method          = 'slice';
ft_sourceplot(cfg, interp)


%% Plot searchlight results
n =  1;
sbj_label = sprintf('subj%d', n);
sbj_dir = [rootdir  sbj_label '/'];

% Load fMRI
fmri = ft_read_mri([sbj_dir 'bold.nii']);
fmri.anatomy = double(fmri.anatomy);
fmri.anatomy = squeeze(mean(fmri.anatomy,4));  % Fake anatomy as mean fmri signal
fmri.dim(4) = [];

% Get performance measure
fmri.perf = reshape(res_searchlight{1}.perf, fmri.dim);
% fmri.mask = fmri.perf > 0.6;

% threshold everything
% fmri.perf(fmri.perf < 0.9) = 0;

cfg = [];
cfg.anaparameter    = 'anatomy';
cfg.funparameter    = 'perf';
cfg.maskparameter   = 'mask';
cfg.method          = 'slice';
% ft_sourceplot(cfg, fmri)


% saveas(gcf,sprintf('%s/subj%d_anatomy.pdf', figdir, n))

% Load MRI
mri = ft_read_mri([sbj_dir 'anat.nii']);

cfg = [];
cfg.parameter = 'perf';
interp = ft_sourceinterpolate(cfg, fmri, mri)

% Create mask on interpolated data
interp.mask = interp.perf > 0.8;

cfg = [];
cfg.anaparameter    = 'anatomy';
cfg.funparameter    = 'perf';
cfg.maskparameter   = 'mask';
cfg.method          = 'slice';
ft_sourceplot(cfg, interp)

figuresize(16,16,'centimeters')
saveas(gcf, [figdir 'fmri_searchlight.pdf'])


%% Calculate averages
mean_time = cell(1,3);

mean_confusion = mv_combine_results(res_confusion,'average');
for i=1:3
    mean_time{i} = mv_combine_results(res_time(:,i),'average');
end
mean_searchlight = mv_combine_results(res_searchlight,'average');

% confusion matrix
classes{end} = 'scramble';
mv_plot_result(mean_confusion)
title('Ventral temporal area')
set(gca,'XTick', 1:numel(classes), 'XTickLabel', classes, 'XTickLabelRotation', 35, 'YTick', 1:numel(classes), 'YTickLabel', classes)
colormap summer
grid off
figuresize(16,12,'centimeters')
saveas(gcf, [figdir 'fmri_mean_confusion.pdf'])

% merge results for the three times
x = 0:2.5:8*2.5;
grand_mean_time = mv_combine_results(mean_time,'merge');
grand_mean_time.plot{1}.legend_labels = {'Face ROI' 'House ROI' 'Ventral temporal'};
grand_mean_time.plot{1}.xlabel = 'Time [s]';
grand_mean_time.plot{1}.ylabel = 'Accuracy';
mv_plot_result(grand_mean_time, x)
figuresize(16,12,'centimeters')
title('Classification performance across time')
saveas(gcf, [figdir 'fmri_mean_time.pdf'])


%% Calculate mean fMRI image to find a tight bounding box around the brain
for n=1:nsubjects
    sbj_label = sprintf('subj%d', n);
    sbj_dir = [rootdir  sbj_label '/'];
    
    % Load BOLD time series
    if n==1
        fmri = ft_read_mri([sbj_dir 'bold.nii']);
        fmri.anatomy = mean(double(fmri.anatomy),4);
    else
        tmp = ft_read_mri([sbj_dir 'bold.nii']);
        fmri.anatomy = fmri.anatomy + mean(double(tmp.anatomy),4);
    end
end

fmri.anatomy    = fmri.anatomy / nsubjects;
fmri.dim(4)     = [];

% Plot the mask and find the bouding box
cfg = [];
cfg.anaparameter ='anatomy';
ft_sourceplot(cfg, fmri)

% -> this is the result
% determine the brain mask that corresponds to active gray matter
% The cubical brain mask has the voxels
% x = [3  : 38]
% y = [10 : 58]
% z = [15 : 55]
brain_mask = zeros(size(fmri.anatomy));
brain_mask(3:38, 10:58, 15:55) = 1;
brain_mask = logical(brain_mask);
fprintf('Fraction selected: %2.3f\n', sum(brain_mask(:))/numel(brain_mask))




%% OLD (combining all subjects for searchlight which is incorrect since they're not aligned)
% anat = cell(1,5);
% for n=1:numel(anat)
%     tmp = ft_read_mri([rootdir sprintf('subj%d', n) '/anat.nii']);
%     anat{n} = double(tmp.anatomy);
% end
% 
% anat = mean(cat(4, anat{:}),4);
% 
% %% Plot cluster permutation test results
% n =  1;
% sbj_label = sprintf('subj%d', n);
% sbj_dir = [rootdir  sbj_label '/'];
% % Load one fMRI
% fmri = ft_read_mri([sbj_dir 'bold.nii']);
% fmri.anatomy = double(fmri.anatomy);
% fmri.anatomy = squeeze(mean(fmri.anatomy,4));  % Fake anatomy as mean fmri signal
% fmri.dim(4) = [];
% 
% % Get performance measure
% fmri.perf = mean_searchlight.perf;
% 
% % Load MRI
% mri.anatomy = anat;
% 
% % Interpolate performance measure
% cfg = [];
% cfg.parameter = 'perf';
% interp = ft_sourceinterpolate(cfg, fmri, mri)
% 
% % Interpolate cluster permutation mask
% fmri.mask = stat_cluster.mask;
% cfg.parameter = 'mask';
% tmp = ft_sourceinterpolate(cfg, fmri, mri)
% interp.mask = tmp.mask;
% 
% % Plot everything
% cfg = [];
% cfg.anaparameter    = 'anatomy';
% cfg.funparameter    = 'perf';
% cfg.maskparameter   = 'mask';
% cfg.method          = 'slice';
% ft_sourceplot(cfg, interp)
% 
% 
% %% Plot searchlight results
% n =  1;
% sbj_label = sprintf('subj%d', n);
% sbj_dir = [rootdir  sbj_label '/'];
% 
% % Load fMRI
% fmri = ft_read_mri([sbj_dir 'bold.nii']);
% fmri.anatomy = double(fmri.anatomy);
% fmri.anatomy = squeeze(mean(fmri.anatomy,4));  % Fake anatomy as mean fmri signal
% fmri.dim(4) = [];
% 
% % Get performance measure
% fmri.perf = reshape(res_searchlight{1}.perf, fmri.dim);
% % fmri.mask = fmri.perf > 0.6;
% 
% % threshold everything
% % fmri.perf(fmri.perf < 0.9) = 0;
% 
% cfg = [];
% cfg.anaparameter    = 'anatomy';
% cfg.funparameter    = 'perf';
% cfg.maskparameter   = 'mask';
% cfg.method          = 'slice';
% % ft_sourceplot(cfg, fmri)
% 
% 
% % saveas(gcf,sprintf('%s/subj%d_anatomy.pdf', figdir, n))
% 
% % Load MRI
% mri = ft_read_mri([sbj_dir 'anat.nii']);
% 
% cfg = [];
% cfg.parameter = 'perf';
% interp = ft_sourceinterpolate(cfg, fmri, mri)
% 
% % Create mask on interpolated data
% interp.mask = interp.perf > 0.8;
% 
% cfg = [];
% cfg.anaparameter    = 'anatomy';
% cfg.funparameter    = 'perf';
% cfg.maskparameter   = 'mask';
% cfg.method          = 'slice';
% ft_sourceplot(cfg, interp)
% 
% figuresize(16,16,'centimeters')
% saveas(gcf, [figdir 'fmri_searchlight.pdf'])
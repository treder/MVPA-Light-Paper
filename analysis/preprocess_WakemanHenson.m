% Uses FieldTrip to preprocess the Wakeman and Henson data prior to
% classification analysis
%
% Tested with:
% Dataset: Wakeman and Henson 1.0.3, downloaded from https://openneuro.org/datasets/ds000117/versions/1.0.3
% MATLAB: R2019a
% FieldTrip: revision r8588-7599-g94c95e995 (August 2019)

% define directories
rootdir     = '/data/neuroimaging/WakemanHensonMEEG/';
rawdir      = [rootdir 'derivatives/meg_derivatives/'];
preprocdir  = [rootdir 'preprocessed/'];
figdir      = [rootdir 'figures/'];

nsubjects  = 16;
nruns      = 6;

diary preprocess_log

%%
for nn=1:nsubjects     % --- loop across subjects
    fprintf('\n***************************\n*** Processing subject #%d\n***************************\n', nn)
    
    nr = num2str(nn,'%02d'); % subject number
    sbjdir = [rawdir 'sub-' nr '/ses-meg/meg/'];

    dat = cell(nruns, 1);
    trl  = cell(1, nruns);
    
    %% Loop across 6 runs
    for run=1:nruns
        filename = [sbjdir 'sub-' nr '_ses-meg_task-facerecognition_run-' num2str(run,'%02d') '_proc-sss_meg.fif'];
        
        %% Load header
        hdr= ft_read_header(filename);
        
        %% Load continuous data
        %  here just load the data since we want to downsample first
        %  before filtering
        cfg = [];
        cfg.continuous              = 'yes';
        cfg.dataset                 = filename;

        % EEG061=HEOG, EEG062=VEOG, EEG063=ECG, so remove these channels
        cfg.channel                 = {'*EG*' '-EEG061' '-EEG062' '-EEG063'};
        dat{run}                     = ft_preprocessing(cfg);
       
        %% Preprocess and filter
        cfg = [];
        cfg.lpfilter            = 'yes';
        cfg.lpfreq              = 100;
        
        cfg.hpfilter            = 'yes';
        cfg.hpinstabilityfix    = 'reduce';
        cfg.hpfilttype          = 'firws';
        cfg.hpfiltdir           = 'onepass-zerophase';
        cfg.hpfreq              = 0.1;
        
        cfg.bsfilter            = 'yes';
        cfg.bsfreq              = [49 51];

        dat{run}                = ft_preprocessing(cfg, dat{run});
        
        %% Load trial information
        
        % Trigger definitions
        % (https://openneuro.org/datasets/ds000117/versions/1.0.3)
        %         
        % 5         Initial Famous Face               FAMOUS
        % 6         Immediate Repeat Famous Face      FAMOUS
        % 7         Delayed Repeat Famous Face        FAMOUS
        % 13        Initial Unfamiliar Face           UNFAMILIAR
        % 14        Immediate Repeat Unfamiliar Face  UNFAMILIAR
        % 15        Delayed Repeat Unfamiliar Face    UNFAMILIAR
        % 17        Initial Scrambled Face            SCRAMBLED
        % 18        Immediate Repeat Scrambled Face   SCRAMBLED
        % 19        Delayed Repeat Scrambled Face     SCRAMBLED
        
        cfg= [];
        cfg.dataset= filename;
        cfg.trialfun            = 'ft_trialfun_general';
        cfg.trialdef.eventtype  = 'STI101';
        cfg.trialdef.eventvalue = [5,6,7, 13,14,15, 17,18,19];
        cfg.trialdef.prestim    = 1;
        cfg.trialdef.poststim   = 1.5;

        tmp = ft_definetrial(cfg);
        
        % apply this trial definition to the data
        cfg = [];
        cfg.trl = tmp.trl;
        dat{run} = ft_redefinetrial(cfg, dat{run});
        
        trl{run} = tmp.trl;
        
%         ev{run}= tmp.event';
%         ev{run}= ev{run}(([ev{run}.value]<20));
%         
%         tmp=[];
%         
%         fprintf('[Run %d] %d stimulus triggers\n', run, numel(ev{run}))
%         
%         % Downsample triggers: the triggers are given wrt the original
%         % sampling frequency, we need to adapt the manually!
%         fprintf('resampling triggers\n')
%         for ii=1:numel(ev{run})
%             ev{run}(ii).sample= round(ev{run}(ii).sample * dat{run}.fsample / hdr.Fs);
%         end
        
        %% Downsample to 220 Hz
        cfg= [];
        cfg.resamplefs          = 220;
        cfg.demean              = 'yes';
        
        dat{run} = ft_resampledata(cfg, dat{run});

        
        %% Load and preprocess data
        % doing filtering after downsampling is faster
%         cfg = [];
%         
%         cfg.dataset             = filename;
%         cfg.trl                 = tmp.trl;
%         % EEG061=HEOG, EEG062=VEOG, EEG063=ECG, so remove these channels
%         cfg.channel             = {'*EG*' '-EEG061' '-EEG062' '-EEG063'};
% 
%         cfg.lpfilter            = 'yes';
%         cfg.lpfreq              = 100;
%         
%         cfg.hpfilter            = 'yes';
%         cfg.hpinstabilityfix    = 'reduce';
%         cfg.hpfilttype          = 'firws';
%         cfg.hpfiltdir           = 'onepass-zerophase';
%         cfg.hpfreq              = 0.1;
%         
%         cfg.bsfilter            = 'yes';
%         cfg.bsfreq              = [49 51];
%         
%         cfg.demean              = 'yes';
%         cfg.baselinewindow      = [-0.2 0];
% 
%         dat{run}                = ft_preprocessing(cfg);

        %% Create trial definition for FAMOUS / UNFAMILIAR / SCRAMBLED
        
        % recode the classes by collapsing initial/immediates/delayed
        % triggers to one class such that
        % 1 = FAMOUS
        % 2 = UNFAMILIAR
        % 3 = SCRAMBLED
%         for ii = 1:numel(ev{run})
%             switch ev{run}(ii).value
%                 case {5, 6, 7},    ev{run}(ii).value = 1;
%                 case {13, 14, 15}, ev{run}(ii).value = 2;
%                 case {17, 18, 19}, ev{run}(ii).value = 3;
%             end
%         end        
%         
%         trl           = [];
%         pretrig       = round(1 * dat{run}.fsample);   % 1 sec before trigger
%         posttrig      = round(1.5 * dat{run}.fsample); % 1.5 sec after trigger
%         
%         for ii = 1:numel(ev{run})
%             offset    = -pretrig;  % number of samples prior to the trigger
%             trlbegin  = ev{run}(ii).sample - pretrig;
%             trlend    = ev{run}(ii).sample + posttrig;
%             newtrl    = [trlbegin trlend offset];
%             trl       = [trl; newtrl]; % store in the trl matrix
%         end
%         
%         cfg      = [];
%         cfg.trl  = trl;
%         dat{run} = ft_redefinetrial(cfg, dat{run});
        
        % Put trials into a matrix
        cfg = [];
        cfg.keeptrials = 'yes';
        dat{run} = ft_timelockanalysis(cfg, dat{run});

    end
    
    %% Concatenate data and events, extract class labels
    dat = ft_appenddata([], dat{:});
    
    %% ERP baseline correction
    cfg = [];
    cfg.demean          = 'yes';
    cfg.baselinewindow  = [-0.2 0];
    
    dat = ft_preprocessing(cfg, dat);
    
    %% Save data   
    fprintf('Saving data...')
    save([preprocdir 'sbj-' num2str(nn)],'dat','trl')
    fprintf('finished\n')
    dat= [];
    
end

fprintf('Finished all.\n')
diary off
return

%% ERP analysis
% sanity check - calculate ERPs and compare them to the ERPs in the Wakeman
% Henson paper (see Figure 1)

famous = cell(nsubjects, 1);
familiar = famous;
scrambled = famous;
ntrials = zeros(nsubjects, 1);

for nn=1:nsubjects     % --- loop across subjects
    fprintf('loading subject #%d\n', nn)
    load([preprocdir 'sbj-' num2str(nn)],'dat')
    
    clabel = dat.trialinfo;
    ntrials(nn) = numel(clabel);
    continue
    % recode the classes by collapsing initial/immediates/delayed
    % triggers to one class such that
    % 1 = FAMOUS
    % 2 = UNFAMILIAR
    % 3 = SCRAMBLED
    clabel(ismember(clabel,[5,6,7])) = 1;
    clabel(ismember(clabel,[13,14,15])) = 2;
    clabel(ismember(clabel,[17,18,19])) = 3;
    
    % Extract data from the three classes separately and calculate ERPs
    cfg = [];
    cfg.trials = find(clabel==1);
    famous{nn} = ft_selectdata(cfg, dat);
    famous{nn} = ft_timelockanalysis([], famous{nn});

    cfg.trials = find(clabel==2);
    familiar{nn} = ft_selectdata(cfg, dat);
    familiar{nn} = ft_timelockanalysis([], familiar{nn});

    cfg.trials = find(clabel==3);
    scrambled{nn} = ft_selectdata(cfg, dat);
    scrambled{nn} = ft_timelockanalysis([], scrambled{nn});
end
return
% calculate grand average
avg_famous    = ft_timelockgrandaverage(cfg, famous{:});
avg_familiar  = ft_timelockgrandaverage(cfg, familiar{:});
avg_scrambled = ft_timelockgrandaverage(cfg, scrambled{:});


cfg = [];
cfg.xlim = [-0.1 0.8];
% cfg.ylim = [-1e-13 3e-13];
cfg.channel = 'EEG065';
ft_singleplotER(cfg, avg_famous, avg_familiar, avg_scrambled);
grid on

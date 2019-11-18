
%% MEG - Single subjects [don't average across subjects]
load benchmarking_results_single_subjects_matlab.mat

nsubjects = 16;
tmp = cell(nsubjects, 1);

for n=1:nsubjects
    acc = table();
    for f = fieldnames(meg_acc{1})'
        acc{:, f{:}}= meg_acc{n}.(f{:});
    end
    tmp{n} = acc;
end

T = mean(cat(3, tmp{:}),3);
writetable(T, 'benchmarking_results_single_subjects_matlab_acc.csv')

for n=1:nsubjects
    tim = table();
    for f = fieldnames(meg_time{1})'
        tim{:, f{:}}= meg_time{n}.(f{:});
    end
    tmp{n} = tim;
end

T = cat(1, tmp{:});

writetable(T, 'benchmarking_results_single_subjects_matlab_time.csv')

%% MEG - Single subjects [average across subjects]
load benchmarking_results_single_subjects_matlab.mat

nsubjects = 16;
tmp = cell(nsubjects, 1);

% add up accuracies
acc = meg_acc{1};
for n=2:nsubjects
    for f = fieldnames(meg_acc{1})'
        acc.(f{:}) = acc.(f{:}) + meg_acc{n}.(f{:});
    end
end
% turn into table
T= table();
for f = fieldnames(meg_acc{1})'
    T{:, f{:}} = acc.(f{:}) / nsubjects;
end

writetable(T, 'benchmarking_results_single_subjects_matlab_acc_average.csv')

% add up times
tim = meg_time{1};
for n=2:nsubjects
    for f = fieldnames(meg_time{1})'
        tim.(f{:}) = tim.(f{:}) + meg_time{n}.(f{:});
    end
end
% turn into table
T= table();
for f = fieldnames(meg_time{1})'
    T{:, f{:}} = tim.(f{:}) / nsubjects;
end

writetable(T, 'benchmarking_results_single_subjects_matlab_time_average.csv')


%% MEG super-subject
clear
load benchmarking_results_supersubject_matlab.mat

T = table();
for f = fieldnames(time)'
    T{:, f{:}}= time.(f{:});
end

writetable(T, 'benchmarking_results_supersubject_matlab.csv')

%% fMRI 
clear
load benchmarking_results_fmri_matlab.mat

T = table();
for f = fieldnames(time)'
    T{:, f{:}}= time.(f{:});
end

writetable(T, 'benchmarking_results_fmri_matlab.csv')
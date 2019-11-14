
%% MEG - Single subjects 
load benchmarking_results_single_subjects.mat

nsubjects = 16;
tmp = cell(nsubjects, 1);

for n=1:nsubjects
    acc = table();
    for f = fieldnames(meg_acc{1})'
        acc{:, f{:}}= meg_acc{n}.(f{:});
    end
    tmp{n} = acc;
end

T = cat(1, tmp{:});
writetable(T, 'benchmarking_results_single_subjects_acc.csv')

for n=1:nsubjects
    tim = table();
    for f = fieldnames(meg_time{1})'
        tim{:, f{:}}= meg_time{n}.(f{:});
    end
    tmp{n} = tim;
end

T = cat(1, tmp{:});

writetable(T, 'benchmarking_results_single_subjects_time.csv')

%% MEG - Single subjects 
clear
load benchmarking_results_supersubject.mat

T = table();
for f = fieldnames(time)'
    T{:, f{:}}= time.(f{:});
end

writetable(T, 'benchmarking_results_supersubject.csv')

%% fMRI 
clear
load benchmarking_results_haxby.mat

T = table();
for f = fieldnames(time)'
    T{:, f{:}}= time.(f{:});
end

writetable(T, 'benchmarking_results_haxby.csv')
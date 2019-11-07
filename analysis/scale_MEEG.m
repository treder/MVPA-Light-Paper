function dat = scale_MEEG(dat)
% Scale EEG and MEG data separately, since they have different units
% and are orders of magnitude apart. We will use the trace of the EEG
% and MEG covariance matrices for scaling.
% In contrast to z-scoring each channel separately, this type of
% scaling preserves the relative covariance relationships among EEG
% and MEG channels.

eeg_chans = ft_channelselection('EEG',dat.label);
meg_chans = ft_channelselection('MEG',dat.label);
eeg_ix = find(ismember(dat.label, eeg_chans));
meg_ix = find(ismember(dat.label, meg_chans));

% Average covariance for MEG and EEG
C_eeg = squeeze(mean(dat.cov(:, eeg_ix, eeg_ix),1));
C_meg = squeeze(mean(dat.cov(:, meg_ix, meg_ix),1));

scale_eeg = trace(C_eeg)/numel(eeg_chans);
scale_meg = trace(C_meg)/numel(meg_chans);

% Normalise EEG and MEG
dat.trial(:, eeg_ix, :) = dat.trial(:, eeg_ix, :) / sqrt(scale_eeg);
dat.trial(:, meg_ix, :) = dat.trial(:, meg_ix, :) / sqrt(scale_meg);
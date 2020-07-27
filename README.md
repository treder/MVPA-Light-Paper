# MVPA-Light-Paper

This is the companion repository of the [MVPA-Light paper](https://www.frontiersin.org/articles/10.3389/fnins.2020.00289/full). 

### Content of the subfolders

* [analysis](analysis): scripts for all the analyses (using of the EEG/MEG and fMRI datasets) reported in the paper. The following scripts replicate the analyses reported in the paper:
  * [preprocess_WakemanHenson](analysis/preprocess_WakemanHenson.m): Uses FieldTrip to preprocess the Wakeman and Henson EEG/MEG data.
  * [classify_WakemanHenson](analysis/classify_WakemanHenson.m): Single-trial classification for each subject (time classification, time generalization, time-frequency classification).
  * [statistics_WakemanHenson](analysis/statistics_WakemanHenson.m): Statistical analysis (both on single subject level and group level) of the classification results.
  * [classify_Haxby_fMRI](analysis/classify_Haxby_fMRI.m): Classification and statistical analyses of the Haxby et al fMRI data.
  * [benchmark_matlab](analysis/benchmark_matlab.m): Benchmarking analysis using MVPA-Light's classifiers/regression models, native MATLAB models, and LIBLINEAR/LIBSVM.
  * [benchmark_scikit_learn](analysis/benchmark_scikit_learn.py): Benchmarking analysis for Scikit-Learn (Python file).
  * [benchmark_R](analysis/benchmark_R.R): Benchmarking analysis for classifiers and regression models in R (R file).
  * [benchmark_plot_results](analysis/benchmark_plot_results.ipynb): Jupyter Notebook that compiles the results of the different benchmarking analyses.
* [paper](paper): tex files of the MVPA-Light manuscript 
* [figures](figures): figures used in the paper

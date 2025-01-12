# HBRL_WCST



## Order of Computation

>> Process raw data

1. b. `process_raw_data.ipynb`: extract raw data from the Psytoolkit `.txt` files and then store the data as `.pkl` files in a dataframe readable structure.
1. a. `encode_processed_data.ipynb`: encode categorical covariates.


>> Clean dataset

2. `Outlier_removal_dimentionily_reduction.ipynb`: remove outliers and stores final dataframes:
    - `../data_objects/final_dataframes/wcst_raw_data.csv`
    - `../data_objects/final_dataframes/covariates.csv`


3. `split subjects into performance groups`: groups the data in n contiguous groups.
    - `experiment.ipynb`: model development - superfluous & omitted.
    - `modules.py`: extract functions - superfluous & omitted.
    - `final_instance.ipynb`: final production run.
    
>> 
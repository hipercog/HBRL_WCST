# HBRL_WCST



## Order of Computation

1.b. `process_raw_data.ipynb`: extract raw data from the Psytoolkit `.txt` files and then store the data as `.pkl` files in a dataframe readable structure.
1.a. `encode_processed_data.ipynb`: encode categorical covariates.


2. `Outlier_removal_dimentionily_reduction.ipynb`: remove outliers and stores final dataframes:
    - `../data objects/final_dataframes/wcst_raw_data.csv`
    - `../data objects/final_dataframes/covariates.csv`
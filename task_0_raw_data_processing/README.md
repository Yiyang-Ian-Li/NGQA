# User Data Preparation 

### UPDATE 10132024

1. SEQN is now consistently stored as type str. This may cause small issues when merging tables together.
2. `label` in df_demo is now `opioid_label` for better clarification. 
3. BMI upper threshold is changed from 25.8 to 30 (obese), as too many Americans are overweighted.
4. `age` column in df_demo is no longer dropped after the aggregation of `age_group`, as many health standards only apply to users over 18. 
5. The user tagging pipeline and raw data generation pipeline have been merged together. 
6. Besides from regular tagging, another set of status columns is implemented, so we can now tell is user is labeled as hypertension or obesity.

### UPDATE 10142024

1. Now all tags in the table should be stored as integer 
2. `user_info_data.csv` table now contains all information on the user side, next step is to: 

    a. Do a filtering based on the stauts columns for a rough filtering.
    
    b. Prompt the data to generate a sentence that serves as the context of the user side info. 
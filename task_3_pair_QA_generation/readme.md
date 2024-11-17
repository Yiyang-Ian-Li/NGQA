# Benchmark generation

## Usage

Given `'../processed_data/user_tagging.csv'` and `'../task_2_foods_filtering_qa_creation/processed_data/reduced_mixed_dishes_v4.csv'`, we can run through `1_generate_pairs.ipynb` , `2_pairs_cleaning.ipynb`, and `4_QA_generation.ipynb` to generate the benchmark file, which will be saved at `'./processed_data/KGQA_benchmark.csv'`.

You can also directly run the last code block in `4_QA_generation.ipynb` to see the visualization of our benchmark if `'./processed_data/KGQA_benchmark.csv'` already exsits.

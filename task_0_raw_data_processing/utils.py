import pandas as pd 


def concat_data_across_years(data_type, file_code, years, year_char):
    """
    This function concat data across years.

    :param data_type: The NHANES data type (demographic, dietary, examination, laboratory, questionnaire)
    :param file_code: the NHANES code of the file.
    :param years: A list of strings identifies which year the data is from. E.g. ['0102', '0304', ..., '1314']
    :param year_char: The char NHANES data used to identify years. This char attaches to the file names as suffix.
        For 01-02 data, the char is B; For 03-04 data, the char is C.
    :return: Returns a pandas dataframe of the concatenated data
    """
    df = pd.DataFrame()
    root_path = '../data/'
    for year in years:
        if year == '1720':
            path = root_path + year + '/' + data_type + '/P_' + file_code + '.XPT'
        else:
            path = root_path + year + '/' + data_type + '/' + file_code + '_' + year_char + '.XPT'

        df_temp = pd.read_sas(path, encoding='ISO-8859-1')
        year_char = chr(ord(year_char) + 1)
        # Record which year the data comes from
        df_temp['years'] = year

        if df.empty:
            df = df_temp.copy()
        else:
            df = pd.concat([df, df_temp])

    # pandas read_sas has an issue to read 0 as 5.397e-79 ...
    df.replace(5.397605346934028e-79, 0, inplace=True)
    return df


def merge_with_or(df1, df2):
    """
    Merges two DataFrames on a specified key(s) and combines shared columns using an 'OR' relation. This function is 
    used when merging nutrition tags because we want to combine the tags from different sources using an 'OR' logic.
    
    Parameters:
    - df1, df2: DataFrames to be merged.
    
    Returns:
    - A merged DataFrame with combined shared columns using an 'OR' logic.
    """
    # Merge the DataFrames
    merged_df = pd.merge(df1, df2, left_index=True, right_index=True, how='left', suffixes=('_df1', '_df2'))
    merged_df = merged_df.fillna(0).astype(int)
    
    # Find shared columns, excluding the key(s) used for merging
    shared_columns = set(df1.columns) & set(df2.columns)
    
    for col in shared_columns:
        col_df1 = f'{col}_df1'
        col_df2 = f'{col}_df2'

        # Apply 'OR' operation for the shared column and assign it to the merged DataFrame
        # Only df2 contains NaN values, so we need to fill them with False before converting to int
        merged_df[col] = (merged_df[col_df1] | merged_df[col_df2].fillna(False).astype(int))
        
        # Drop the original columns from the merge
        merged_df.drop(columns=[col_df1, col_df2], inplace=True)
    
    return merged_df


def convert_tags(row, primary_nutrition_tags, nutrition_dict={}):
    """
    Converts high/low nutrition tags to a uniform dictionary format.
    Returns a dictionary with tag names as keys and values as -1 (low) or 1 (high).
    """
    for nutrition_tag in primary_nutrition_tags:
        if row[nutrition_tag] == 1:
            tag_name = nutrition_tag[4:] if 'low' in nutrition_tag else nutrition_tag[5:]
            nutrition_dict[tag_name] = -1 if 'low' in nutrition_tag else 1
    
    return nutrition_dict


def generate_pairs(food_nutrition_dict, nutrition_list, user_list, count, level, food_id):
    """
    Helper function to generate pairs based on the specified difficulty level.
    """
    pair_results = []
    loop_seed = food_id
    while True:
        loop_seed += 1
        random.seed(loop_seed)
        random.shuffle(nutrition_list)

        # Set number of tags in common based on difficulty level
        if level == 'easy':
            num_tag_in_common = 1
        else:
            num_tag_in_common = random.randint(2, len(nutrition_list))

        selected_nutrition = nutrition_list[:num_tag_in_common]
        avoid_nutrition = nutrition_list[num_tag_in_common:]

        # Shuffle user_list in place
        random.shuffle(user_list)
        for user in user_list:
            # Check if user only has selected nutrition tags
            if any(nutrition in user for nutrition in avoid_nutrition):
                continue
            if not all(nutrition in user for nutrition in selected_nutrition):
                continue

            # Specific condition for 'medium' and 'hard'
            if level in ['medium', 'hard']:
                tag_match_count = sum(
                    1 if food_nutrition_dict[nutrition] == user[nutrition] else -1
                    for nutrition in selected_nutrition
                )
                if (level == 'medium' and abs(tag_match_count) != len(selected_nutrition)) or \
                   (level == 'hard' and abs(tag_match_count) == len(selected_nutrition)):
                    continue

            pair_results.append(user)
            break

        if len(pair_results) == count or loop_seed > 50 + food_id:
            break

    return pair_results
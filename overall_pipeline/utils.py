import pandas as pd
import random 


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
    Generates user-food pairs by selecting users who match specific nutrition tags associated with a food item, 
    according to a specified difficulty level.

    Parameters:
    - food_nutrition_dict: Dictionary with nutrition tags and values for the target food.
    - nutrition_list: List of nutrition tags relevant to the matching.
    - user_list: List of users with their nutrition tags.
    - count: The number of pairs to generate.
    - level: Difficulty level ('easy', 'medium', 'hard') controlling tag matching criteria.
    - food_id: Identifier for the food, also used as a seed for randomness.

    Returns:
    - A list of deduplicated user dictionaries that match the nutrition criteria based on the selected difficulty level.
    """
    pair_results = []
    added_users = set()  # Track users to avoid duplication
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
            user_id = user.get('user_id')  # Assuming each user has a unique identifier
            if user_id in added_users:
                continue
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
            added_users.add(user_id)
            break

        if len(pair_results) == count or loop_seed > 50 + food_id:
            break

    return pair_results


def generate_answer(food_tag, user):
    """
    Generates answers based on the compatibility between a food's nutrition tags and a user's preferences.

    Parameters:
    - food_tag: Dictionary containing nutrition tags and their levels ('high' or 'low') for the food.
    - user: Dictionary containing user-specific nutrition preferences.

    Returns:
    - answer_easy: A simple 'Yes' or 'No' indicating compatibility at a high level.
    - answer_medium: A string listing the nutrition tags involved in the decision-making process.
    - answer_hard: A detailed explanation of why the food is or isn't compatible with the user.
    """
    # List to hold positive reasons and negative reasons 
    reasons_for_yes = []  
    reasons_for_no = []   
    used_nutrition_tags = []  # Tracks nutrition tags considered in the decision

    # Iterate through user's nutrition preferences, skipping the first key (e.g., user_id)
    for nutrition in list(user.keys())[1:]:
        if nutrition in list(food_tag.keys()):  
            level = 'high' if food_tag[nutrition] == 1 else 'low'
            used_nutrition_tags.append(level + '_' + nutrition)  
            
            # Append matching or mismatching reasons
            if user[nutrition] == food_tag[nutrition]:
                reasons_for_yes.append(f"{level} in {nutrition}")
            else:
                reasons_for_no.append(f"{level} in {nutrition}")

    # Generate the detailed "hard" answer based on majority matching or non-matching reasons.
    # Note that only hard questions can have conflicting reasons. This is how we determine the final answer.
    if len(reasons_for_no) >= len(reasons_for_yes):
        answer_hard = 'No, because the food is '
        for reason in reasons_for_no:
            answer_hard += reason + ', '  
        answer_hard = answer_hard[:-2] + '. '  
    else:
        answer_hard = 'Yes, because the food is '
        for reason in reasons_for_yes:
            answer_hard += reason + ', '  
        answer_hard = answer_hard[:-2] + '. '  

    answer_easy = 'Yes' if 'Yes' in answer_hard else 'No'
    answer_medium = ', '.join(used_nutrition_tags)

    return answer_easy, answer_medium, answer_hard


def generate_graph(food_id, user_id, food_info, user_info, food_primary_nutrition_tags, reference_dict):
    """
    Generates a graph structure representing relationships between a user, food, and their attributes.

    Parameters:
    - food_id: The ID of the food item.
    - user_id: The ID of the user.
    - food_info: DataFrame containing food attributes.
    - user_info: DataFrame containing user attributes.
    - food_primary_nutrition_tags: List of primary nutrition tags for foods.
    - reference_dict: Dictionary mapping user statuses to nutrition tags.

    Returns:
    - node_list: List of nodes in the graph, each represented as [node_id, {'name': str, 'attr': value}].
    - edge_list: List of edges in the graph, each represented as [node_id, relation, node_id].
    """
    node_list, edge_list  = [], []  
    node_id = 2  # Start assigning node IDs from 2 (0: user, 1: food)

    # Add the food node
    food = food_info[food_info['food_id'] == food_id]
    node_list.append([1, {'name': food['food_desc'].item(), 'attr': food_id}])  # Node ID 1 is the food

    # Add nutrition tags for the food
    for column in food_primary_nutrition_tags:
        if food[column].item() == 1:
            node_list.append([node_id, {'name': 'food_nutrition_tag', 'attr': column}])
            edge_list.append([1, 'belongs to', node_id])  # Food has this nutrition tag
            node_id += 1

    # Add the user node
    node_list.append([0, {'name': 'user', 'attr': user_id}])  # Node ID 0 is the user
    user = user_info[user_info['SEQN'] == user_id]

    # Add user statuses and match them to nutrition tags
    for column, nutrition_tags in reference_dict.items():
        if user[column].item() == 1:  # If the user has this status
            status_node_id = node_id
            node_list.append([node_id, {'name': 'status', 'attr': column}])
            edge_list.append([0, 'has', node_id])  # User has this status
            node_id += 1

            # Match user statuses to food nutrition tags or create new user nutrition tag nodes
            for nutrition_tag in nutrition_tags:
                level, nutrition = nutrition_tag.split('_')
                # Check for matching food nutrition tags
                match_found = False
                for node in node_list:
                    if node[1]['name'] == 'food_nutrition_tag' and node[1]['attr'] == nutrition:
                        relation = 'match' if level in node[1]['attr'] else 'contradict'
                        edge_list.append([status_node_id, relation, node[0]])
                        match_found = True
                        break

                if not match_found:
                    # Add new user nutrition tag if no matching food nutrition tag is found
                    node_list.append([node_id, {'name': 'user_nutrition_tag', 'attr': nutrition_tag}])
                    edge_list.append([status_node_id, 'need', node_id])
                    node_id += 1

    return node_list, edge_list


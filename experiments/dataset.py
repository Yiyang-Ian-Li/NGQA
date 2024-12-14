import pandas as pd
import ast
import networkx as nx

import warnings
warnings.filterwarnings("ignore")


class Dataset:
    def __init__(self, file_path):
        """
        Initialize the Dataset object by loading data from a CSV file.
        """
        self.data = pd.read_csv(file_path)

    def task_level_filtering(self, task_level, data):
        """
        Filter rows based on the task level.
        Args:   
            task_level (str): Task level ('easy', 'medium', 'hard').
            data (pd.DataFrame): DataFrame containing the data to filter.
        Returns: 
            pd.DataFrame: Filtered rows for the specified task level.
        """
        if task_level not in ['easy', 'medium', 'hard']:
            raise ValueError("Invalid task level. Must be 'easy', 'medium', or 'hard'.")

        question_column = f"question_{task_level}"
        answer_column = f"answer_{task_level}"

        # Filter rows where the question and answer columns for the task level are not null
        filtered_data = data[data[question_column].notnull() & data[answer_column].notnull()]
        return filtered_data

    def question_level_filtering(self, question_level):
        """
        Filter rows based on the question level (e.g., 'easy', 'medium', 'hard').
        Args: question_level (str): Question level to filter by.
        Returns: pd.DataFrame: Filtered rows for the specified question level.
        """
        if question_level:
            return self.data[self.data['difficulty'] == question_level]
        return self.data

    def get_graphs(self, filtered_data):
        """
        Generate a list of graphs corresponding to the questions in the filtered data.
        Args: filtered_data (pd.DataFrame): DataFrame containing the filtered questions.
        Returns: list: A list of NetworkX graphs corresponding to the questions.
        """
        graphs = []
        for _, row in filtered_data.iterrows():
            node_list = ast.literal_eval(row['node_list'])
            edge_list = ast.literal_eval(row['edge_list'])
            
            # Create NetworkX graph
            graph = nx.DiGraph()

            # Add nodes with attributes
            for node in node_list:
                graph.add_node(node[0], **node[1])

            # Add edges with attributes
            for edge in edge_list:
                if len(edge) == 3:  # [source, relationship, target]
                    graph.add_edge(edge[0], edge[2], relationship=edge[1])

            graphs.append(graph)
        
        return graphs

    def sampling(self, n, filtered_data, seed=42):
        """
        Randomly sample a specific number of rows from the dataset or filtered data.
        """
        return filtered_data.sample(n=n, random_state=seed)
    
    def process(self, question_level=None, task_level=None, sample=False, n=100):
        """
        Process the dataset to filter by task level, question level, and optionally sample rows.
        
        Args:
            question_level: Question level to filter by ('easy', 'medium', 'hard').
                                            If None, include all question levels.
            task_level: Task level to filter by ('easy', 'medium', 'hard'). 
            sample (bool): Whether to sample the filtered rows.
            n (int): Number of rows to sample.
        
        Returns:
            - questions (list): List of filtered questions.
            - answers (list): List of corresponding answers.
            - graphs (list): List of corresponding graphs.
        """
        if task_level is None:
            raise ValueError("Task level must be specified. Please provide 'easy', 'medium', or 'hard'.")

        # Filter by question level (include all levels if not specified)
        filtered_data = self.question_level_filtering(question_level)

        # Filter by task level
        filtered_data = self.task_level_filtering(task_level, filtered_data)

        # Sample rows if required
        if sample:
            filtered_data = self.sampling(n, filtered_data)

        # Extract questions, answers, and graphs
        questions = filtered_data[f"question_{task_level}"].tolist()
        answers = filtered_data[f"answer_{task_level}"].tolist()
        graphs = self.get_graphs(filtered_data)

        return questions, answers, graphs
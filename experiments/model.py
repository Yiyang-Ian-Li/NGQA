import time
from tqdm import tqdm
import logging
from openai import OpenAI
import networkx as nx

from utils import find_relations, prune_relations, find_entities, prune_entities, convert_to_sg

import warnings
warnings.filterwarnings("ignore")


class Retriever:
    def __init__(self, graphs, model_name=''):
        """
        Initialize the Retriever with a list of graphs.
        Args: graphs (list): List of NetworkX graphs.
        """
        self.graphs = graphs
        self.model_name = model_name

    def plain_retriever(self, graph):
        """
        Plain Baseline: Do nothing. Return the entire graph.
        """
        return graph

    def KAPING_retriever(self, graph):
        """
        KAPING-style retrieval: Retrieve the user node (0), food node (1),
        their immediate neighbors, and only the edges directly connecting them.
    
        Args:
            graph (nx.Graph): Input graph.
    
        Returns:
            nx.Graph: Subgraph containing the user, food, their neighbors,
                      and only edges connected to user and food.
        """
        # Initialize subgraph
        subgraph = nx.DiGraph()
    
        # Define source nodes (user and food)
        source_nodes = [0, 1]
    
        # Add source nodes to the subgraph
        for source in source_nodes:
            if source in graph:
                # Add the source node
                subgraph.add_node(source, **graph.nodes[source])
    
                # Add neighbors and edges directly connected to the source
                for neighbor in graph.neighbors(source):
                    subgraph.add_node(neighbor, **graph.nodes[neighbor])  # Add neighbor node
                    subgraph.add_edge(source, neighbor, **graph[source][neighbor])  # Add edge from source to neighbor
    
        return subgraph

    def tog_retriever(self, graph, api_key, question):
        """
        Custom retrieval method: Implement your own retrieval logic here.
        """
        # args for ToG
        depth = 3
        width = 5
        
        # Initialize the OpenAI client
        if 'gpt' in self.model_name:
            client = OpenAI(api_key=api_key)
        else:
            # TODO - Implement the LLM API client
            pass

        # Initial raw subgraph
        reasoning_path_list = [[graph.nodes[0]['attr']], [graph.nodes[1]['attr']]]
        
        # Start thinking
        for i in range(depth):
            
            # Relation's round
            candidate_reasoning_path_list = []
            for reasoning_path in reasoning_path_list:
                # find new
                new_reasoning_path_list = find_relations(graph, reasoning_path)
                if new_reasoning_path_list != []:
                    candidate_reasoning_path_list.extend(new_reasoning_path_list)
                else:
                    candidate_reasoning_path_list.append(reasoning_path)
            # Prune
            # candidate_reasoning_path_list = [path for path in candidate_reasoning_path_list if len(path) >= (i + 1) * 2]
            if i > 0:
                reasoning_path_list = prune_relations(client, candidate_reasoning_path_list, question, self.model_name, width)
            else:
                reasoning_path_list = candidate_reasoning_path_list
            reasoning_path_list = candidate_reasoning_path_list
            
            # Entity's round
            candidate_reasoning_path_list = []
            for reasoning_path in reasoning_path_list:
                # find new
                new_reasoning_path_list = find_entities(graph, reasoning_path)
                if new_reasoning_path_list != []:
                    candidate_reasoning_path_list.extend(new_reasoning_path_list)
                else:
                    candidate_reasoning_path_list.append(reasoning_path)
            # Prune
            if i > 0:
                reasoning_path_list = prune_entities(client, candidate_reasoning_path_list, question, self.model_name, width)
            else:
                reasoning_path_list = candidate_reasoning_path_list
            
            # Return the subgraphs if the depth is reached
            if i == depth - 1:
                # for reasoning_path in reasoning_path_list:
                #     print(reasoning_path)
                subgraph = convert_to_sg(graph, reasoning_path_list)
                return subgraph
            
    def custom_retriever(self, graph):
        """
        Custom retrieval method: Implement your own retrieval logic here.
        """
        pass
    
    def retrieve(self, method="plain", **kwargs):
        """
        Apply the specified retrieval method to all graphs in the dataset.
        
        Args:
            method (str): Retrieval method ('plain', 'kaping', 'custom').
            **kwargs: Additional arguments for the retrieval methods.
        
        Returns:
            list: List of retrieved subgraphs.
        """
        retrieved_graphs = []
        for i, graph in tqdm(enumerate(self.graphs), desc="Retrieving Subgraphs", total=len(self.graphs)):
            if method == "plain" or method == "zero_cot" or method == "cot_bag":
                retrieved_graphs.append(self.plain_retriever(graph))
            elif method == "KAPING":
                retrieved_graphs.append(self.KAPING_retriever(graph))
            elif method == 'ToG':
                retrieved_graphs.append(self.tog_retriever(graph, api_key=kwargs['api_key'], question=kwargs['questions'][i]))
            elif method == 'custom':
                retrieved_graphs.append(self.custom_retriever(graph))
            else:
                raise ValueError(f"Unknown retrieval method: {method}")
        return retrieved_graphs


class RetrievalEvaluator:
    def __init__(self, graphs):
        """
        Initialize the RetrievalEvaluator with a list of graphs.
        Args: graphs (list): List of NetworkX graphs.
        """
        self.graphs = graphs
    
    def evaluate(self, retrieved_graphs):
        '''
        Evaluate the retrieved subgraphs against the optimal subgraphs.
        '''
        results = {
            'Precision': 0,
            'Recall': 0,
            'F1 Score': 0
        }
        
        for r_g, g in zip(retrieved_graphs, self.graphs):
            optimal_nodes = set()
            optimal_paths = nx.all_simple_paths(g.to_undirected(), 0, 1)
            for path in optimal_paths:
                for i in path:
                    optimal_nodes.add(i)
            optimal_names = [g.nodes[i]['attr'] for i in optimal_nodes]
            
            retrieved_names = [r_g.nodes[i]['attr'] for i in r_g.nodes]
            
            precision = len(set(retrieved_names).intersection(optimal_names)) / len(retrieved_names)
            recall = len(set(retrieved_names).intersection(optimal_names)) / len(optimal_names)
            
            if precision + recall > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = 0
            
            results['Precision'] += precision
            results['Recall'] += recall
            results['F1 Score'] += f1_score
            
        results['Precision'] /= len(self.graphs)
        results['Recall'] /= len(self.graphs)
        results['F1 Score'] /= len(self.graphs)
        
        # Round the results
        for metric in results:
            results[metric] = round(results[metric], 3)
            
        return results


class Augmenter:
    def __init__(self):
        pass

    def graph_to_triplets(self, graph):
        """
        Convert a graph into triplets and textualize them in a proper sequence.
        Args: graph (nx.Graph): The input graph (filtered subgraph).
        Returns: str: A textualized sequence of triplets.
        """
        triplets = []
        # Iterate over edges to form triplets (source_node, relationship, target_node)
        for source, target, data in graph.edges(data=True):
            # Extract the relationship (edge name/attribute) from the edge data
            relationship = data.get("relationship", "related to")  # Default if no relationship is defined

            # Extract node names or fallback to their IDs
            source_name = graph.nodes[source].get("attr", str(source))
            target_name = graph.nodes[target].get("attr", str(target))

            # Construct the triplet using the edge relationship
            triplets.append(f"({source_name} {relationship} {target_name})")

        # Combine triplets into a single textualized sentence
        return ", ".join(triplets)
    
    def augment(self, graphs, method='to_triplets'):
        """
        Convert a list of graphs into a list of textualized triplets.
        Returns: ist: A list of textualized sequences, one for each graph.
        """
        textualized_graphs = []
        # Iterate over the list of graphs and convert each to triplets
        for graph in graphs:
            if method == 'to_triplets':
                textualized_graph = self.graph_to_triplets(graph)
            else:
                raise ValueError(f"Unknown augmentation method: {method}")
            textualized_graphs.append(textualized_graph)

        return textualized_graphs


import time
import logging
from tqdm import tqdm

class Generator:
    def __init__(self, api_key, model_name, note_prompt, method_prompt, sleeptime=0):
        """
        Initialize the Generator class with API key, model, and prompts.
        
        Args:
            api_key (str): API key for authentication.
            model_name (str): Name of the model to use (e.g., "llama3-70b").
            note_prompt (str): The prompt for important notes. Used for limit the output format. 
            method_prompt (str): The prompt unique to the baseline method for better task clarification.
            sleeptime (int): Time (in seconds) to sleep between API calls to avoid rate limiting.
        """
        self.api_key = api_key
        
        if 'gpt' in model_name:
            self.gpt = OpenAI(
                api_key=self.api_key
            )
        else:
            # TODO - Implement the LLM API client
            pass
        
        self.model_name = model_name
        self.system_prompt = "Act as a nutritionist. Analyze if a given food is healthy to a user and why."
        self.note_prompt = note_prompt
        self.method_prompt = method_prompt
        self.sleeptime = sleeptime

        logging.basicConfig(level=logging.INFO)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.logger = logging.getLogger(__name__)

    def generate_prompt(self, question, textualized_graph):
        """
        Generate a prompt by combining question, method_prompt, textualized_graph, and note_prompt.
        """
        return f"{question}. {self.method_prompt}. {textualized_graph}. {self.note_prompt}"

    def query_api(self, prompt, retries=3, delay=2):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        
        for attempt in range(retries):
            try:
                if 'gpt' in self.model_name:  
                    response = self.gpt.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0
                    ).choices[0].message.content
                    
                else:
                    # TODO - Implement the LLM API call
                    pass
                    
                # time.sleep(0.15)
                return response
            except Exception as e:
                self.logger.error(f"API Error: {e}")
                time.sleep(delay)
        
        return "API Error after multiple retries"

    def generate_predictions(self, questions, textualized_graphs):
        """
        Generate predictions for a list of questions and corresponding textualized graphs.

        Args:
            questions (list): List of questions.
            textualized_graphs (list): List of textualized graphs.

        Returns:
            list: List of predictions (one for each question).
        """
        if len(questions) != len(textualized_graphs):
            raise ValueError("The number of questions and textualized graphs must be the same.")

        predictions = []

        # Generate and query prompts
        for question, textualized_graph in tqdm(zip(questions, textualized_graphs), desc="Generating Predictions", total=len(questions)):
            prompt = self.generate_prompt(question, textualized_graph)
            prediction = self.query_api(prompt)
            predictions.append(prediction)
            time.sleep(self.sleeptime)

        return predictions
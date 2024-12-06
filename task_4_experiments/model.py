import time
from tqdm import tqdm
import logging
from llamaapi import LlamaAPI

import warnings
warnings.filterwarnings("ignore")


class Retriever:
    def __init__(self, graphs):
        """
        Initialize the Retriever with a list of graphs.
        Args: graphs (list): List of NetworkX graphs.
        """
        self.graphs = graphs

    def plain_retriever(self, graph):
        """
        Plain Baseline: Do nothing. Return the entire graph.
        """
        return graph

    def KAPING_retriever(self, graph):
        """
        KAPING-style retrieval: Find the user node (0) and food node (1),
        and include their immediate neighbors in the subgraph.
        
        Returns:
            nx.Graph: Subgraph containing the user, food, and their neighbors.
        """
        # Initialize nodes to include in the subgraph
        nodes_to_include = {0, 1}  # User and food nodes

        # Find neighbors of user (0) and food (1)
        for node in [0, 1]:
            neighbors = set(graph.neighbors(node))
            nodes_to_include.update(neighbors)

        # Create a subgraph with selected nodes
        return graph.subgraph(nodes_to_include)

    def custom_retriever(self, graph):
        """
        Custom retrieval method: Implement your own retrieval logic here.
        """
        pass
        
    def retrieve(self, method="plain", **kwargs):
        """
        Apply the specified retrieval method to all graphs in the dataset.

        Args:
            method (str): Retrieval method ('plain', 'KAPING', 'custom', 'zero_cot').
            **kwargs: Additional arguments for the retrieval methods.

        Returns:
            list: List of retrieved subgraphs.
        """
        retrieved_graphs = []
        for graph in self.graphs:
            if method == "plain" or method == "zero_cot" or method == "cot_bag":
                retrieved_graphs.append(self.plain_retriever(graph))
            elif method == "KAPING":
                retrieved_graphs.append(self.KAPING_retriever(graph))
            elif method == "custom":
                retrieved_graphs.append(self.custom_retriever(graph))
            else:
                raise ValueError(f"Unknown retrieval method: {method}")
        return retrieved_graphs


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
import openai 

class Generator:
    def __init__(self, api_key, model_name, note_prompt, method_prompt, sleeptime=0):
        """
        Initialize the Generator class with API key, model, and prompts.
        
        Args:
            api_key (str): API key for authentication.
            model_name (str): Name of the model to use (e.g., "llama3-70b", "gpt-3.5-turbo").
            note_prompt (str): The prompt for important notes. Used for limit the output format. 
            method_prompt (str): The prompt unique to the baseline method for better task clarification.
            sleeptime (int): Time (in seconds) to sleep between API calls to avoid rate limiting.
        """
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = "Act as a nutritionist. Analyze if a given food is healthy to a user and why."
        self.note_prompt = note_prompt
        self.method_prompt = method_prompt
        self.sleeptime = sleeptime

        if self.model_name == "llama3.1-70b":
            self.llama = LlamaAPI(self.api_key)  
        elif self.model_name == "gpt-3.5-turbo" or self.model_name == "gpt-4o-mini":
            openai.api_key = self.api_key 
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def generate_prompt(self, question, textualized_graph):
        """
        Generate a prompt by combining question, method_prompt, textualized_graph, and note_prompt.
        """
        return f"{question}. {self.method_prompt}. {textualized_graph}. {self.note_prompt}"

    def query_llama(self, prompt):
        api_request_json = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
        }
        try:
            response = self.llama.run(api_request_json)
            content = response.json().get('choices', [{}])[0].get('message', {}).get('content', "No content returned.")
            return content
        except Exception as e:
            self.logger.error(f"LLama API Error for prompt '{prompt}': {e}")
            return "API Error"

    def query_gpt_35_turbo(self, prompt, retries=3, delay=2):
        for attempt in range(retries):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )
                return response.choices[0].message.content
            except Exception as e:
                self.logger.error(f"GPT-3.5-turbo API Error on attempt {attempt + 1}: {e}")
                time.sleep(delay)
        return "API Error after multiple retries"
    
    def query_gpt_4o_mini(self, prompt, retries=3, delay=2):
        for attempt in range(retries):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0  # Adjust temperature if needed
                )
                return response.choices[0].message.content
            except Exception as e:
                self.logger.error(f"GPT-4o-mini API Error on attempt {attempt + 1}: {e}")
                time.sleep(delay)
        return "API Error after multiple retries"

    def query_api(self, prompt):
        if self.model_name == "llama3.1-70b":
            return self.query_llama(prompt)
        elif self.model_name == "gpt-3.5-turbo":
            return self.query_gpt_35_turbo(prompt)
        elif self.model_name == "gpt-4o-mini":
            return self.query_gpt_4o_mini(prompt)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

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
        
        print(f"Using model: {self.model_name}")
        predictions = []
        for question, textualized_graph in tqdm(zip(questions, textualized_graphs), desc="Generating Predictions", total=len(questions)):
            prompt = self.generate_prompt(question, textualized_graph)
            prediction = self.query_api(prompt)
            predictions.append(prediction)
            time.sleep(self.sleeptime)
        return predictions
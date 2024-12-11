import re
import networkx as nx


def convert_to_txt(path):
    full_path = path[:1]
    for i in range(1, len(path) - 1):
        if i % 2 == 1:
            full_path.append(path[i])
        else:
            full_path.extend([path[i], ',', path[i]])
    full_path.append(path[-1])
    return ' '.join(full_path)


def convert_to_sg(graph, path_list):
    """
    Generate subgraphs based on the list of paths and merge these subgraphs into a new subgraph.

    Args:
        graph (nx.Graph or nx.DiGraph): The original graph.
        path_list (list of list): Each path is a collection of node attributes and relationships.

    Returns:
        nx.DiGraph: The merged subgraph.
    """
    merged_graph = nx.DiGraph()  # Used to store the merged subgraph

    for path in path_list:
        # Create a subgraph for the current path
        sg = graph.copy()
        
        # Remove nodes not included in the path
        nodes_to_remove = [node for node in sg.nodes() if str(sg.nodes[node]['attr']) not in path]
        sg.remove_nodes_from(nodes_to_remove)

        # Remove edges not included in the path
        edges_to_remove = [
            (u, v) for u, v, attr in sg.edges(data=True)
            if str(sg.nodes[u]['attr']) not in path
            or str(sg.nodes[v]['attr']) not in path
            or attr['relationship'] not in path
        ]
        sg.remove_edges_from(edges_to_remove)

        # Merge the nodes and edges of the current subgraph into the merged graph
        merged_graph.add_nodes_from(sg.nodes(data=True))
        merged_graph.add_edges_from(sg.edges(data=True))

    return merged_graph


def find_relations(graph, path):
    candidate_path_list = []
    for edge in graph.edges(data=True):
        if (str(graph.nodes[edge[0]]['attr']) == path[-1] or str(graph.nodes[edge[1]]['attr']) == path[-1])\
              and not (str(graph.nodes[edge[0]]['attr']) in path and str(graph.nodes[edge[1]]['attr']) in path):
            candidate_path = path + [edge[2]['relationship']]
            if candidate_path not in candidate_path_list:
                candidate_path_list.append(candidate_path)
    return candidate_path_list


def prune_relations(client, path_list, question, model_name, width):
    if len(path_list) <= width:
        return path_list
    
    reasoning_path_list = [f'{i + 1}. ' + convert_to_txt(path) + '.\n' for i, path in enumerate(path_list)]
    
    messages = [
                    {
                        'role': 'system',
                        'content': f'Identify the top-{width} reasoning paths extracted from a knowledge graph that are most likely to lead to the answer for the query. \
                                    Respond with the indices of the reasoning paths, starting from 1, and separate them with commas (e.g., 1,2,5). Include nothing else in your response.'
                    },
                    {
                        'role': 'user', 
                        'content': f'The query is {question}, and the reasoning paths are: \n{reasoning_path_list}. Your selected top-{width} reasoning paths are:' 
                    }
                ]
    
    if 'gpt' in model_name:
        answer = client.chat.completions.create(
            model = model_name,
            messages = messages,
            temperature = 0,
        ).choices[0].message.content
    elif 'llama' in model_name:
        answer = client.run({
            'model': model_name,
            'messages': messages
        }).json()['choices'][0]['message']['content']

    indices = re.findall(r'\d+', answer)
    indices = [int(index) - 1 for index in indices]
    if len(indices) > width:
        indices = indices[:width]
    # Delete indices that are out of range
    indices = [index for index in indices if index < len(path_list)]
    
    if len(indices) == 0:
        return path_list[:width]
    
    path_list = [path_list[index] for index in indices]
    return path_list


def find_entities(graph, path):
    node_name = path[-2]
    edge_name = path[-1]
    candidate_path_list = []
    for edge in graph.edges(data=True):
        if str(graph.nodes[edge[0]]['attr']) == node_name and edge[2]['relationship'] == edge_name \
            and not str(graph.nodes[edge[1]]['attr']) in path:
            candidate_path = path + [str(graph.nodes[edge[1]]['attr'])]
            if candidate_path not in candidate_path_list:
                candidate_path_list.append(candidate_path)
        elif str(graph.nodes[edge[1]]['attr']) == node_name and edge[2]['relationship'] == edge_name \
            and not str(graph.nodes[edge[0]]['attr']) in path:
            candidate_path = path + [str(graph.nodes[edge[0]]['attr'])]
            if candidate_path not in candidate_path_list:
                candidate_path_list.append(candidate_path)
    return candidate_path_list


def prune_entities(client, path_list, question, model_name, width):
    if len(path_list) <= width:
        return path_list
    
    reasoning_path_list = [f'{i + 1}. ' + convert_to_txt(path) + '.\n' for i, path in enumerate(path_list)]

    messages = [
                    {
                        'role': 'system',
                        'content': f'Identify the top-{width} reasoning paths extracted from a knowledge graph that are most likely to lead to the answer for the query. \
                                    Respond with the indices of the reasoning paths, starting from 1, and separate them with commas (e.g., 1,2,5). Include nothing else in your response.'
                    },
                    {
                        'role': 'user', 
                        'content': f'The query is {question}, and the reasoning paths are: \n{reasoning_path_list}. Your selected top-{width} reasoning paths are:' 
                    }
                ]
    
    if 'gpt' in model_name:
        answer = client.chat.completions.create(
            model = model_name,
            messages = messages,
            temperature = 0,
        ).choices[0].message.content
    elif 'llama' in model_name:
        answer = client.run({
            'model': model_name,
            'messages': messages
        }).json()['choices'][0]['message']['content']

    indices = re.findall(r'\d+', answer)
    indices = [int(index) - 1 for index in indices]
    if len(indices) > width:
        indices = indices[:width]
    # Delete indices that are out of range
    indices = [index for index in indices if index < len(path_list)]
        
    if len(indices) == 0:
        return path_list[:width]
    
    path_list = [path_list[index] for index in indices]
    return path_list

def generate_paragraph_cot_bag(textualized_triplets):
    """
    Convert textualized triplets into a paragraph format for CoT_BaG.

    Args:
        textualized_triplets (str): A single string of textualized triplets.

    Returns:
        str: Paragraph format of the input triplets.
    """
    nodes = set()
    edges = []

    # Define possible relationships to extract
    valid_relationships = ["belongs to", "has", "contains", "match", "contradict", "need"]

    # Split the textualized triplets into individual triplets
    triplet_list = textualized_triplets.strip().strip("()").split("), (")
    
    for triplet in triplet_list:
        try:
            # Ensure each triplet is well-formatted and split based on relationships
            for relationship in valid_relationships:
                if f" {relationship} " in triplet:
                    source, target = triplet.split(f" {relationship} ", 1)
                    source = source.strip()
                    target = target.strip()

                    # Add nodes and edges
                    nodes.update([source, target])
                    edges.append(f'an edge between node "{source}" directed to node "{target}" with attribute "{relationship}"')
                    break
            else:
                print(f"Warning: Relationship not found in triplet '{triplet}'")  # Debugging
        except Exception as e:
            print(f"Error processing triplet '{triplet}': {e}")  # Debugging

    # Convert nodes and edges to paragraph format
    node_list = ", ".join(f'"{node}"' for node in nodes)
    edge_list = ", ".join(edges)

    paragraph = (
        f"You are given a directed graph where the nodes and edges are: {edge_list}. Let's construct a graph with the nodes and edges, then provide the output adhering to the following guideline."
    )

    return paragraph

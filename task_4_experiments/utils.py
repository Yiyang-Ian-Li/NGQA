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
    if len(path_list) < 4:
        return path_list
    
    reasoning_path_list = [convert_to_txt(path) for path in path_list]
    
    messages = [
                    {
                        'role': 'system',
                        'content': f'You are asked to find the top-{width} reasoning paths extracted from a knowledge graph that are most likely be able to lead the anwer to the query.\
                                    You should anwer with the index of the reasoning paths, starting from 0, separated by comma, e.g., 0,2,5. Nothin else should be included.\
                                    The knowledge graph schema contains the following triplets:\
                                    (food, belongs to, category),\
                                    (food, has, ingredients), \
                                    (food, contains, nutrition tag), \
                                    (nutrition tag, match, status),\
                                    (nutrition tag, contradict, status),\
                                    (user, has, status), \
                                    (user, has, dietary habits).'
                    },
                    {
                        'role': 'user', 
                        'content': f'The query is {question}, and the reasoning paths are: \n{reasoning_path_list}.\n Your choice of top-{width} reasoning paths are:' 
                    }
                ]
    
    if 'gpt' in model_name:
        answer = client.chat.completions.create(
            model = model_name,
            # temperature = 0.4,
            messages = messages
        ).choices[0].message.content
    elif 'llama' in model_name:
        answer = client.run({
            'model': model_name,
            'messages': messages
        }).json()['choices'][0]['message']['content']

    indices = re.findall(r'\d+', answer)
    indices = [int(index) for index in indices]
    # Delete indices that are out of range
    indices = [index for index in indices if index < len(path_list)]
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
    if len(path_list) < 4:
        return path_list
    
    reasoning_path_list = [convert_to_txt(path) for path in path_list]

    messages = [
                    {
                        'role': 'system',
                        'content': f'You are asked to find the top-{width} reasoning paths extracted from a knowledge graph that are most likely be able to lead the anwer to the query.\
                                    You should anwer with the index of the reasoning paths, starting from 0, separated by comma, e.g., 0,2,5. Nothin else should be included.\
                                    The knowledge graph schema contains the following triplets:\
                                    (food, belongs to, category),\
                                    (food, has, ingredients), \
                                    (food, contains, nutrition tag), \
                                    (nutrition tag, match, status),\
                                    (nutrition tag, contradict, status),\
                                    (user, has, status), \
                                    (user, has, dietary habits).'
                    },
                    {
                        'role': 'user', 
                        'content': f'The query is {question}, and the reasoning paths are: \n{reasoning_path_list}.\n Your choice of top-{width} reasoning paths are:' 
                    }
                ]
    
    if 'gpt' in model_name:
        answer = client.chat.completions.create(
            model = model_name,
            # temperature = 0.4,
            messages = messages
        ).choices[0].message.content
    elif 'llama' in model_name:
        answer = client.run({
            'model': model_name,
            'messages': messages
        }).json()['choices'][0]['message']['content']

    indices = re.findall(r'\d+', answer)
    indices = [int(index) for index in indices]
    # Delete indices that are out of range
    indices = [index for index in indices if index < len(path_list)]
    path_list = [path_list[index] for index in indices]
    return path_list
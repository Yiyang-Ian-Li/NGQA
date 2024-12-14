import os
from dotenv import load_dotenv
load_dotenv()
import argparse
from dataset import Dataset
from model import Retriever, Augmenter, Generator, RetrievalEvaluator
from evaluate import Evaluator

import warnings
import logging
import absl.logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)

def main():
    

    # Define note and method prompts
    note_prompts = {
        "easy": "Important Note: Your output will strictly be Yes or No with no other words.",
        "medium": "Important Note: Your output must be strictly formatted as a comma-separated list of nutrients prefixed with “high” or “low”, based solely on the provided options: carb, protein, sugar, sodium, cholesterol, saturated_fat, calorie. \
            For example, a valid output would be: high_carb, low_protein, high_sugar. No extra words or deviations are allowed.",
        "hard": "Important Note: Your output must consist of “Yes” or “No”, followed by a list of nutrients addressed with “high” or “low,” selected from the following options: carb, protein, sugar, sodium, cholesterol, saturated fat, and calorie. \
            For example, a valid output would be: Yes, because the food is high in carb, low in protein, high in sugar. Ensure the output adheres to this format without any additional words or deviations.",
    }

    method_prompts = {
        "plain": "Below are the extra information you use to answer the question, note that you should not use your general knowledge and the answer is among this information.",
        "KAPING": "Below are the extra information you use to answer the question, note that you should not use your general knowledge and the answer is among this information.",
        "ToG": "Below are the extra information you use to answer the question, note that you should not use your general knowledge and the answer is among this information.", 
        "CoT_Zero": "Below are the extra information you use to answer the question, note that you should not use your general knowledge and the answer is among this information. Let's think step by step to determine the healthiness of the food, by extracting the nutritional properties of the food from the given graph, then comparing them to the nutrition requirements of the health status, dietary need and habits of the user. Do not be too strict with your criteria, since not all nutritional tags are important in determining the food's healthiness.",
        "CoT_BaG": "Below are the extra information you use to answer the question, note that you should not use your general knowledge and the answer is among this information. You will be given the textual description of a directed graph."

    }

    # Initialize dataset
    data = Dataset(args.file_path)
    for question_level in args.question_levels:
        for task_level in args.task_levels:
            for method in args.methods:
                print(f"\nProcessing: Question Level={question_level}, Task Level={task_level}, Method={method}, model={args.model_name}")
                # Process dataset
                questions, answers, graphs = data.process(question_level=question_level, task_level=task_level, sample=args.is_sample, n=args.n)
                # Retrieve subgraphs
                retriever = Retriever(graphs, model_name=args.model_name)
                retrieved_graphs = retriever.retrieve(method=method, api_key=args.api_key, questions=questions)
                # Evaluate retrieval
                retrieval_evaluator = RetrievalEvaluator(graphs)
                retrieval_evaluation_results = retrieval_evaluator.evaluate(retrieved_graphs)
                
                print(f"Retrieval evaluation results for Question Level={question_level}, Task Level={task_level}, Method={method}:")
                for metric, value in retrieval_evaluation_results.items():
                    print(f"{metric}: {value}")
                    
                # Augment graphs to text
                augmenter = Augmenter(args.method)
                textualized_graphs = augmenter.augment(retrieved_graphs)
                # Generate predictions
                note_prompt = note_prompts.get(task_level)
                method_prompt = method_prompts.get(method)
                generator = Generator(api_key=args.api_key, model_name=args.model_name, note_prompt=note_prompt, method_prompt=method_prompt)
                predictions = generator.generate_predictions(questions, textualized_graphs)
                # Evaluate predictions
                evaluator = Evaluator()
                final_output_evaluation_results = evaluator.evaluate(task_level, predictions, answers)

                print(f"Final output evaluation results for Question Level={question_level}, Task Level={task_level}, Method={method}:")
                for metric, value in final_output_evaluation_results.items():
                    print(f"{metric}: {value}")
                

if __name__ == "__main__":
    # Argument parser for hyperparameters
    parser = argparse.ArgumentParser(description="Run multi-level NutriGraphQA benchmark evaluation.")
    parser.add_argument("--file_path", type=str, default="./processed_data/NGQA_benchmark.csv", help="Path to the dataset file.")
    parser.add_argument("--api_key", type=str, 
                        default=os.getenv('API_KEY'), 
                        help="API key for the model.")
    parser.add_argument("--model_name", type=str, 
                        default="gpt-4o-mini",
                        help="Model name for generation.")
    parser.add_argument("--is_sample", type=bool, default=False, help="Whether to sample data or use the full dataset.")
    parser.add_argument("--n", type=int, default=20, help="Number of rows to sample if sampling is enabled.")
    
    parser.add_argument("--task_levels", nargs="+", default=['easy', 'medium', 'hard'], help="List of task levels to evaluate.")
    parser.add_argument("--question_levels", nargs="+", default=['easy', 'medium', 'hard'], help="List of question levels to evaluate.")
    parser.add_argument("--methods", nargs="+", default=["ToG"], help="List of methods to use for retrieval.")
    
    args = parser.parse_args()

    main()
import os
import argparse
from dataset import Dataset
from model import Retriever, Augmenter, Generator
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
        "medium": "Important Note: You output must be strictly, with no extra words, separated by comma, \
            a list of nutrients with high or low before the nutrients among these options: carb, protein, sugar, sodium, cholesterol, \
            saturated_fat, calorie. For example, the output is: high_carb, low_protein, high_sugar.",
        "hard": "Important Note: You output must be a Yes or No followed by strictly a list of nutrients with high or low as prefix among these options: \
            carb, protein, sugar, sodium, cholesterol, saturated fat, calorie. For example, the output is: Yes, because the food is high carb, low protein, high sugar.",
    }

    method_prompts = {
        "plain": "Below are the extra information you use to answer the question, note that you should not use your general knowledge and the answer is among this information.",
        "KAPING": "Below are the extra information you use to answer the question, note that you should not use your general knowledge and the answer is among this information.",
        "zero_cot": "Let's think step by step.",
        "cot_bag": "Let's construct a graph with the nodes and edges first."
    }

    # Initialize dataset
    data = Dataset(args.file_path)
    for question_level in args.question_levels:
        for task_level in args.task_levels:
            for method in args.methods:
                print(f"\nProcessing: Question Level={question_level}, Task Level={task_level}, Method={method}")
                # Process dataset
                questions, answers, graphs = data.process(question_level=question_level, task_level=task_level, sample=args.is_sample, n=args.n)
                # Retrieve subgraphs
                retriever = Retriever(graphs)
                retrieved_graphs = retriever.retrieve(method=method)
                # Augment graphs to text
                augmenter = Augmenter()
                textualized_graphs = augmenter.augment(retrieved_graphs)
                # Generate predictions
                note_prompt = note_prompts.get(task_level, "Default note prompt")
                method_prompt = method_prompts.get(method, "Default method prompt")
                generator = Generator(api_key=args.api_key, model_name=args.model_name, note_prompt=note_prompt, method_prompt=method_prompt)
                predictions = generator.generate_predictions(questions, textualized_graphs)
                # Evaluate predictions
                evaluator = Evaluator()
                results = evaluator.evaluate(task_level, predictions, answers)

                print(f"Results for Question Level={question_level}, Task Level={task_level}, Method={method}:")
                for metric, value in results.items():
                    print(f"{metric}: {value}")
                

if __name__ == "__main__":
    # Argument parser for hyperparameters
    parser = argparse.ArgumentParser(description="Run multi-level NutriGraphQA benchmark evaluation.")
    parser.add_argument("--file_path", type=str, default="../processed_data/NutriGraphQA_benchmark.csv", help="Path to the dataset file.")
    parser.add_argument("--api_key", type=str, default=os.getenv("API_KEY"), help="API key for the model.")
    parser.add_argument("--model_name", type=str, default="llama3.1-70b", help="Model name for generation.")
    parser.add_argument("--is_sample", type=bool, default=True, help="Whether to sample data or use the full dataset.")
    parser.add_argument("--n", type=int, default=100, help="Number of rows to sample if sampling is enabled.")
    
    parser.add_argument("--task_levels", nargs="+", default=["medium"], help="List of task levels to evaluate.")
    parser.add_argument("--question_levels", nargs="+", default=["easy", "medium", "hard"], help="List of question levels to evaluate.")
    parser.add_argument("--methods", nargs="+", default=["plain"], help="List of methods to use for retrieval.")
    
    args = parser.parse_args()

    main()
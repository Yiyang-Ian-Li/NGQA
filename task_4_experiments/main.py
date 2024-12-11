import os
from dotenv import load_dotenv
load_dotenv()
import argparse
from dataset import Dataset
from model import Retriever, Augmenter, Generator, RetrievalEvaluator
from evaluate import Evaluator
from utils import generate_paragraph_cot_bag

import warnings
import logging
import absl.logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)

def main():
    

    # Define note and method prompts
    note_prompts = {
        "easy": "Important Note: Your output will strictly be Yes or No with no other words or punctuation marks.",
        "medium": "Important Note: Your output must be strictly, with no extra words, separated by comma, \
            a list of nutrients with high or low before the nutrients among these options: carb, protein, sugar, sodium, cholesterol, \
            saturated_fat, calorie. For example, the output is: high_carb, low_protein, high_sugar.",
        "hard": "Important Note: Your output must be a Yes or No followed by strictly a list of nutrients with high or low as prefix among these options: \
            carb, protein, sugar, sodium, cholesterol, saturated fat, calorie. For example, the output is: Yes, because the food is high in carb, low in protein, high in sugar.",
    }

    method_prompts = {
        "plain": "Below are the extra information you use to answer the question, note that you should not use your general knowledge and the answer is among this information.",
        "KAPING": "Below are the extra information you use to answer the question, note that you should not use your general knowledge and the answer is among this information.",
        "ToG": "Below are the extra information you use to answer the question, note that you should not use your general knowledge and the answer is among this information.", 
        "Zero_CoT": "Let's think step by step",
        "CoT_BaG": ""
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
                augmenter = Augmenter()
                textualized_graphs = augmenter.augment(retrieved_graphs)

                # For CoT_BaG, convert textualized graphs into paragraph format
                if method == "CoT_BaG":
                    textualized_graphs = [generate_paragraph_cot_bag(graph) for graph in textualized_graphs]
                
                # Generate predictions
                note_prompt = note_prompts.get(task_level, "Default note prompt")
                method_prompt = method_prompts.get(method, "Default method prompt")
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
                        # default=os.getenv('LLAMA_API_KEY'), 
                        default=os.getenv('OPENAI_API_KEY'),
                        help="API key for the model.")
    parser.add_argument("--model_name", type=str, 
                        # default="llama3.1-70b", 
                        # default="gpt-3.5-turbo", 
                        default="gpt-4o-mini",
                        help="Model name for generation.")
    parser.add_argument("--is_sample", type=bool, default=True, help="Whether to sample data or use the full dataset.")
    parser.add_argument("--n", type=int, default=20, help="Number of rows to sample if sampling is enabled.")
    
    parser.add_argument("--task_levels", nargs="+", default=["hard"], help="List of task levels to evaluate.")
    parser.add_argument("--question_levels", nargs="+", default=['easy', 'medium', 'hard'], help="List of question levels to evaluate.")
    parser.add_argument("--methods", nargs="+", default=["plain"], help="List of methods to use for retrieval.")
    
    args = parser.parse_args()

    main()

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from rouge_score import rouge_scorer
from bert_score import score
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu

import warnings
warnings.filterwarnings("ignore")

class Evaluator:
    """
    A unified class for evaluating predictions across different levels of tasks (easy, medium, hard).

    Attributes:
        tags (list): List of tags for multi-label classification.
        rouge_scorer (RougeScorer): Scorer for ROUGE metrics.

    Use Example: 
        evaluator = Evaluator()
        easy_metrics = evaluator.evaluate(level="easy", predictions=easy_predictions, ground_truths=easy_ground_truths)
        for metric, value in easy_metrics.items():
            print(f"{metric}: {value}")
    """

    def __init__(self, tags=None):
        """
        Initialize the Evaluator with optional tags for multi-label tasks.

        Args:
            tags (list): List of tags for multi-label classification. Defaults to a predefined set of tags.
        """
        self.tags = tags or [
            'low_carb', 'high_carb',
            'low_sugar', 'high_sugar',
            'low_sodium', 'high_sodium',
            'low_calorie', 'high_calorie',
            'low_protein', 'high_protein',
            'low_cholesterol', 'high_cholesterol',
            'low_saturated_fat', 'high_saturated_fat',
        ]
        # Initialize ROUGE scorer for hard-level tasks
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    @staticmethod
    def _preprocess_labels(text):
        """
        Preprocess multi-label text into a list of tags.

        Args:
            text (str): Comma-separated string of tags.

        Returns:
            list: List of processed tags.
        """
        if pd.isna(text):
            return []
        return [tag.strip() for tag in text.split(',')]

    @staticmethod
    def _format_metrics(metrics):
        """
        Format all metrics to 4 decimal places.

        Args:
            metrics (dict): Dictionary of evaluation metrics.

        Returns:
            dict: Dictionary with formatted metric values.
        """
        formatted_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, float) or isinstance(value, np.float64):  # Format float-like metrics
                formatted_metrics[key] = round(float(value), 4)
            elif isinstance(value, str):  # Preserve string values like "N/A"
                formatted_metrics[key] = value
            else:
                formatted_metrics[key] = value  # Preserve other types (if any)
        return formatted_metrics

    def evaluate_hard(self, predictions, ground_truths):
        """
        Evaluate predictions for hard tasks using ROUGE, BERT, and BLEU metrics.

        Args:
            predictions (list or pd.Series): Predicted text outputs.
            ground_truths (list or pd.Series): Ground truth text outputs.

        Returns:
            dict: Dictionary of evaluation metrics (ROUGE-1, ROUGE-2, ROUGE-L, BERT, BLEU).
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have the same length.")

        # Initialize metric storage
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        bert_scores = []
        bleu_scores = []

        # Compute metrics for each pair of prediction and ground truth
        for pred, truth in zip(predictions, ground_truths):
            # ROUGE scores
            rouge = self.rouge_scorer.score(truth, pred)
            for key in rouge_scores:
                rouge_scores[key].append(rouge[key].fmeasure)

            # BERT scores
            _, _, F1 = score([pred], [truth], lang='en', verbose=False)
            bert_scores.append(F1.item())

            # BLEU scores
            ref_tokens = [word_tokenize(truth)]
            pred_tokens = word_tokenize(pred)
            try:
                bleu_scores.append(sentence_bleu(ref_tokens, pred_tokens))
            except ZeroDivisionError:
                bleu_scores.append(0)
        
        results = {
            "ROUGE-1": np.mean(rouge_scores['rouge1']),
            "ROUGE-2": np.mean(rouge_scores['rouge2']),
            "ROUGE-L": np.mean(rouge_scores['rougeL']),
            "BERT": np.mean(bert_scores),
            "BLEU": np.mean(bleu_scores),
        }

        # Aggregate results
        return self._format_metrics(results)
    
    def _weighted_accuracy(self, y_true_bin, y_pred_bin):
        """
        Compute instance-based weighted accuracy for multi-label classification.

        Args:
            y_true_bin (np.ndarray): Ground truth binary matrix (shape: n_samples x n_classes).
            y_pred_bin (np.ndarray): Predicted binary matrix (shape: n_samples x n_classes).

        Returns:
            float: Instance-based weighted accuracy.
        """
        # Calculate per-instance accuracy
        correct_per_instance = (y_true_bin & y_pred_bin).sum(axis=1)  # Intersection: Correct labels
        total_labels_per_instance = (y_true_bin | y_pred_bin).sum(axis=1)  # Union: Total relevant labels

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            instance_accuracies = np.divide(correct_per_instance, total_labels_per_instance)
            instance_accuracies[np.isnan(instance_accuracies)] = 0  # Set undefined accuracies to 0

        # Return the average accuracy across all instances
        return instance_accuracies.mean()

    def evaluate_medium(self, predictions, ground_truths):
        """
        Evaluate predictions for medium tasks using multi-label classification metrics.

        Args:
            predictions (list or pd.Series): Predicted multi-label outputs (as text).
            ground_truths (list or pd.Series): Ground truth multi-label outputs (as text).

        Returns:
            dict: Dictionary of evaluation metrics (Accuracy, Precision, Recall, F1 Score, ROC AUC).
        """
        # Preprocess predictions and ground truths into tag lists
        y_true = predictions.apply(self._preprocess_labels)
        y_pred = ground_truths.apply(self._preprocess_labels)

        # Convert to binary format using MultiLabelBinarizer
        mlb = MultiLabelBinarizer(classes=self.tags)
        y_true_bin = mlb.fit_transform(y_true)
        y_pred_bin = mlb.transform(y_pred)

        # Compute metrics
        results = {
            "Accuracy": self._weighted_accuracy(y_true_bin, y_pred_bin),
            "Precision": precision_score(y_true_bin, y_pred_bin, average='weighted', zero_division=0),
            "Recall": recall_score(y_true_bin, y_pred_bin, average='weighted', zero_division=0),
            "F1 Score": f1_score(y_true_bin, y_pred_bin, average='weighted', zero_division=0),
        }

        # ROC AUC score, if applicable
        try:
            results["ROC AUC"] = roc_auc_score(y_true_bin, y_pred_bin, average='weighted', multi_class='ovr')
        except ValueError:
            results["ROC AUC"] = np.nan

        return self._format_metrics(results)

    def evaluate_easy(self, predictions, ground_truths):
        """
        Evaluate predictions for easy tasks using binary classification metrics.

        Args:
            predictions (list or pd.Series): Predicted binary outputs (e.g., 'Yes' or 'No').
            ground_truths (list or pd.Series): Ground truth binary outputs (e.g., 'Yes' or 'No').

        Returns:
            dict: Dictionary of evaluation metrics (Accuracy, Precision, Recall, F1 Score, AUC Score).
        """
         # Map text labels to binary values, marking invalid predictions as -1
        valid_mapping = {'Yes': 1, 'No': 0}
        y_true = ground_truths.map(valid_mapping).astype(int)
        y_pred = predictions.map(valid_mapping).fillna(-1).astype(int) # In case yes or no is not in the answer. 

        if len(y_true) != len(y_pred):
            raise ValueError("Predictions and ground truths must have the same length.")

        # Treat -1 predictions as always wrong (regardless of y_true)
        incorrect_indices = (y_pred == -1)
        y_pred[incorrect_indices] = 1 - y_true[incorrect_indices]  # Flip y_true to mark incorrectness

        # Compute metrics
        results = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        }

        # AUC score, if applicable
        if len(set(y_true)) > 1:
            results["AUC Score"] = roc_auc_score(y_true, y_pred)
        else:
            results["AUC Score"] = "N/A (only one class in ground truth)"

        return self._format_metrics(results)

    def evaluate(self, level, predictions, ground_truths):
        """
        Unified evaluation method for all task levels.

        Args:
            level (str): Task level ('easy', 'medium', 'hard').
            predictions (list or pd.Series): Predicted outputs.
            ground_truths (list or pd.Series): Ground truth outputs.

        Returns:
            dict: Dictionary of evaluation metrics.
        """
        predictions = pd.Series(predictions)
        ground_truths = pd.Series(ground_truths)

        if level == "easy":
            return self.evaluate_easy(predictions, ground_truths)
        elif level == "medium":
            return self.evaluate_medium(predictions, ground_truths)
        elif level == "hard":
            return self.evaluate_hard(predictions, ground_truths)
        else:
            raise ValueError(f"Unknown evaluation level: {level}")
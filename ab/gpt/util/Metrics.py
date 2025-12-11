import evaluate
import numpy as np


class Metrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def compute_bleu(self, eval_preds):
        bleu_metric = evaluate.load('bleu')
        predictions, labels = eval_preds

        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # BLEU expects tokenized sentences
        pred_tokens = [pred.split() for pred in decoded_preds]
        label_tokens = [[label.split()] for label in decoded_labels]  # list of references

        result = bleu_metric.compute(predictions=pred_tokens, references=label_tokens)
        return {'bleu': result['bleu']}

    def init_compute_metrics(self, test_metric: str | None):
        metric_switch = {'bleu': self.compute_bleu}
        if test_metric:
            if test_metric not in metric_switch: raise ValueError(f'Unsupported metric: {test_metric}')
            return metric_switch[test_metric]
        else:
            return None

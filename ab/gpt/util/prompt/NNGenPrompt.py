import json

import ab.nn as lemur
from overrides import override
import pandas as pd
from pandas import DataFrame
from transformers import PreTrainedTokenizerBase

from ab.gpt.util.prompt.Prompt import Prompt
from tqdm import tqdm

from ab.nn.api import JoinConf


def shuffle_data(df: DataFrame):
    return df.sample(frac=1).reset_index(drop=True)


class NNGenPrompt(Prompt):
    """
    Assumes the existence of accuracies.json and folder-based dataset
    """

    def __init__(self, max_len: int, tokenizer: PreTrainedTokenizerBase, prompts_path):
        super().__init__(max_len, tokenizer)
        self.prompts_path = prompts_path

    @override
    def get_raw_dataset(self, only_best_accuracy, n_training_prompts=None) -> DataFrame:
        """
        :return:
            pandas.Dataframe object with columns described in nn_api.data()
        """
        prompt_lists = []

        with open(self.prompts_path) as prompt_file:
            prompt_dict = json.load(prompt_file)
            assert isinstance(prompt_dict, dict)

        for key in prompt_dict.keys():
            dataframe = DataFrame(columns=['instruction', 'context', 'response', 'category', 'text'])
            prompt_lists.append(dataframe)
            prompt = '\n'.join(prompt_dict[key]['prompt'])
            print('Preparing Data...', flush=True)
            key_dict = prompt_dict[key]
            num_joint_nns = key_dict.get('num_joint_nns') or 1
            data = lemur.data(only_best_accuracy=only_best_accuracy, task=key_dict.get('task'),
                              nn_prefixes=tuple(key_dict.get('nn_prefixes')), max_rows=n_training_prompts,
                              sql=None if num_joint_nns < 2 else JoinConf(num_joint_nns=num_joint_nns,
                                                                          same_columns=tuple(key_dict.get('keep_same')),
                                                                          diff_columns=tuple(key_dict.get('no_repeat')),
                                                                          enhance_nn=key_dict.get('improve')))
            print('Data acquisition complete', flush=True)
            for _, row in tqdm(data.iterrows(), total=n_training_prompts or len(data)):
                if n_training_prompts and len(dataframe) >= n_training_prompts:
                    break
                para_dict = dict()
                for it in prompt_dict[key]['input_list']:
                    para_dict[it['para']] = row[it['value']]
                inst = prompt.format(**para_dict)
                # Having the same name prefix before '-'
                output = '\n'.join(prompt_dict[key]['output'])
                response = output.format(**para_dict)
                text = self.tokenizer.apply_chat_template(
                    [
                        {'role': 'user', 'content': inst},
                        {'role': 'assistant', 'content': response}
                    ], tokenize=False
                )

                # print(f"Prompt: {inst}")
                # print(f"Output: {response}")

                dataframe.loc[len(dataframe)] = [inst, "", response, "", text]

        print('Prompts successfully generated', flush=True)
        del data
        return pd.concat(prompt_lists, ignore_index=True)

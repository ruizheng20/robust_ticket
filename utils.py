import logging
import datasets
from torch.utils.data import Dataset

import torch
from torch.nn.utils.rnn import pad_sequence

task_to_keys = {
    "ag_news": ("text", None),
    "imdb": ("text", None),
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
MAX_CONTEXT_LEN = 50
MAX_SEQ_LEN = 512
logger = logging.getLogger(__name__)


def pad_squeeze_sequence(sequence, *args, **kwargs):
    """Squeezes fake batch dimension added by tokenizer before padding sequence."""
    return pad_sequence([x.squeeze(0) for x in sequence], *args, **kwargs)


class OutputStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """

    def __init__(self, module):
        self._stored_output = None
        module.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        self._stored_output = output

    def get(self):
        return self._stored_output


class ExponentialMovingAverage:
    def __init__(self, weight=0.3):
        self._weight = weight
        self.reset()

    def update(self, x):
        self._x += x
        self._i += 1

    def reset(self):
        self._x = 0
        self._i = 0

    def get_metric(self):
        return self._x / (self._i + 1e-13)


class Collator:
    """
    Collates transformer outputs.
    """

    def __init__(self, pad_token_id=0):
        self._pad_token_id = pad_token_id

    def __call__(self, features):
        # Separate the list of inputs and labels
        model_inputs, labels = list(zip(*features))
        # Assume that all inputs have the same keys as the first
        proto_input = model_inputs[0]
        keys = list(proto_input.keys())
        padded_inputs = {}
        for key in keys:
            if key == 'input_ids':
                padding_value = self._pad_token_id
            else:
                padding_value = 0
            # NOTE: We need to squeeze to get rid of fake batch dim.
            sequence = [x[key] for x in model_inputs]
            padded = pad_squeeze_sequence(sequence, batch_first=True, padding_value=padding_value)
            padded_inputs[key] = padded
        labels = torch.tensor([x for x in labels])
        return padded_inputs, labels


class Huggingface_dataset(Dataset):
    def __init__(
            self,
            args,
            tokenizer,
            name_or_dataset: str,
            subset: str = None,
            split="train",
            shuffle=False,
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.name = name_or_dataset
        self.subset = subset
        self.dataset = datasets.load_dataset(self.name, subset)[split]
        if subset is not None:
            self.input_columns = task_to_keys[subset]
        else:
            self.input_columns = task_to_keys[name_or_dataset]
        self.shuffled = shuffle
        if shuffle:
            self.dataset.shuffle()

    def _format_examples(self, examples):
        """
        Only for some task which has ONE input column, such as SST-2 and IMDB, NOT work for NLI such as MRPC.
        """
        text = [examples[self.input_columns[0]]]
        sentence = "".join(text)
        inputs = self.tokenizer(sentence, truncation=True, max_length=self.args.max_seq_length, return_tensors='pt')
        output = int(examples['label'])
        return (inputs, output)

    def shuffle(self):
        self.dataset.shuffle()
        self.shuffled = True

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        """Return i-th sample."""
        if isinstance(i, int):
            return self._format_examples(self.dataset[i])
        else:
            # `idx` could be a slice or an integer. if it's a slice,
            # return the formatted version of the proper slice of the list
            return [
                self._format_examples(self.dataset[j]) for j in range(i.start, i.stop)
            ]
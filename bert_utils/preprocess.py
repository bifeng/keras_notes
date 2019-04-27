import pandas as pd
from pathlib import Path
import csv


def _read_data(path):
    table = pd.read_csv(path, sep='\t', header=0, quoting=csv.QUOTE_NONE)
    df = pd.DataFrame({
        'text_left': table['Question'],
        'text_right': table['Sentence'],
        'id_left': table['QuestionID'],
        'id_right': table['SentenceID'],
        'label': table['Label']
    })
    return df


def load_data(data_root, stage: str = 'train', task: str = 'ranking', filter: str = False
              ):
    """
    Load WikiQA data.

    :param stage: One of `train`, `dev`, and `test`.
    :param task: Could be one of `ranking`, `classification` or a
        :class:`matchzoo.engine.BaseTask` instance.
    :param filter: Whether remove the questions without correct answers.
    :return: A DataPack if `ranking`, a tuple of (DataPack, classes) if
        `classification`.
    """
    if stage not in ('train', 'dev', 'test'):
        raise ValueError(f"{stage} is not a valid stage."
                         f"Must be one of `train`, `dev`, and `test`.")
    data_root = Path(data_root)
    file_path = data_root.joinpath(f'WikiQA-{stage}.tsv')
    data_pack = _read_data(file_path)
    if filter and stage in ('dev', 'test'):
        ref_path = data_root.joinpath(f'WikiQA-{stage}.ref')
        filter_ref_path = data_root.joinpath(f'WikiQA-{stage}-filtered.ref')
        with open(filter_ref_path, mode='r') as f:
            filtered_ids = set([line.split()[0] for line in f])
        filtered_lines = []
        with open(ref_path, mode='r') as f:
            for idx, line in enumerate(f.readlines()):
                if line.split()[0] in filtered_ids:
                    filtered_lines.append(idx)
        data_pack = data_pack[filtered_lines]

    return data_pack

#
# train = load_data('train')
# dev = load_data('dev')
# test = load_data('test')

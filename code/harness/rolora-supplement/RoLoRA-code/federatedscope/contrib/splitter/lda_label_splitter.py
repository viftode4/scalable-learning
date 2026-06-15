"""LDA (Dirichlet) splitter that skews over the TASK label, not the context.

Why this exists
---------------
The generic `lda` splitter reads ``x['categories']`` (lda_splitter.py:28).
For the LLM datasets here, ``categories`` is ``pd.Categorical(df["category"])
.codes`` (llm_dataset.py:125) -- i.e. the integer code of the *context
paragraph*, which is unique per example. On QNLI that is ~105k distinct
"classes", so ``dirichlet_distribution_noniid_slice`` iterates
``for k in range(classes)`` ~105k times per pass and effectively hangs (a real
5h+ stall was observed). It is also semantically wrong: heterogeneity for QNLI
should be over the answer label (entailment vs not), i.e. ``x['labels']`` in
{0,1}, not over which paragraph an example came from.

This splitter extracts the scalar task label per item and remaps the unique
labels to a contiguous 0..C-1 range, which is exactly the contract
``dirichlet_distribution_noniid_slice`` assumes. With 2 classes it is fast and
produces genuine label skew controlled by ``alpha`` (smaller = more skew).

Opt in with ``data.splitter: 'lda_label'`` and
``data.splitter_args: [{'alpha': <a>}]``. The default ``lda``/``iid`` paths and
every prior result are untouched.
"""

import numpy as np

from federatedscope.register import register_splitter
from federatedscope.core.splitters import BaseSplitter
from federatedscope.core.splitters.utils import \
    dirichlet_distribution_noniid_slice


def _scalar_label(item):
    """Return a single int class id from a dataset item.

    Handles dict items with a scalar ``labels`` (classification head), and is
    defensive about 0-d tensors / 1-element sequences. Raises on token-level
    label sequences so we fail loudly instead of mis-splitting.
    """
    if isinstance(item, dict):
        if 'labels' in item:
            val = item['labels']
        elif 'categories' in item:
            val = item['categories']
        else:
            raise KeyError(
                "lda_label splitter needs 'labels' (or 'categories') on each "
                f"item; got keys {list(item.keys())}")
    elif isinstance(item, (tuple, list)) and len(item) >= 2:
        val = item[1]
    else:
        val = item

    # Coerce common scalar containers (python int, numpy scalar, 0-d / 1-elem
    # torch tensor) to a python int.
    if hasattr(val, 'item'):
        try:
            return int(val.item())
        except (ValueError, RuntimeError) as err:
            raise ValueError(
                "lda_label splitter expected a scalar class label but got a "
                f"non-scalar tensor (shape "
                f"{getattr(val, 'shape', '?')}); this dataset uses token-level "
                "labels and needs a classification target to split on.") \
                from err
    if isinstance(val, (list, tuple, np.ndarray)):
        arr = np.asarray(val).ravel()
        if arr.size != 1:
            raise ValueError(
                "lda_label splitter expected a scalar class label per item "
                f"but got a sequence of length {arr.size}.")
        return int(arr[0])
    return int(val)


class LDALabelSplitter(BaseSplitter):
    """Dirichlet split over the scalar task label.

    Args:
        client_num: number of clients.
        alpha (float): Dirichlet concentration; smaller -> more heterogeneous.
    """

    def __init__(self, client_num, alpha=0.5, **kwargs):
        self.alpha = alpha
        super(LDALabelSplitter, self).__init__(client_num)

    def __call__(self, dataset, prior=None, **kwargs):
        from torch.utils.data import Dataset, Subset

        tmp_dataset = [ds for ds in dataset]
        raw = np.array([_scalar_label(x) for x in tmp_dataset])

        # Remap whatever the labels are to a contiguous 0..C-1 range, which is
        # the exact contract dirichlet_distribution_noniid_slice assumes
        # (`for k in range(classes)` then `label == k`). This makes the slice
        # both correct and impossible to hang on sparse/non-zero-based ids.
        _, label = np.unique(raw, return_inverse=True)

        idx_slice = dirichlet_distribution_noniid_slice(
            label, self.client_num, self.alpha, prior=prior)
        if isinstance(dataset, Dataset):
            data_list = [Subset(dataset, idxs) for idxs in idx_slice]
        else:
            data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice]
        return data_list

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha})'


def call_lda_label_splitter(splitter_type, client_num, **kwargs):
    if splitter_type == 'lda_label':
        return LDALabelSplitter(client_num, **kwargs)
    return None


register_splitter('lda_label', call_lda_label_splitter)

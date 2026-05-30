# OPHD

We evaluate the proposed model using real-world healthcare data and leverage the national [All of Us Research Platform](https://www.researchallofus.org/) and [OHSU EHR data warehouse](https://www.ohsu.edu/octri/powering-innovation-and-translational-science-octri-informatics).

## Environment

The code is written in Python and uses PyTorch, Hugging Face Transformers, PEFT, TRL, Datasets, scikit-learn, NumPy, and tqdm.

Example setup:

```bash
pip install torch transformers peft trl datasets scikit-learn numpy tqdm
```

## Example Usage

Running commands are provided in the experiment scripts. The typical workflow is:

```bash
python train_robust.py
python eval_generator_scorer.py
```

## Update

The code will be further organized and refactored upon acceptance.
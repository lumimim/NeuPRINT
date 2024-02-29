# NeuPRINT
Official implementation of Neuronal Time-Invariant Representations (NeuPRINT).

Learning Time-Invariant Representations for Individual Neurons from Population Dynamics. NeurIPS 2023

Lu Mi*, Trung Le*, Tianxing He, Eli Shlizerman, Uygar Sümbül

![intro_16](https://github.com/lumimim/NeuPRINT/assets/41974416/48e72abb-13af-42cf-87ac-145700024755)

## Datasets
Our framework is evaluated on the datasets from [Bugeon et al. 2022, Nature](https://www.nature.com/articles/s41586-022-04915-7) (A transcriptomic axis predicts state modulation of cortical interneurons), download dataset from this [link](https://figshare.com/articles/dataset/A_transcriptomic_axis_predicts_state_modulation_of_cortical_interneurons/19448531).

## Environment Setup
Assuming you have Python 3.8+ and Miniconda installed, run the following to set up the environment with necessary dependencies:
```
conda env create -f environment.yml
```

## Run Experiments

```
python main.py --exp-tag neuprint_train
```

## Citations
If you find our code helpful, please cite our paper:

```
@article{mi2024learning,
  title={Learning Time-Invariant Representations for Individual Neurons from Population Dynamics},
  author={Mi, Lu and Le, Trung and He, Tianxing and Shlizerman, Eli and S{\"u}mb{\"u}l, Uygar},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```
If you use the dataset, please cite this paper:

```
@article{bugeon2022transcriptomic,
  title={A transcriptomic axis predicts state modulation of cortical interneurons},
  author={Bugeon, Stephane and Duffield, Joshua and Dipoppa, Mario and Ritoux, Anne and Prankerd, Isabelle and Nicoloutsopoulos, Dimitris and Orme, David and Shinn, Maxwell and Peng, Han and Forrest, Hamish and others},
  journal={Nature},
  volume={607},
  number={7918},
  pages={330--338},
  year={2022},
  publisher={Nature Publishing Group UK London}
}
```


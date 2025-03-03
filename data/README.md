The detailed description of the data directory can be found below. For demonstrative purposes, only 1010 ppm CH<sub>4</sub> concentration was taken to group data by sequences (see `data/seq_grouped/`) and form the datasets (see `data/datasets/`).

```text
The structure if the data directory:
.
├── README.md
├── preprocess.ipynb       \\ Notebook with the whole data preprocessing flow with examples
├── raw/                   \\ Raw experimental data
├── smoothed/              \\ Data with smoothed time series
├── computed_derivatives/  \\ Data with computed derivatives
├── normalized/            \\ Normalized data
├── seq_grouped/
  └── CH4_1010_ppm         \\ Data grouped sequentially for sequence length from 1 to 10 (for 1010 ppm CH4)
└── datasets/
  └── CH4_1010_ppm/        \\ Data split into train/valid/test sets for each sequence length (for 1010 ppm CH4)
```

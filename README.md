**OADlib** library is a set of tools that can be useful in reconstructing a gas cell's resonance frequency by the data of its temperature with the use of LSTM-based neural networks.

See `OADlib/data/` to find details on the data preprocessing flow.

See `OADlib/optuna_example/` to find details on the hyperparameters optimization process.

```text
The structure of the OADlib modules:


The structure if the data directory:
.
├── data/
  ├── README.md
  ├── preprocess.ipynb       \\ Notebook with the whole data preprocessing flow with examples
  ├── raw/                   \\ Raw experimental data
  ├── smoothed/              \\ Data with smoothed time series
  ├── computed_derivatives/  \\ Data with computed derivatives
  ├── normalized/            \\ Normalized data
  ├── seq_grouped/           \\ Data grouped sequentially for sequence length from 1 to 10
  └── datasets/              \\ Data split into train/valid/test sets for each sequence length


The structure of the optuna examples directory:
.
├── optuna_examples/
  ├── README.md
  └── tuning.py  \\ The example of hyperparameters tuning with optuna library for LSTMFullyConnected model


The structure if the OAD dataset module:
.
├── dataset/
  ├── __init__.py
  ├── README.md
  └── oad_dataset.py  \\ The implementation of a OAD dataset class


The structure if the logging module:
.
├── logging/
  ├── __init__.py
  ├── README.md
  └── logger_manager.py  \\ The implementation of a logger manager class


The structure if the LSTM-based models module:
.
├── models/
  ├── __init__.py
  ├── README.md
  ├── lstm_linear_regression.py   \\ The implementation of a LSTM + Linear Regression model
  ├── lstm_fully_connected.py     \\ The implementation of a LSTM + Fully-Connected model
  ├── lstm_self_attention.py      \\ The implementation of a LSTM + Self-Attention model
  └── src/                        \\ Source files with implementations of LSTM block and several head blocks


The structure if the data preprocessing module:
.
└── preprocessing/
  ├── __init__.py
  ├── README.md
  ├── data_preprocessing.py      \\ The implementation of a data preprocessing class 
  ├── derivative_computation/    \\ The implementations of derivative computation algorithms
  └── smoothing/                 \\ The implementations of data smoothing algorithms
```

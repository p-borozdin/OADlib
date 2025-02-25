**OADlib** library is a set of tools that can be useful in reconstructing a gas cell's resonance frequency by the data of its temperature with the use of LSTM-based neural networks.
```text
The structure of the OADlib modules:


The structure if the data module:
.
└── data/
  ├── raw/  \\ Raw experimental data


The structure if the OAD dataset module:
.
└── dataset/
  ├── __init__.py
  ├── README.md
  └── oad_dataset.py  \\ The implementation of a OAD dataset class


The structure if the logging module:
.
└── logging/
  ├── __init__.py
  ├── README.md
  └── logger_manager.py  \\ The implementation of a logger manager class


The structure if the LSTM-based models module:
.
└── models/
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

```text
The structure if the LSTM-based models module.
.
├── __init__.py
├── README.md
├── lstm_linear_regression.py     \\ The implementation of a LSTMLinearRegression class
├── lstm_fully_connected.py       \\ The implementation of a LSTMFullyConnected class
├── lstm_self_attention.py        \\ The implementation of a LSTMSelfAttention class
└── src/                          \\ Source files with implementations of LSTM block and several head blocks
  ├── lstm_block.py               \\ The implementation of a LSTMBlock block
  ├── linear_regression_head.py   \\ The implementation of a LinearRegressionHead block
  ├── fully_connected_head.py     \\ The implementation of a FullyConnectedHead block
  └── self_attention_head.py      \\ The implementation of a SelfAttentionHead block
```

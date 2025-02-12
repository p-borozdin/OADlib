The structure of the module with implementations of the LSTM-based models and their elements.

```text
.
└── models/
  ├── __init__.py
  ├── README.md                   \\ The module's documentation you are reading right now
  ├── lstm_block.py               \\ The implementation of a LSTM block common for all the models' architectures 
  ├── self_attention_part.py      \\ The implementation of a Self-Attention part of the LSTM with Self-Attention head
  ├── base_lstm_model.py          \\ The implementation of a BaseLSTMModel base class
  ├── lstm_linear_regression.py   \\ The implementation of a LSTMLinearRegresion model
  ├── lstm_fully_connected.py     \\ The implementation of a LSTMFullyConnected model
  ├── lstm_self_attention.py      \\ The implementation of a LSTMSelfAttention model
```

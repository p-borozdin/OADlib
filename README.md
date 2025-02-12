**OADlib** library provides the implementations of some LSTM-based deep learning models that can be helpful in regression tasks dealing with time series processing.
Each model consists of two blocks: **lstm** block and **head** block. The models differ in their head blocks, when the lstm block is kept the same for all the implemented models.

The implementations of the models can be found in `OADlib/models` direcotry. For now the next LSTM-based models are implemented:
* LSTM with Linear Regression head
* LSTM with Fully-Connected head
* LSTM with Self-Attention head

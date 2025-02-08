""" The Module provides the architecture of an LSTM-based model
with the Linear Regression block after the LSTM block
"""
import torch

from lstm_block import LSTMBlock


class LSTMLinearRegression(torch.nn.Module):
    """ LSTM-based model with Linear Regression block as its head.\n
    The class inherits from torch.nn.Module
    """
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            device: torch.device
            ):
        """ Model's initialization

        Args:
            input_size (int): number of consecutive points used to make a \
                  prediction
            hidden_size (int): size of hidden state in a LSTM cell
            device (torch.device): CPU or GPU torch.device
        """
        super(LSTMLinearRegression, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        self.blocks = torch.nn.ParameterDict()
        self.init_blocks()

    def init_blocks(self):
        """ Initialization of the model's blocks (LSTM and head blocks)
        """
        blocks = {}

        blocks['lstm'] = self._init_lstm_block()
        blocks['head'] = self._init_head_block()

        self.blocks = torch.nn.ParameterDict(blocks)

    def _init_lstm_block(self) -> torch.nn.Module:
        """ Initialize the model's LSTM block

        Returns:
            torch.nn.Module: LSTM block for the model
        """
        return LSTMBlock(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            device=self.device
        )

    def _init_head_block(self) -> torch.nn.Module:
        """ Initialize the model's head block

        Returns:
            torch.nn.Module: head block for the model
        """
        head_layers: list[torch.nn.Module] = []

        head_layers.append(torch.nn.ReLU())
        head_layers.append(
            torch.nn.Linear(
                in_features=self.hidden_size,
                out_features=1
                )
            )

        return torch.nn.ParameterList(head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass for LSTM-based model with Linear Regression block

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        lstm_output = self.blocks['lstm'](x)
        output = lstm_output[:, -1, :]
        for layer in self.blocks['head']:
            output = layer(output)

        return output

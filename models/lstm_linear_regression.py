""" The Module provides the architecture of an LSTM-based model
with the Linear Regression block after the LSTM block
"""
import torch

from OADlib.models.src.lstm_block import LSTMBlock
from OADlib.models.src.linear_regression_head import LinearRegressionHead


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
            input_size (int): number of input features
            hidden_size (int): size of hidden state in a LSTM cell
            device (torch.device): CPU or GPU torch.device
        """
        super(LSTMLinearRegression, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        self.blocks = torch.nn.ParameterDict()
        self.__init_blocks()

    def __init_blocks(self):
        """ Initialize the model's blocks
        (LSTM and Linear Regression head blocks)
        """
        blocks = {}

        blocks['lstm'] = LSTMBlock(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            device=self.device
        )
        blocks['head'] = LinearRegressionHead(
            in_features=self.hidden_size
        )

        self.blocks = torch.nn.ParameterDict(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass for the model

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        lstm_output = self.blocks['lstm'](x)
        head_output = self.blocks['head'](lstm_output)

        return head_output

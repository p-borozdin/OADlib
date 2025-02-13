""" The Module provides the architecture of an LSTM-based model
with the Fully Connected block after the LSTM block
"""
import torch

from OADlib.models.src.lstm_block import LSTMBlock
from OADlib.models.src.fully_connected_head import FullyConnectedHead


class LSTMFullyConnected(torch.nn.Module):
    """ LSTM-based model with Fully Connected block as its head.\n
    The class inherits from torch.nn.Module
    """
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            hidden_layer_size: int,
            activation_func: torch.nn.Module,
            device: torch.device
            ):
        """ Model's initialization

        Args:
            input_size (int): number of input features
            hidden_size (int): size of hidden state in a LSTM cell
            hidden_layer_size (int): size of a hidden fully connected layer
            activation_func (torch.nn.Module): hidden layers' activation \
                function
            device (torch.device):CPU or GPU torch.device
        """
        super(LSTMFullyConnected, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layer_size = hidden_layer_size
        self.activation_func = activation_func
        self.device = device

        self.blocks = torch.nn.ParameterDict()
        self.__init_blocks()

    def __init_blocks(self):
        """ Initialize the model's blocks
        (LSTM and Fully-Connected head blocks)
        """
        blocks = {}

        blocks['lstm'] = LSTMBlock(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            device=self.device
        )
        blocks['head'] = FullyConnectedHead(
            in_features=self.hidden_size,
            hidden_layer_size=self.hidden_layer_size,
            activation_func=self.activation_func
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

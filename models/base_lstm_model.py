""" The Module provides the architecture of a base LSTM-based model
"""
import torch

from OADlib.models.lstm_block import LSTMBlock


class BaseLSTMModel(torch.nn.Module):
    """ Base class for all LSTM-based models.\n
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
        super(BaseLSTMModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        self.blocks = torch.nn.ParameterDict()

    def _init_blocks(self):
        """ Initialization of the model's blocks (LSTM and head blocks).\n
        Must be called in constructor
        after all the member fields are initialized
        """
        blocks = {}

        blocks['lstm'] = self._init_lstm_block()
        blocks['head'] = self._init_head_block()

        self.blocks = torch.nn.ParameterDict(blocks)

    def _init_lstm_block(self) -> torch.nn.Module:
        """ Initialize the model's LSTM block.\n
        Optionally can be overriden in derived classes.

        Returns:
            torch.nn.Module: LSTM block for the model
        """
        return LSTMBlock(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            device=self.device
            )

    def _init_head_block(self) -> torch.nn.Module:
        """ Initialize the model's head block.\n
        Virtual method, must be overriden in all derived classes.

        Raises:
            NotImplementedError: method was called from the base class

        Returns:
            torch.nn.Module: head block for the model
        """
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Default forward pass for LSTM-based model,
        when only the last LSTM cell's results are taken.\n
        Optionally can be overriden in derived classes.

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

""" The Module provides the architecture of an LSTM block
"""
import torch
from torch.autograd import Variable


class LSTMBlock(torch.nn.Module):
    """ LSTM block.\n
    The class inherits from torch.nn.Module
    """
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            device: torch.device
            ):
        """ LSTM block initialization

        Args:
            input_size (int): number of input features
            hidden_size (int): size of hidden state in a LSTM cell
            device (torch.device): CPU or GPU torch.device
        """
        super(LSTMBlock, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        self.lstm = torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            device=self.device
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass for LSTM block

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        self.__check_dim(x)
        batch_size = x.size(0)

        h0 = Variable(
            torch.zeros(1, batch_size, self.hidden_size)
            ).to(self.device)
        c0 = Variable(
            torch.zeros(1, batch_size, self.hidden_size)
            ).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        return out

    def __check_dim(self, x: torch.Tensor) -> bool:
        """ Check dimension and shape for a tensor passed to the LSTM block

        Args:
            x (torch.Tensor): input tensor

        Returns:
            bool: True if check is passed successfully
        """
        assert x.dim() == 3, f"input shape expected to be " \
            f"(batch size, seq len, input size), but got shape = {x.shape}"

        assert x.size(2) == self.input_size, f"input size expected to be " \
            f"{self.input_size}, but got {x.size(2)}"

        return True

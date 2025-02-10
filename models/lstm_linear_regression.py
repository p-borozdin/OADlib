""" The Module provides the architecture of an LSTM-based model
with the Linear Regression block after the LSTM block
"""
import torch

from OADlib.models.base_lstm_model import BaseLSTMModel


class LSTMLinearRegression(BaseLSTMModel):
    """ LSTM-based model with Linear Regression block as its head.\n
    The class inherits from BaseLSTMModel
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
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            device=device
            )

    def _init_head_block(self) -> torch.nn.Module:
        """ Initialize the model's head block.\n
        Overriding the method from the base class.

        Returns:
            torch.nn.Module: Linear Regression head block for the model
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

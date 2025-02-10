""" The Module provides the architecture of an LSTM-based model
with the Fully Connected block after the LSTM block
"""
import torch

from OADlib.models.base_lstm_model import BaseLSTMModel


class LSTMFullyConnected(BaseLSTMModel):
    """ LSTM-based model with Fully Connected block as its head.\n
    The class inherits from BaseLSTMModel
    """
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            activation_func: torch.nn.Module,
            hidden_layer_size: int,
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

        self.activation_func = activation_func
        self.hidden_layer_size = hidden_layer_size

        self._init_blocks()

    def _init_head_block(self) -> torch.nn.Module:
        """ Initialize the model's head block.\n
        Overriding the method from the base class.

        Returns:
            torch.nn.Module: Fully Connected head block for the model
        """
        head_layers: list[torch.nn.Module] = []

        head_layers.append(
            torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=self.hidden_size,
                    out_features=self.hidden_layer_size
                ),
                self.activation_func
            )
        )
        head_layers.append(
            torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=self.hidden_layer_size,
                    out_features=self.hidden_layer_size
                ),
                self.activation_func
            )
        )
        head_layers.append((
            torch.nn.Linear(
                in_features=self.hidden_layer_size,
                out_features=1
                )
            )
        )

        return torch.nn.ParameterList(head_layers)

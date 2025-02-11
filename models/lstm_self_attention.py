""" The Module provides the architecture of an LSTM-based model
with the Self-Attention block after the LSTM block
"""
import torch

from OADlib.models.base_lstm_model import BaseLSTMModel
from OADlib.models.self_attention_part import SelfAttentionPart


class LSTMSelfAttention(BaseLSTMModel):
    """ LSTM-based model with Self-Attention block as its head.\n
    The class inherits from BaseLSTMModel
    """
    def __init__(
            self,
            input_size: int,
            seq_len: int,
            hidden_size: int,
            model_sa_dim: int,
            hidden_layer_size: int,
            activation_func: torch.nn.Module,
            device: torch.device
            ):
        """ Model's initialization

        Args:
            input_size (int): number of input features
            seq_len (int): number of consecutive samples \
                used to make a prediction
            hidden_size (int): size of hidden state in a LSTM cell
            model_sa_dim (int): dimension of the internal data representation \
                in the Self-Attention block
            hidden_layer_size (int): size of a hidden fully connected \
                layer after the Self-Attention block
            activation_func (torch.nn.Module): activation function of hidden \
                layers after the Self-Attention block
            device (torch.device): CPU or GPU torch.device
        """
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            device=device
        )

        self.seq_len = seq_len
        self.model_sa_dim = model_sa_dim
        self.hidden_layer_size = hidden_layer_size
        self.activation_func = activation_func

        self._init_blocks()

    def _init_head_block(self) -> torch.nn.Module:
        """ Initialize the model's head block.\n
        Overriding the method from the base class.

        Returns:
            torch.nn.Module: Self-Attention head block for the model
        """
        head_parts = {}

        head_parts['sa'] = self.__init_sa_part()  # Self-Attention part
        head_parts['fc'] = self.__init_fc_part()  # Fully-connected part

        return torch.nn.ParameterDict(head_parts)

    def __init_sa_part(self) -> torch.nn.Module:
        """ Initialize the model head's Self-Attention part

        Returns:
            torch.nn.Module: Self-Attention part of the head block
        """
        return SelfAttentionPart(
            input_dim=self.hidden_size,
            model_dim=self.model_sa_dim,
            device=self.device
            )

    def __init_fc_part(self) -> torch.nn.Module:
        """ Initialize the model head's fully-connected part

        Returns:
            torch.nn.Module: fully-connected part of the head block
        """
        layers: list[torch.nn.Module] = []

        layers.append(
            torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=self.seq_len * self.model_sa_dim,
                    out_features=self.hidden_layer_size
                    ),
                self.activation_func
                )
            )
        layers.append(
            torch.nn.Linear(
                in_features=self.hidden_layer_size,
                out_features=1
            )
        )

        return torch.nn.ParameterList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass for the model

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        self.__check_dim(x)

        lstm_output = self.blocks['lstm'](x)
        sa_output = self.blocks['head']['sa'](lstm_output)
        output = sa_output.view(sa_output.size(0), -1)

        for layer in self.blocks['head']['fc']:
            output = layer(output)

        return output

    def __check_dim(self, x: torch.Tensor) -> bool:
        """ Check dimension and shape for a tensor passed to \
            the Self-Attention model

        Args:
            x (torch.Tensor): input tensor

        Returns:
            bool: True if check is passed successfully
        """
        assert x.size(1) == self.seq_len, f"expected seq len to be " \
            f"{self.seq_len}, but got {x.size(1)}"

        return True

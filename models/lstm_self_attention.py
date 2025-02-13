""" The Module provides the architecture of an LSTM-based model
with the Self-Attention block after the LSTM block
"""
import torch

from OADlib.models.src.lstm_block import LSTMBlock
from OADlib.models.src.self_attention_head import SelfAttentionHead


class LSTMSelfAttention(torch.nn.Module):
    """ LSTM-based model with Self-Attention block as its head.\n
    The class inherits from torch.nn.Module
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
        super(LSTMSelfAttention, self).__init__()

        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.model_sa_dim = model_sa_dim
        self.hidden_layer_size = hidden_layer_size
        self.activation_func = activation_func
        self.device = device

        self.blocks = torch.nn.ParameterDict()
        self.__init_blocks()

    def __init_blocks(self):
        """ Initialize the model's blocks
        (LSTM and Self-Attention head blocks)
        """
        blocks = {}

        blocks['lstm'] = LSTMBlock(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            device=self.device
        )
        blocks['head'] = SelfAttentionHead(
            seq_len=self.seq_len,
            input_dim=self.hidden_size,
            model_dim=self.model_sa_dim,
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

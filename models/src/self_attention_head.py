""" The Module provides the architectures of the Self-Attention head block
"""
import torch


class SelfAttentionPart(torch.nn.Module):
    """ Self-Attention part.\n
    The class inherits from torch.nn.Module
    """
    def __init__(
            self,
            input_dim: int,
            model_dim: int,
            ):
        """ Self-Attention part initialization

        Args:
            input_dim (int): dimension of the part's input
            model_dim (int): dimension of the internal data representation
        """
        super(SelfAttentionPart, self).__init__()

        self.input_dim = input_dim
        self.model_dim = model_dim

        # normalization factor
        self.norm = self.model_dim ** 0.5

        self.units = torch.nn.ParameterDict()
        self.__init_units()

    def __init_units(self):
        """ Initialization of the part's units
        """
        units = {}

        units['query'] = torch.nn.Linear(
            in_features=self.input_dim,
            out_features=self.model_dim
            )
        units['key'] = torch.nn.Linear(
            in_features=self.input_dim,
            out_features=self.model_dim
            )
        units['value'] = torch.nn.Linear(
            in_features=self.input_dim,
            out_features=self.model_dim
            )

        units['softmax'] = torch.nn.Softmax(dim=2)

        self.units = torch.nn.ParameterDict(units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass for the Self-Attention part

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        queries = self.units['query'](x)
        keys = self.units['key'](x)
        values = self.units['value'](x)

        scores = torch.bmm(queries, keys.transpose(1, 2)) / self.norm
        attention = self.units['softmax'](scores)

        out = torch.bmm(attention, values)

        return out


class SelfAttentionHead(torch.nn.Module):
    """ Self-Attention head block.\n
    The class inherits from torch.nn.Module
    """
    def __init__(
            self,
            seq_len: int,
            input_dim: int,
            model_dim: int,
            hidden_layer_size: int,
            activation_func: torch.nn.Module
            ):
        """ Head block initialization

        Args:
            seq_len (int): number of consecutive samples \
                used to make a prediction
            input_dim (int): dimension of the head's input
            model_dim (int): dimension of the internal data representation \
                in the Self-Attention block
            hidden_layer_size (int): size of a hidden fully connected \
                layer after the Self-Attention block
            activation_func (torch.nn.Module): activation of hidden \
                layers after the Self-Attention block
        """
        super(SelfAttentionHead, self).__init__()

        self.seq_len = seq_len
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.hidden_layer_size = hidden_layer_size
        self.activation_func = activation_func
        self.out_features = 1

        self.parts = torch.nn.ParameterDict()  # SA and FC head parts
        self.__init_parts()

    def __init_parts(self):
        """ Initialization of the head's parts
        """
        parts = {}

        # Self-Attention head part
        parts['sa'] = SelfAttentionPart(
            input_dim=self.input_dim,
            model_dim=self.model_dim
        )
        # Fully-Connected head part
        parts['fc'] = torch.nn.ParameterList(
            (
                torch.nn.Linear(
                    in_features=self.seq_len * self.model_dim,
                    out_features=self.hidden_layer_size
                ),
                self.activation_func,
                torch.nn.Linear(
                    in_features=self.hidden_layer_size,
                    out_features=self.out_features
                )
            )
        )

        self.parts = torch.nn.ParameterDict(parts)

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """ Forward pass for Self-Attention head block

        Args:
            lstm_output (torch.Tensor): input tensor for a head block
            == out tensor for a LSTM block

        Returns:
            torch.Tensor: output tensor
        """
        self.__check_dim(lstm_output)

        sa_output = self.parts['sa'](lstm_output)
        out = sa_output.view(sa_output.size(0), -1)
        for layer in self.parts['fc']:
            out = layer(out)

        return out

    def __check_dim(self, x: torch.Tensor) -> bool:
        """ Check dimension for a tensor passed to the Self-Attention
        head block

        Args:
            x (torch.Tensor): input tensor

        Returns:
            bool: True if check is passed successfully
        """
        assert x.dim() == 3, f"input shape for a Self-Attention head " \
            f"expected to be (batch size, seq len, lstm hidden size), " \
            f"but got shape = {x.shape}"

        return True

""" The Module provides the architecture of a Self-Attention part
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
            device: torch.device
            ):
        """ Self-Attention part initialization

        Args:
            input_dim (int): dimension of the part's input
            model_dim (int): dimension of the internal data representation
            device (torch.device): CPU or GPU torch.device
        """
        super(SelfAttentionPart, self).__init__()

        self.input_dim = input_dim
        self.model_dim = model_dim
        self.device = device

        # normalization factor
        self.norm = self.model_dim ** 0.5

        self.units = torch.nn.ParameterDict()
        self.__init_part()

    def __init_part(self):
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

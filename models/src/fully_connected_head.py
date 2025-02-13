""" The Module provides the architectures of the Fully-Connected head block
"""
import torch


class FullyConnectedHead(torch.nn.Module):
    """ Fully-Connected head block.\n
    The class inherits from torch.nn.Module
    """
    def __init__(
            self,
            in_features: int,
            hidden_layer_size: int,
            activation_func: torch.nn.Module,
            ):
        """ Head block initialization

        Args:
            in_features (int): number of input features of the head block
            hidden_layer_size (int): size of a hidden fully-connected layer
            activation_func (torch.nn.Module): activation of a hidden \
                fully-connected layer
        """
        super(FullyConnectedHead, self).__init__()

        self.in_features = in_features
        self.out_features = 1
        self.hidden_layer_size = hidden_layer_size
        self.activation_func = activation_func

        self.layers = torch.nn.ParameterList()
        self.__init_layers()

    def __init_layers(self):
        """ Initialization of the head's layers
        """
        layers = []

        layers.append(
            torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=self.in_features,
                    out_features=self.hidden_layer_size
                ),
                self.activation_func
            )
        )
        layers.append(
            torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=self.hidden_layer_size,
                    out_features=self.hidden_layer_size
                ),
                self.activation_func
            )
        )
        layers.append(
            torch.nn.Linear(
                in_features=self.hidden_layer_size,
                out_features=self.out_features
                )
            )

        self.layers = torch.nn.ParameterList(layers)

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """ Forward pass for Fully-Connected head block

        Args:
            lstm_output (torch.Tensor): input tensor for a head block
            == out tensor for a LSTM block

        Returns:
            torch.Tensor: output tensor
        """
        self.__check_dim(lstm_output)

        out = lstm_output[:, -1, :]
        for layer in self.layers:
            out = layer(out)

        return out

    def __check_dim(self, x: torch.Tensor) -> bool:
        """ Check dimension for a tensor passed to the Fully-Connected
        head block

        Args:
            x (torch.Tensor): input tensor

        Returns:
            bool: True if check is passed successfully
        """
        assert x.dim() == 3, f"input shape for a Fully-Connected head " \
            f"expected to be (batch size, seq len, lstm hidden size), " \
            f"but got shape = {x.shape}"

        return True

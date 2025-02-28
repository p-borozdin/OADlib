""" The module provides the implementation of a OAD Dataset class
"""
import os
import torch
from torch.utils.data import Dataset


class OADDataset(Dataset):
    """ The OAD Dataset class.\n
    The class inherits from `torch.utils.data.Dataset` class
    """
    def __init__(
            self,
            *,
            path_to_file: str,
            predictors: str | list[str],
            target: str,
            mode: str,
            device: torch.device = torch.device('cpu')
            ):
        """ Initialization of a OAD dataset. Uses `*.pth` files with
        keys `train`, `valid`, `test` to form the dataset from.

        Args:
            path_to_file (str): path to a `*.pth` file to load the dataset from
            predictors (str | tuple[str]): key(s) specifying the name(s) of
                values in the `*.pth` file used as inputs to a model
            target (str): key specifying the name of a target values in the
                `*.pth` file
            mode (str): must be one of `"train"`, `"valid"`, `"test"`
            device (torch.device, optional): CPU or GPU torch device.
                Default is `torch.device("cpu")`.
        """
        assert os.path.exists(path_to_file), \
            f"Unable to load data from file \"{path_to_file}\": " \
            f"the file doesn't exist"

        assert mode in ("train", "valid", "test"), "'mode' expected to be " \
            "one of 'train', 'valid', 'test'"

        super().__init__()

        self.path = path_to_file
        self.x_names = ((predictors,)
                        if isinstance(predictors, str) else predictors)
        self.y_name = target
        self.mode = mode
        self.device = device

        self.x_data = torch.Tensor()
        self.y_data = torch.Tensor()
        self.__load_data()

        self.size = len(self.y_data)

    def __load_data(self):
        """ Loading data from a `*.pth` file on the given `torch.device`
        """
        data = torch.load(self.path, map_location=self.device)

        self.x_data = torch.transpose(torch.stack(
            [torch.from_numpy(data[self.mode][name]) for name in self.x_names],
            axis=1
            ), -2, -1).to(torch.float32)
        self.y_data = (torch.from_numpy(data[self.mode][self.y_name])
                       .to(torch.float32))

    def __len__(self) -> int:
        """ Get the length of the dataset (i.e. the number of samples in it)

        Returns:
            int: dataset's length
        """
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """ Get the dataset's item at given index

        Args:
            index (int): index to take item at.
                Must be in `[0, len(OADDataset) - 1]`

        Returns:
            tuple: `X` and `y` at given index (tuple of `torch.Tensor`s).
                Note that the dimension of `X` is `(seq_len, in_features)`.
        """
        return self.x_data[index], self.y_data[index]

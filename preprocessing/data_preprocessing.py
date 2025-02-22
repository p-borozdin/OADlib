""" The module provides the implementation of the data preprocessing class
"""
import os
import pandas as pd
import numpy as np
import torch
from sklearn import model_selection
from OADlib.preprocessing.smoothing.base_smoother import BaseSmoother
from OADlib.preprocessing.derivative_computation.\
    base_derivative_computer import BaseDerivativeComputer


class Preprocessing:
    """ Data preprocessing class
    """
    def __init__(
            self,
            data_dir: str
            ):
        """ Initalization of a data preprocessing class

        Args:
            data_dir (str): path to raw data (*.txt files)
        """
        assert os.path.exists(data_dir), \
            f"directory \"{data_dir}\" doesn't exist"

        self.data_dir = data_dir

    def smooth(
            self,
            smoother: BaseSmoother,
            col_name: str,
            save_to: str,
            ext: str = "csv",
            use_files: tuple[str] | None = None,
            load_from: str | None = None,
            sep: str = ","
            ):
        """ Performs data smoothing for each file
        in the directory with raw data

        Args:
            smoother (BaseSmoother): smoothing algorithm
            col_name (str): column that needs to be smoothed
                save_to (str): directory to save `pandas.Dataframes`
                with smoothed files\n
            ext (str, optional): extension of saved files. Default is `"csv"`
            use_files (tuple[str] | None, optional): specify files to use
                for smoothing. If `None`, all files from source directory
                will be used. Default is `None`
            load_from (str | None, optional): if `load_from` is `None`,
                the files from directory with raw data will be used.
                If `load_data` is specified, it denotes a directory
                with source files for smoothing. Default is `None`
            sep (str, optional): column separator in input data.
                Default is `","`

        The method goes through each file in the source directory
        (if `use_files` is `None`), or through each file in `use_files`
        (if the latter is specified)
        and performs smoothing of a specific column denoted as `col_name`.
        The smoothed `*.<ext>` files are saved to a `save_to` directory.
        In the created files the smoothed series is denoted as `col_name`,
        and initial series is denorted as `Unsmoothed col_name`.
        For example, if you use `ext = "csv"` and want to smooth
        a column with name `MyColumn` in your file `my_file.txt`,
        then a new file in `save_to` directory with name `my_file.csv`
        will be created, where the column `MyColumn` will stand for
        the smoothed data, and column `Unsmoothed MyColumn` will stand for
        the initial data.
        """
        assert os.path.exists(save_to), \
            f"Unable to smooth data: directory \"{save_to}\" doesn't exist"

        if load_from is not None:
            assert os.path.exists(load_from), \
                f"Unable to smooth data: " \
                f"directory \"{load_from}\" doesn't exist"

        src_dir = self.data_dir if load_from is None else load_from
        filenames = os.listdir(src_dir) if use_files is None else use_files

        for filename in sorted(filenames):
            df = pd.read_csv(f"{src_dir}/{filename}", sep=sep)
            smoothed_data = smoother.execute(df[col_name])

            df.rename(
                columns={col_name: f"Unsmoothed {col_name}"},
                inplace=True)
            df[col_name] = smoothed_data

            fn, _ = os.path.splitext(filename)
            df.to_csv(f"{save_to}/{fn}.{ext}", index=False)

    def compute_derivatives(
            self,
            deriv_computer: BaseDerivativeComputer,
            x_col_name: str,
            y_col_name: str,
            deriv_col_name: str,
            save_to: str,
            ext: str = "csv",
            use_files: tuple[str] | None = None,
            load_from: str | None = None,
            sep: str = ","
            ):
        """ Computes the specified derivative for each filename
        in the given directory

        Args:
            deriv_computer (BaseDerivativeComputer): algorithm of
                derivative computation
            x_col_name (str): name of a column used as x-values
            y_col_name (str): name of a column used as y-values
            deriv_col_name (str): name of a column used
                to store computed derivative
            save_to (str): directory to save `pandas.DataFrames`
                with computed derivatives
            ext (str, optional): extension of saved files. Default is `"csv"`
            use_files (tuple[str] | None, optional): specify files to use
                for computing derivatives. If `None`, all files from source
                directory will be used. Default is `None`
            load_from (str | None, optional): if `load_from` is `None`,
                the files from directory with raw data will be used.
                If `load_data` is specified, it denotes a directory
                to take files from to compute derivatives. Default is `None`
            sep (str, optional): column separator in input data.
                Default is `","`

        The method goes through each file in the source directory
        (if `use_files` is `None`), or through each file in `use_files`
        (if the latter is specified)
        and computes the derivative `dy/dx`,
        where series `x` is specified by values from column `x_col_name`,
        and series `y` is specified by values from column `y_col_name`.
        The computed derivatives will be stored under column `deriv_col_name`
        in files in the `save_to` directory. For example,
        if you use `ext = "csv"`, and the series `x` and `y` were taken from
        file `my_file.txt`, then the computed derivatives can be found in
        `save_to/my_file.csv`
        """
        assert os.path.exists(save_to), \
            f"Unable to compute derivatives: " \
            f"directory \"{save_to}\" doesn't exist"

        if load_from is not None:
            assert os.path.exists(load_from), \
                f"Unable to compute derivatives: " \
                f"directory \"{load_from}\" doesn't exist"

        src_dir = self.data_dir if load_from is None else load_from
        filenames = os.listdir(src_dir) if use_files is None else use_files

        for filename in sorted(filenames):
            df = pd.read_csv(f"{src_dir}/{filename}", sep=sep)
            derivatives = deriv_computer.compute(
                x=df[x_col_name],
                y=df[y_col_name]
            )

            df[deriv_col_name] = derivatives

            fn, _ = os.path.splitext(filename)
            df.to_csv(f"{save_to}/{fn}.{ext}", index=False)

    def group_by(
            self,
            seq_len: int,
            *,
            group_columns: str | tuple[str],
            target_columns: str | tuple[str],
            save_to: str,
            use_files: tuple[str] | None = None,
            load_from: str | None = None,
            sep: str = ","
            ):
        """ Groupes values from `group_columns` by `seq_len` samples
        and mathes them to a corresponding single value from each
        column in `target_columns` and stores the data in an `numpy.npz`
        archive

        Args:
            seq_len (int): the number of consecutive samples to group by
            group_columns (str | tuple[str]): denotes a column/columns
                values from which are grouped by `seq_len` samples
            target_columns (str | tuple[str]): denotes a column/columns
                to values from which the previous gropued values are matched
            save_to (str): directory to save `numpy.npz` archive
                with grouped values
            use_files (tuple[str] | None, optional): specify files to use
                for grouping samples in. If `None`, all files from source
                directory will be used. Default is `None`
            load_from (str | None, optional): if `load_from` is `None`,
                the files from directory with raw data will be used.
                If `load_data` is specified, it denotes a directory
                to take files from to group values. Default is `None`
            sep (str, optional): column separator in input data.
                Default is `","`

        Imagine you have a data frame with columns named `a`, `b`, `c` and `d`;
        and you want to group values from columns `a`, `b` so that slices
        `a[i - n:i]`, `b[i - n:i]` must match single values
        `c[i - 1]` and `d[i - 1]`.
        To do so you can call `group_by()` method with `seq_len` = `n`,
        `group_columns` = `('a', 'b')`, `target_columns` = `('c', 'd')`.
        The desired matching will be stored in the `save_to` directory in
        `numpy.npz` format. The method goes through each file in the source
        directory (if `use_files` is `None`), or through each file in
        `use_files` (if the latter is specified)
        """
        assert os.path.exists(save_to), \
            f"Unable to group data: " \
            f"directory \"{save_to}\" doesn't exist"

        if load_from is not None:
            assert os.path.exists(load_from), \
                f"Unable to group data: " \
                f"directory \"{load_from}\" doesn't exist"

        src_dir = self.data_dir if load_from is None else load_from
        filenames = os.listdir(src_dir) if use_files is None else use_files

        groups_list = ((group_columns,)
                       if isinstance(group_columns, str) else group_columns)
        targets_list = ((target_columns,)
                        if isinstance(target_columns, str) else target_columns)

        for filename in sorted(filenames):
            df = pd.read_csv(f"{src_dir}/{filename}", sep=sep)

            target_data = {
                name: np.array(df[name][seq_len - 1:])
                for name in targets_list}

            data_len = len(df) - (seq_len - 1)
            grouped_data = {
                name: np.array([df[name][i:i + seq_len]
                                for i in range(data_len)])
                for name in groups_list}

            fn, _ = os.path.splitext(filename)
            np.savez(f"{save_to}/{fn}.npz", **(target_data | grouped_data))

    def train_valid_test_split(
            self,
            load_from: str | None = None,
            use_files: tuple[str] | None = None,
            *,
            train_size: float,
            valid_size: float,
            test_size: float,
            save_to: str,
            random_state: int = 42
            ):
        """ Performs splitting the data from `arrays`
        into training/validation/testing sets in the given proportions

        Args:
            load_from (str | None, optional): if `load_from` is `None`,
                the files from directory with raw data will be used.
                If `load_data` is specified, it denotes a directory
                to take files from to split data. Default is `None`.
                Note that the files from source directory are expected
                to have `*.npz` extension
            use_files (tuple[str] | None, optional): specify files to use
                for splitting. If `None`, all files from source
                directory will be used. Default is `None`. Note that
                files are expected to have `*.npz` extension
            train_size (float): training set part, must be in `(0, 1)`
            valid_size (float): validation set part, must be in `(0, 1)`
            test_size (float): testing set part, must be in `(0, 1)`
            save_to (str): directory to save splitted data into
            random_state (int, optional): random state for `sklearn`'s
                `train_test_split()` function. Default is `42`

        If `load_from` is `None`, source directory is directory with raw
        data, otherwise source directory is `load_from`.
        The method splits each `numpy.npz` array (if `use_files`
        is `None`), or files from `use_files` list (if `use_files` is
        specified) in the source directory in the given proportions.
        stores splitted `*.pth` data in the `save_to` directory in form of a
        dictionary with keys `"train"`, `"valid"`, `"test"`. Note that
        the next equation must be satisfied:
        `train_size + valid_size + test_size == 1.0`
        """
        assert os.path.exists(save_to), \
            f"Unable to group data: " \
            f"directory \"{save_to}\" doesn't exist"

        if load_from is not None:
            assert os.path.exists(load_from), \
                f"Unable to group data: " \
                f"directory \"{load_from}\" doesn't exist"

        for size in (train_size, valid_size, test_size):
            assert not np.isclose(size, 1.0), "All sizes must be < 1.0"
            assert not np.isclose(size, 0.0), "All sizes must be > 0.0"

        assert np.isclose(train_size + valid_size + test_size, 1.0), \
            "train_size + valid_size + test_size == 1.0 isn't satisfied with "\
            f"train_size = {train_size}, valid_size = {valid_size}, " \
            f"test_size = {test_size}"

        src_dir = self.data_dir if load_from is None else load_from
        filenames = os.listdir(src_dir) if use_files is None else use_files

        for filename in sorted(filenames):
            fn, ext = os.path.splitext(filename)
            assert ext == ".npz", "Only *.npz arrays are supported"

            dataset: dict[str, dict[str, torch.Tensor]] = {
                "train": {}, "valid": {}, "test": {}
            }

            archive = np.load(f"{src_dir}/{filename}")  # loading *.npz archive
            for key in archive.keys():
                train, valid_and_test = model_selection.train_test_split(
                    archive[key],
                    train_size=train_size,
                    random_state=random_state
                )
                valid, test = model_selection.train_test_split(
                    valid_and_test,
                    test_size=test_size / (test_size + valid_size),
                    random_state=random_state
                )

                dataset['train'][key] = train
                dataset['valid'][key] = valid
                dataset['test'][key] = test

            torch.save(dataset, f"{save_to}/{fn}.pth")

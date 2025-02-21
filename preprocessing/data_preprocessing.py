""" The module provides the implementation of the data preprocessing class
"""
import os
import pandas as pd
import numpy as np
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
            load_from (str | None, optional): if `load_from` is `None`,
                the files from directory with raw data will be used.
                If `load_data` is specified, it denotes a directory
                to take files from for smoothing. Default is `None`
            sep (str, optional): column separator in input data.
                Default is `","`

        The method goes through each file in a raw data directory
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

        for filename in sorted(os.listdir(src_dir)):
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
            load_from (str | None, optional): if `load_from` is `None`,
                the files from directory with raw data will be used.
                If `load_data` is specified, it denotes a directory
                to take files from to compute derivatives. Default is `None`
            sep (str, optional): column separator in input data.
                Default is `","`

        The method goes through each file in a source directory
        (raw data directory or specified `load_from` directory)
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

        for filename in sorted(os.listdir(src_dir)):
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
            load_from (str | None, optional): if `load_from` is `None`,
                the files from directory with raw data will be used.
                If `load_data` is specified, it denotes a directory
                to take files from to group values. Default is `None`
            sep (str, optional): column separator in input data.
                Default is `","`

        Imagine you have a dataframe with columns named `a`, `b`, `c` and `d`;
        and you want to group values from columns `a`, `b` so that slices
        `a[i - n:i]`, `b[i - n:i]` must match values `c[i - 1]` and `d[i - 1]`.
        To do so you can call `group_by()` method with `seq_len` = `n`,
        `group_columns` = `('a', 'b')`, `target_columns` = `('c', 'd')`.
        The desired matching will be stored in the `save_to` directory in
        `numpy.npz` format
        """
        assert os.path.exists(save_to), \
            f"Unable to group data: " \
            f"directory \"{save_to}\" doesn't exist"

        if load_from is not None:
            assert os.path.exists(load_from), \
                f"Unable to group data: " \
                f"directory \"{load_from}\" doesn't exist"

        src_dir = self.data_dir if load_from is None else load_from

        groups_list = ((group_columns,)
                       if isinstance(group_columns, str) else group_columns)
        targets_list = ((target_columns,)
                        if isinstance(target_columns, str) else target_columns)

        for filename in sorted(os.listdir(src_dir)):
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

""" The module provides the implementation of the data preprocessing class
"""
import os
import numpy as np
import pandas as pd
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
            sep: str = ","
            ):
        """ Performs data smoothing for each file
        in the directory with raw data

        Args:
            smoother (BaseSmoother): smoothing algorithm
            col_name (str): column that needs to be smoothed
                save_to (str): directory to save `pandas.Dataframes`
                with smoothed files\n
            ext (str): extension of saved files. Default is `"csv"`
            sep (str): column separator. Default is `","`

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

        for filename in sorted(os.listdir(self.data_dir)):
            df = pd.read_csv(f"{self.data_dir}/{filename}", sep=sep)
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
            ext (str): extension of saved files. Default is `"csv"`
            load_from (str | None): if `load_from` is `None`,
                the files from directory with raw data will be used.
                If `load_data` is specified, it denotes a directory
                to take files from to compute derivatives
            sep (str): column separator. Default is `","`

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

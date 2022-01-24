# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import random
import secrets
from datetime import timedelta
from itertools import accumulate
from typing import List, Tuple

import numpy as np
import pandas as pd

AMOUNT_DAY = 96  # Duration of the periods (in T-15) that are in a day (default = 96)
PERIOD_TIMEDELTA = 1  # Duration of the periods (in days) that will be sampled as validation data for each split.
PEAK_FRACTION = 0.15


def group_kfold(
    input_data: pd.DataFrame, n_folds: int, random_split: bool = True
) -> pd.DataFrame:
    """Function to group data into groups, according to the date and the number of folds
        - each date gets assigned a number between 0 and n_folds

    Args:
        input_data (pd.DataFrame): Input data
        n_folds (int): Number of folds
        random_split (bool): Indicates if random split needs to be applied

    Returns:
        grouped data (pandas.DataFrame)

    """
    unique_dates = input_data["dates"].unique()  # dates defines the day (Y-M-D)
    # Group separators
    len_data = len(unique_dates)  # number of indices that can be used for splitting
    size = len_data // n_folds  # size of each fold set
    rem = (
        len_data % n_folds
    )  # remaining number of indices when divided in fold sets of equal size
    separators = list(
        accumulate([0] + [size + 1] * rem + [size] * (n_folds - rem))
    )  # location of seperators

    items = list(unique_dates)

    if random_split:
        random.shuffle(items)  # if random, shuffle the days

    for i, s in enumerate(zip(separators, separators[1:])):
        group = items[slice(*s)]
        input_data.loc[
            input_data[input_data["dates"].isin(group)].index, "random_fold"
        ] = i
    return input_data


def sample_indices_train_val(
    data: pd.DataFrame, peaks: pd.DataFrame
) -> Tuple[np.array, np.array]:
    """
    Sample indices of given period length assuming the peaks are evenly spreaded.

    Args:
        data (pandas.DataFrame): Clean data with features
        peaks (pd.DataFrame): Data frame of selected peaks to sample the dates from

    Returns:
        np.array: List with the start point of each peak
        np.array: Sorted list with the indices corresponding to the peak

    """

    sampled = set()
    peaks_val = []

    for peak in peaks:
        sampled |= set(data[data.index.date == peak].index)
        peaks_val.append(peak)
    return peaks_val, np.sort(list(sampled))


def random_sample(all_peaks: np.array, k: int) -> np.array:
    """
    Random sampling of numbers out of a np.array
    (implemented due to security sonar cloud not accepting the random built-in functions)

    Args:
        all_peaks (np.array): List with numbers to sample from
        k (int): Number of wanted samples

    Returns:
        np.array: Sorted array with the random samples (dates from the peaks)

    """

    random_peaks = []
    all_peaks_list = all_peaks.tolist()
    for _ in range(k):
        element_random = secrets.choice(all_peaks_list)
        all_peaks_list.remove(element_random)
        random_peaks.append(element_random)
    return np.array(random_peaks)


def split_data_train_validation_test(
    data_: pd.DataFrame,
    test_fraction: float = 0.1,
    validation_fraction: float = 0.15,
    back_test: bool = False,
    stratification_min_max: bool = True,
) -> (List[int], pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Split input data into train, test and validation set.

    Function for splitting data with features in a train, test and
    validation dataset. In an operational setting the following sequence is
    returned (when using stratification):

    Test >> Train >> Validation

    For a back test (indicated with argument "back_test") the following sequence
    is returned:

    Train >> Validation >> Test

    The ratios of the different types can be set with test_fraction and
    validation fraction.

    Args:
        data_ (pandas.DataFrame): Cleaned data with features
        test_fraction (float): Number between 0 and 1 that indicates the desired
            fraction of test data.
        validation_fraction (float): Number between 0 and 1 that indicates the
            desired fraction of validation data.
        back_test (bool): Indicates if data is intended for a back test.
        stratification_min_max (bool): Indicates if validation data must be sampled as
            periods, using stratification on min and max values per day.
            If True, 'extreme days' are ensured to be included in the validation and train sets,
            ensuring the validation set to be representative of the train set.

    Returns:
        min_max_dates (List[pd.DatetimeIndex]),
        peaks_val_train (List[List: 'peak dates in validation',
                              List: 'peak dates in train'])
        train_data (pandas.DataFrame): Train data.
        validation_data (pandas.DataFrame): Validation data.
        test_data (pandas.DataFrame): Test data.

    """

    train_fraction = 1 - (test_fraction + validation_fraction)
    if train_fraction < 0:
        raise ValueError(
            "Test ({test_fraction}) and validation fraction ({validation_fraction}) too high."
        )

    # Get start date from the index
    start_date = data_.index.min().to_pydatetime()
    end_date = data_.index.max().to_pydatetime()

    # Calculate total of quarter hours (PTU's) in input data
    number_indices = len(data_.index.unique())  # Total number of unique timepoints
    delta = (
        data_.index.unique().sort_values()[1] - data_.index.unique().sort_values()[0]
    )  # Delta t, assumed to be constant throughout DataFrame
    delta = timedelta(
        seconds=delta.seconds
    )  # Convert from pandas timedelta to original python timedelta

    # Determine which dates are in testset
    if back_test:
        start_date_test = end_date - np.round(number_indices * test_fraction) * delta
        test_data = data_[start_date_test:]
        train_val_data = data_[:start_date_test]
    else:
        start_date_val = start_date + np.round(number_indices * test_fraction) * delta
        test_data = data_[:start_date_val]
        train_val_data = data_[start_date_val:]

    if stratification_min_max and (len(set(train_val_data.index.date)) >= 4):
        # First determine the dates with min and max values
        # Let's consider the 15% lowest or highest days, with a minimum
        # of two dates in the min and max list
        train_val_dates = list(set(train_val_data.index.date))
        n_days_per_subset = int(max(0.15 * len(train_val_dates), 2))
        # Find max_dates
        max_dates = (
            train_val_data[["load"]]
            .resample("1D")
            .max()
            .sort_values(by="load", ascending=False)
            .dropna()
            .index[:n_days_per_subset]
        )
        # Find min_dates, but do not consider the max_dates
        min_dates_subset = train_val_data.loc[
            ~np.isin(train_val_data.index.date, max_dates), ["load"]
        ]
        min_dates = (
            min_dates_subset[["load"]]
            .resample("1D")
            .min()
            .sort_values(by="load", ascending=True)
            .dropna()
            .index[:n_days_per_subset]
        )
        other_dates = [
            x for x in train_val_dates if x not in min_dates and x not in max_dates
        ]

        # Divide min, max and other dates fairly over validation and train set, with at least 1 min and max in train and validation
        val_dates = []
        train_dates = []
        for date_set in [max_dates, min_dates, other_dates]:
            n_days_val = max(1, int(validation_fraction * len(date_set)))
            val_dates += random.sample(list(date_set), n_days_val)
            train_dates += [x for x in date_set if x not in val_dates]

        validation_data = train_val_data[np.isin(train_val_data.index.date, val_dates)]
        train_data = train_val_data[np.isin(train_val_data.index.date, train_dates)]

    # Default sampling, take a one single validation set.
    else:
        if back_test:
            start_date_train = (
                start_date + np.round(number_indices * validation_fraction) * delta
            )
            end_date_train = end_date - np.round(number_indices * test_fraction) * delta
            validation_data = data_[:start_date_train]
            train_data = data_[start_date_train:end_date_train]
        else:
            start_date_val = (
                start_date + np.round(number_indices * test_fraction) * delta
            )
            start_date_train = (
                start_date_val + np.round(number_indices * validation_fraction) * delta
            )
            train_data = data_[start_date_train:None]
            validation_data = data_[start_date_val:start_date_train]

    train_data = train_data.sort_index()
    validation_data = validation_data.sort_index()
    test_data = test_data.sort_index()

    return (
        None,  # min_max_dates, TODO: check if this is required or we can remove this
        None,  # peaks_val_train,
        train_data,
        validation_data,
        test_data,
    )

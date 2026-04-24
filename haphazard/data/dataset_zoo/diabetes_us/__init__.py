"""
haphazard.data.dataset_zoo.diabetes_us
--------------------------------------
Diabetes_US dataset implementation for binary classification.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ...base_dataset import BaseDataset
from ...dataset_zoo import register_dataset
from ....utils import find_file


@register_dataset("diabetes_us")
class Diabetes_US(BaseDataset):
    """
    Diabetes_US dataset for binary classification.

    Attributes
    ----------
    name : str
        Name of the dataset.
    haphazard_type : str
        Indicates dataset type, e.g., "intrinsic".
    task : str
        Task type, e.g., "classification".
    num_classes : int
        Number of unique output classes.
    """

    def __init__(self, base_path: str = "./", **kwargs) -> None:
        """
        Initialize the diabetes_us dataset instance.

        Parameters
        ----------
        base_path : str, default="./"
            Base directory under which the dataset file is located.
        **kwargs
            Additional keyword arguments passed to the parent constructor.
        """
        self.name = "diabetes_us"
        self.haphazard_type = "intrinsic"
        self.task = "classification"
        self.num_classes = 2

        super().__init__(base_path=base_path, **kwargs)

    def read_data(self, base_path: str = ".") -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """
        Load and process the diabetes_us dataset.

        Parameters
        ----------
        base_path : str, default="."
            Directory under which the dataset file is searched.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.int64]]
            x : ndarray of shape (n_samples, n_features)
                Feature matrix with dtype float64.
            y : ndarray of shape (n_samples,)
                Binary label vector (0 or 1) with dtype int64.
        """
        data_path: str = find_file(base_path, "diabetes_us.csv")
        df =  pd.read_csv(data_path, header = None, engine = 'python')

        # 1st row contains the header name so we remove that
        df = df[1:]

        # Only readmission days <30 is positive class, otherwise it is a negative class
        label = np.array(df[49] == '<30')*1

        # The last column contains the labels
        df = df.iloc[:,:49]

        # The first two column is admission and patient id, so we drop these
        df = df.iloc[:,2:]

        # "?", "nan" for glucose serum test result (feat no. 22) and "nan" for A1c test result (feat no. 23), is considered
        # unavailable/unmeasured features. For the time being we denote this by "-1" and later we replace it with np.nan
        df[df == '?'] = "-1"
        df[22] = df[22].fillna("-1")
        df[23] = df[23].fillna("-1")

        # The age feature (feat no. 4) indicates only if the age is in a bracket of 10 ([0,10], [10,20], ...). We consider 
        # the median value of the bracket as the actual value and replace it with that
        val_list = sorted(list(set(np.unique(df[4])).difference({"-1"})))
        age_list = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
        # for j in range(len(val_list)):
        #     df.loc[df[4] == val_list[j], 4] = age_list[j]
        # Create a mapping dictionary from your two lists
        mapping = dict(zip(val_list, age_list))
        df[4] = df[4].replace(mapping)
        df[4] = df[4].astype(int)

        # Similar to age feature, the weight feature (feat no. 5) is represented in brackets. We consider the median value 
        # and replace with that
        val_list = ['[0-25)', '[25-50)', '[50-75)', '[75-100)', '[100-125)', '[125-150)', '[150-175)', '[175-200)', '>200']
        weight_list = [12.5, 37.5, 62.5, 87.5, 112.5, 137.5, 162.5, 187.5, 212.5]
        # for j in range(len(val_list)):
        #     df.loc[df[5] == val_list[j], 5] = age_list[j]
        mapping = dict(zip(val_list, weight_list))
        df[5] = df[5].replace(mapping)
        df[5] = df[5].astype(int)

        # The below feat_list features contains categorical value. We transform them to numerical value by assigning them
        # value from 1 to the number of categories in each feature
        feat_list = [2, 3, 10, 11, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
        for i in feat_list:
            # val_list = sorted(list(set(np.unique(df[i])).difference({"-1"})))
            # # print(val_list)
            # for j in range(len(val_list)):
            #     df.loc[df[i] == val_list[j], i] = j+1
            # Get unique values excluding "-1", sorted
            labels = sorted([x for x in df[i].unique() if x != "-1"])
            mapping = {label: idx + 1 for idx, label in enumerate(labels)}
            df[i] = df[i].replace(mapping)
            
        # Substitute each position containing -1 with nan value
        df[df == "-1"] = np.nan

        # Convert everything to float type
        for i in df.columns:
                df[i] = np.array(df[i]).astype(float)
                
        df.insert(0, column="class", value=label)
        data = df.sample(frac = 1, random_state=42)

        x = data.iloc[:,1:].to_numpy(dtype=np.float64)
        y = data.iloc[:,:1].to_numpy(dtype=np.int64).ravel()

        return x, y


__all__ = ["Diabetes_US"]

import os
from typing import Callable, Optional, Tuple
from warnings import warn

import numpy as np
import pandas as pd
import SimpleITK as sitk
from joblib import Parallel, delayed

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import scale

from torchmtlr.utils import make_time_bins, encode_survival

def find_centroid(mask: sitk.Image) -> np.ndarray:
    """Find the centroid of a binary image in image
    coordinates.

    Parameters
    ----------
    mask
        The binary mask image.

    Returns
    -------
    np.ndarray
        The (x, y, z) coordinates of the centroid
        in image space.
    """
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    centroid_coords = stats.GetCentroid(1)
    centroid_idx = mask.TransformPhysicalPointToIndex(centroid_coords)
    return np.asarray(centroid_idx, dtype=np.float64)


class RadcureDataset(Dataset):
    """Dataset class used in simple CNN baseline training.

    The images are loaded using SimpleITK, preprocessed and cached for faster
    retrieval during training.
    """
    def __init__(self,
                 root_directory: str,
                 clinical_data_path: str,
                 patch_size: int = 50,
                 target_col: str = "target_binary",
                 train: bool = True,
                 cache_dir: str = "data_cache/",
                 transform: Optional[Callable] = None,
                 num_workers: int = 1):
        """Initialize the class.

        If the cache directory does not exist, the dataset is first
        preprocessed and cached.

        Parameters
        ----------
        root_directory
            Path to directory containing the training and test images and
            segmentation masks.
        clinical_data_path
            Path to a CSV file with subject metadata and clinical information.
        patch_size
            The size of isotropic patch to extract around the tumour centre.
        target_col
            The name of a column in the clinical dataframe used as prediction
            target.
        train
            Whether to load the training or test set.
        cache_dir
            Path to directory where the preprocessed images will be cached.
        transform
            Callable used to transform the images after preprocessing.
        num_workers
            Number of parallel processes to use for data preprocessing.
        """
        print (clinical_data_path)
        self.root_directory = root_directory
        self.patch_size = patch_size
        self.target_col = target_col
        self.train = train
        self.transform = transform
        self.num_workers = num_workers
        
        if self.train:
            self.split = "training"
        else:
            self.split = "test"
            
        self.clinical_data = self.make_data(clinical_data_path, split=self.split)
        
        
#         try:
#             self.clinical_data = clinical_data[clinical_data["split"] == self.split]
#         except:
#             self.clinical_data = clinical_data
        
        if self.train:
            self.time_bins = make_time_bins(self.clinical_data["time"], event=self.clinical_data["event"])
            # self.y        = encode_survival(clinical_data["time"], clinical_data["event"], time_bins)
            # print(clinical_data)
            multi_events   = self.clinical_data.apply(lambda x: self.multiple_events(x), axis=1)
            self.y         = encode_survival(self.clinical_data["time"], multi_events, self.time_bins)

        self.cache_path = os.path.join(cache_dir, self.split)

#         if not self.train and len(self.clinical_data) == 0:
#             warn(("The test set is not available at this stage of the challenge."
#                   " Testing will be disabled"), UserWarning)
#         else:
#             # TODO we should also re-create the cache when the patch size is changed
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        
        if len(os.listdir(self.cache_path))==0:
            self._prepare_data()
            
    def multiple_events(self, row):
        event        = row["event"]
        cancer_death = row["cancer_death"]
        
        if event==0:
            return 0
        elif cancer_death==0:
            return 1
        elif cancer_death==1:
            return 2
        else:
            raise UhOh
        
    def make_data(self, path, split="training"):
        """Load and preprocess the data."""
        clinical_data = (pd.read_csv(path)
                         .query("split == @split")
                         #.set_index("Study ID")
                         .drop(["split"], axis=1, errors="ignore"))
        if split == "training":
            clinical_data = clinical_data.rename(columns={"death": "event", "survival_time": "time"})
            # Convert time to months
            clinical_data["time"] *= 12

        # binarize T stage as T1/2 = 0, T3/4 = 1
        clinical_data["T Stage"] = clinical_data["T Stage"].map(
            lambda x: "T1/2" if x in ["T1", "T1a", "T1b", "T2"] else "T3/4")

        # use more fine-grained grouping for N stage
        clinical_data["N Stage"] = clinical_data["N Stage"].map({
                                                                "N0":  "N0",
                                                                "N1":  "N1",
                                                                "N2":  "N2",
                                                                "N2a": "N2",
                                                                "N2b": "N2",
                                                                "N2c": "N2",
                                                                "N3":  "N3",
                                                                "N3a": "N3",
                                                                "N3b": "N3"})
        
        clinical_data["Stage"] = clinical_data["Stage"].map(
            lambda x: "I/II" if x in ["I", "II", "IIA"] else "III/IV")

        clinical_data["ECOG"] = clinical_data["ECOG"].map(
            lambda x: ">0" if x > 0 else "0")

        clinical_data = pd.get_dummies(clinical_data,
                                       columns=["Sex",
                                                "T Stage",
                                                "N Stage",
                                                "Disease Site",
                                                "Stage",
                                                "ECOG"],
                                       drop_first=True)
        
        clinical_data = pd.get_dummies(clinical_data, columns=["HPV Combined"])
        return clinical_data

    def _prepare_data(self):
        """Preprocess and cache the dataset."""

        Parallel(n_jobs=self.num_workers)(
            delayed(self._preprocess_subject)(subject_id)
            for subject_id in self.clinical_data["Study ID"])

    def _preprocess_subject(self, subject_id: str):
        """Preprocess and cache a single subject."""
        # load image and GTV mask
        print(self.root_directory)
        print(self.split)
        print(subject_id)
        path = os.path.join(self.root_directory, self.split,
                            "{}", f"{subject_id}.nrrd")
        image = sitk.ReadImage(path.format("images"))
        mask = sitk.ReadImage(path.format("masks"))

        # crop the image to (patch_size)^3 patch around the tumour centre
        tumour_centre = find_centroid(mask)
        size = np.ceil(self.patch_size / np.asarray(image.GetSpacing())).astype(np.int) + 1
        min_coords = np.floor(tumour_centre - size / 2).astype(np.int64)
        max_coords = np.floor(tumour_centre + size / 2).astype(np.int64)
        min_x, min_y, min_z = min_coords
        max_x, max_y, max_z = max_coords
        image = image[min_x:max_x, min_y:max_y, min_z:max_z]

        # resample to isotropic 1 mm spacing
        reference_image = sitk.Image([self.patch_size]*3, sitk.sitkFloat32)
        reference_image.SetOrigin(image.GetOrigin())
        image = sitk.Resample(image, reference_image)

        # window image intensities to [-500, 1000] HU range
        image = sitk.Clamp(image, sitk.sitkFloat32, -500, 1000)

        sitk.WriteImage(image, os.path.join(self.cache_path, f"{subject_id}.nrrd"), True)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get an input-target pair from the dataset.

        The images are assumed to be preprocessed and cached.

        Parameters
        ----------
        idx
            The index to retrieve (note: this is not the subject ID).

        Returns
        -------
        tuple of torch.Tensor and int
            The input-target pair.
        """
        
        try:      # training data
            # clin_var_data = self.clinical_data.drop(["target_binary", 'time', 'event', 'Study ID'], axis=1)
            clin_var_data = self.clinical_data.drop(["target_binary", 'time', 'event', 'cancer_death', 'Study ID'], axis=1)
        except:   # test data
            clin_var_data = self.clinical_data.drop(['Study ID'], axis=1)
        clin_var = clin_var_data.iloc[idx].to_numpy(dtype='float32')
        
        if self.train:
            target = self.y[idx]
        else:
            target = torch.tensor(np.zeros(29))
        
        labels = self.clinical_data.iloc[idx].to_dict()
        
        subject_id = self.clinical_data.iloc[idx]["Study ID"]
        path = os.path.join(self.cache_path, f"{subject_id}.nrrd")
#         print('hi:', path)
        image = sitk.ReadImage(path)

        if self.transform is not None:
            image = self.transform(image)
    
        return (image, clin_var), target, labels

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.clinical_data)




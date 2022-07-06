"""
Class for loading data in h5 format
"""
import random

# torch imports
from torch import from_numpy
from torch import flip

# torch imports
from torch.utils.data import Dataset

# generic imports
import h5py
import numpy as np
from abc import ABC, abstractmethod

class H5CommonDataset(Dataset, ABC):
    """
    Initialize with file of h5 data.  Sets up access to all of the data that is common between
    the digitized hits data and the truth hits data.  These are:

    event_ids 	(n_events,) 	int32 	ID of the event in the ROOT file
    root_files 	(n_events,) 	object 	File name and location of the ROOT file
    labels 	(n_events,) 	int32 	Label for event classification (gamma=0, electron=1, muon=2)
    positions 	(n_events,1,3) 	float32 	Initial (x, y, z) position of simulated particle
    angles 	(n_events,2) 	float32 	Initial direction of simulated particle as (polar, azimuth) angles
    energies 	(n_events,1) 	float32 	Initial total energy of simulated particle
    veto 	(n_events,) 	bool 	OD veto estimate based on any Cherenkov producing particles exiting the tank, with initial energy above threshold
    veto2 	(n_events,) 	bool 	OD veto estimate based on any Cherenkov producing particles exiting the tank, with an estimate of energy at the point the particle exits the tank being above threshold
    event_hits_index 	(n_events,) 	int64 	Starting index in the hit dataset objects for hits of a particular event

    hit_pmt 	(n_hits,) 	int32 	PMT ID of the digitized hit
    hit_time 	(n_hits,) 	float32 	Time of the digitized hit
    """
    def __init__(self, h5_path):
        """
        Args:
            h5_path             ... path to h5 dataset file
            transforms          ... transforms to apply
        """
        self.h5_path = h5_path
        with h5py.File(self.h5_path, 'r') as h5_file:
            self.dataset_length = h5_file["labels"].shape[0]

        self.initialized = False
        self.initialize()

    def initialize(self):
        self.h5_file = h5py.File(self.h5_path, "r")

#        self.event_ids  = np.array(self.h5_file["event_ids"])
#        self.root_files = np.array(self.h5_file["root_files"])
        self.labels     = np.array(self.h5_file["labels"])
#        self.positions  = np.array(self.h5_file["positions"])
#        self.angles     = np.array(self.h5_file["angles"])
#        self.energies   = np.array(self.h5_file["energies"])
#        if "veto" in self.h5_file.keys():
#            self.veto  = np.array(self.h5_file["veto"])
#            self.veto2 = np.array(self.h5_file["veto2"])
        self.event_hits_index = np.append(self.h5_file["events_hits_index"], self.h5_file["hit_pmt"].shape[0]).astype(np.int64)

        self.hdf5_hit_pmt  = self.h5_file["hit_pmt"]
        self.hdf5_hit_time = self.h5_file["hit_time"]

        self.hit_pmt = np.memmap(self.h5_path, mode="r", shape=self.hdf5_hit_pmt.shape,
                                 offset=self.hdf5_hit_pmt.id.get_offset(),
                                 dtype=self.hdf5_hit_pmt.dtype)

        self.time = np.memmap(self.h5_path, mode="r", shape=self.hdf5_hit_time.shape,
                              offset=self.hdf5_hit_time.id.get_offset(),
                              dtype=self.hdf5_hit_time.dtype)
        self.load_hits()

        # Set attribute so that method won't be invoked again
        self.initialized = True

    @abstractmethod
    def load_hits(self):
        pass

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, item):
        if not self.initialized:
            self.initialize()

        data_dict = {
            "labels": self.labels[item].astype(np.int64)-1,
#            "energies": self.energies[item],
#            "angles": self.angles[item],
#            "positions": self.positions[item],
#            "event_ids": self.event_ids[item],
#            "root_files": self.root_files[item],
            "indices": item
        }
        return data_dict


class H5Dataset(H5CommonDataset, ABC):
    """
    Initialize digihits dataset.  Adds access to digitized hits data.  These are:
    hit_charge 	(n_hits,) 	float32 	Charge of the digitized hit
    """
    def __init__(self, h5_path):
        H5CommonDataset.__init__(self, h5_path)

    def load_hits(self):
        self.hdf5_hit_charge = self.h5_file["hit_charge"]
        self.hit_charge = np.memmap(self.h5_path, mode="r", shape=self.hdf5_hit_charge.shape,
                                    offset=self.hdf5_hit_charge.id.get_offset(),
                                    dtype=self.hdf5_hit_charge.dtype)

    def __getitem__(self, item):
        data_dict = super().__getitem__(item)

        start = self.event_hits_index[item]
        stop = self.event_hits_index[item + 1]

        self.event_hit_pmts = self.hit_pmt[start:stop]
        self.event_hit_charges = self.hit_charge[start:stop]
        self.event_hit_times = self.time[start:stop]

        return data_dict

class H5TrueDataset(H5CommonDataset, ABC):
    """
    Initializes truehits dataset. Adds access to true photon hits data. These are:
    hit_parent 	(n_hits,) 	float32 	Parent track ID of the true hit, as defined by WCSim's true hit parent. -1 is used for dark noise.
    """
    def __init__(self, h5_path, transforms=None, digitize_hits=True):
        H5CommonDataset.__init__(self, h5_path, transforms)
        self.digitize_hits = digitize_hits

    def load_hits(self):
        self.all_hit_parent = self.h5_file["hit_parent"]
        self.hit_parent = np.memmap( self.h5_path, mode="r", shape=self.all_hit_parent.shape,
                              offset=self.all_hit_parent.id.get_offset(),
                              dtype=self.all_hit_parent.dtype)

    def digitize(self, truepmts, truetimes, trueparents):
        """
        Replace below with a real digitization.  For now take time closest to zero as time, and sum of photons as charge.
        """
        pmt_time_dict = { pmt: truetimes[ truepmts==pmt ] for pmt in truepmts }
        pmt_photons_dict = { pmt : len(truetimes[ truepmts==pmt]) for pmt in truepmts }
        pmt_mintimes_dict = { pmt : min( abs(truetimes[ truepmts==pmt]) )   for pmt in truepmts }

        timeoffset = 950.0
        allpmts  = np.array( list(pmt_photons_dict.keys()) )
        alltimes = np.array( list(pmt_mintimes_dict.values()) ) + timeoffset
        allcharges = np.array( list(pmt_photons_dict.values()) )
        return allpmts, alltimes, allcharges

    def __getitem__(self, item):
        data_dict = super().__getitem__(item)

        start = self.event_hits_index[item]
        stop = self.event_hits_index[item + 1]

        true_pmts    = self.hit_pmt[start:stop].astype(np.int16)
        true_times   = self.time[start:stop]
        true_parents = self.hit_parent[start:stop]

        if self.digitize_hits:
            self.event_hit_pmts, self.event_hit_times, self.event_hit_charges = self.digitize(true_pmts, true_times, true_parents)
        else:
            self.event_hit_pmts = true_pmts
            self.event_hit_times = true_times
            self.event_hit_parents = true_parents

        return data_dict


def get_transformations(transformations, transform_names):
    if transform_names is not None:
        for transform_name in transform_names:
            assert hasattr(transformations, transform_name), f"Error: There is no defined transform named {transform_name}"
        transform_funcs = [getattr(transformations, transform_name) for transform_name in transform_names]
        return transform_funcs
    else:
        return None

def apply_random_transformations(transforms, data, segmented_labels = None):
    if transforms is not None:
        for transformation in transforms:
            if random.getrandbits(1):
                data = transformation(data)
                if segmented_labels is not None:
                    segmented_labels = transformation(segmented_labels)
    return data


barrel_map_array_idxs = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 15, 16, 17, 12, 13, 14, 18]
pmts_per_mpmt = 19

class CNNmPMTDataset(H5Dataset):
    def __init__(self, h5file, mpmt_positions_file, padding_type=None, transforms=None, collapse_arrays=False):
        """
        Args:
            h5_path             ... path to h5 dataset file
            transforms          ... transforms to apply
            collapse_arrays     ... whether to collapse arrays in return
        """
        super().__init__(h5file)


        self.mpmt_positions = np.load(mpmt_positions_file)['mpmt_image_positions']
        self.data_size = np.max(self.mpmt_positions, axis=0) + 1
        self.barrel_rows = [row for row in range(self.data_size[0]) if
                            np.count_nonzero(self.mpmt_positions[:,0] == row) == self.data_size[1]]
        n_channels = pmts_per_mpmt
        self.data_size = np.insert(self.data_size, 0, n_channels)
        self.collapse_arrays = collapse_arrays
        self.transforms = get_transformations(self, transforms)

        if padding_type is not None:
            self.padding_type = getattr(self, padding_type)
        else:
            self.padding_type = lambda x : x

        self.horizontal_flip_mpmt_map=[0, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 12, 17, 16, 15, 14, 13, 18]
        self.vertical_flip_mpmt_map=[6, 5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 15, 14, 13, 12, 17, 16, 18]


    def process_data(self, hit_pmts, hit_data):
        """
        Returns event data from dataset associated with a specific index
        Args:
            hit_pmts                ... array of ids of hit pmts
            hid_data                ... array of data associated with hits

        Returns:
            data                    ... array of hits in cnn format
        """
        hit_mpmts = hit_pmts // pmts_per_mpmt
        hit_pmt_in_modules = hit_pmts % pmts_per_mpmt

        hit_rows = self.mpmt_positions[hit_mpmts, 0]
        hit_cols = self.mpmt_positions[hit_mpmts, 1]

        data = np.zeros(self.data_size, dtype=np.float32)
        data[hit_pmt_in_modules, hit_rows, hit_cols] = hit_data

        # fix barrel array indexing to match endcaps in xyz ordering
        barrel_data = data[:, self.barrel_rows, :]
        data[:, self.barrel_rows, :] = barrel_data[barrel_map_array_idxs, :, :]

        # collapse arrays if desired
        if self.collapse_arrays:
            data = np.expand_dims(np.sum(data, 0), 0)

        return data

    def  __getitem__(self, item):

        data_dict = super().__getitem__(item)

        processed_data = from_numpy(self.process_data(self.event_hit_pmts, self.event_hit_charges))

        processed_data = apply_random_transformations(self.transforms, processed_data)

        processed_data = self.padding_type(processed_data)

        data_dict["data"] = processed_data.double()

        return data_dict['data'], data_dict['labels']


    def horizontal_flip(self, data):
        return flip(data[self.horizontal_flip_mpmt_map, :, :], [2])

    def vertical_flip(self, data):
        return flip(data[self.vertical_flip_mpmt_map, :, :], [1])

    def flip_180(self, data):
        return self.horizontal_flip(self.vertical_flip(data))

    def front_back_reflection(self, data):
        """
        Returns an image with horizontal flip of the left and right halves of the barrels and
        vertical flip of the endcaps
        Specs in transform.pdf
        :param data : torch.tensor
        :returns transform_data: torch.tensor
        """

        barrel_row_start, barrel_row_end = self.barrel_rows[0], self.barrel_rows[-1]
        radius_endcap = barrel_row_start//2                     # 5
        half_barrel_width = data.shape[2]//2                    # 20
        l_endcap_index = half_barrel_width - radius_endcap      # 15
        r_endcap_index = half_barrel_width + radius_endcap      # 25

        transform_data = data.clone()

        # Take out the left and right halves of the barrel
        left_barrel = data[:, self.barrel_rows, :half_barrel_width]
        right_barrel = data[:, self.barrel_rows, half_barrel_width:]
        # Horizontal flip of the left and right halves of barrel
        transform_data[:, self.barrel_rows, :half_barrel_width] = self.horizontal_flip(left_barrel)
        transform_data[:, self.barrel_rows, half_barrel_width:] = self.horizontal_flip(right_barrel)

        # Take out the top and bottom endcaps
        top_endcap = data[:, :barrel_row_start, l_endcap_index:r_endcap_index]
        bottom_endcap = data[:, barrel_row_end+1: , l_endcap_index:r_endcap_index]
        # Vertical flip of the top and bottom endcaps
        transform_data[:, :barrel_row_start, l_endcap_index:r_endcap_index] = self.vertical_flip(top_endcap)
        transform_data[:, barrel_row_end+1: , l_endcap_index:r_endcap_index] = self.vertical_flip(bottom_endcap)

        return transform_data


    def rotation180(self, data):
        """
        Returns an image with horizontal and vertical flip of the endcaps and
        shifting of the barrel rows by half the width
        Specs in transforms.pdf
        :param data : torch.tensor
        :returns transform_data: torch.tensor
        """
        barrel_row_start, barrel_row_end = self.barrel_rows[0], self.barrel_rows[-1]   # 10,18 respectively
        radius_endcap = barrel_row_start//2                 # 5
        l_endcap_index = data.shape[2]//2 - radius_endcap   # 15
        r_endcap_index = data.shape[2]//2 + radius_endcap   # 25

        transform_data = data.clone()

        # Take out the top and bottom endcaps
        top_endcap = data[:, :barrel_row_start, l_endcap_index:r_endcap_index]
        bottom_endcap = data[:, barrel_row_end+1: , l_endcap_index:r_endcap_index]
        # Vertical and horizontal flips of the endcaps
        transform_data[:, :barrel_row_start, l_endcap_index:r_endcap_index] = self.flip_180(top_endcap)
        transform_data[:, barrel_row_end+1: , l_endcap_index:r_endcap_index] = self.flip_180(bottom_endcap)

        # Swap the left and right halves of the barrel
        transform_data[:,self.barrel_rows, :] = torch.roll(transform_data[:, self.barrel_rows, :], 20, 2)

        return transform_data


    def mpmtPadding(self, data):
        """
        :param data: torch.tensor
        :returns transform_data: torch.tensor
        """
        w = data.shape[2]
        barrel_row_start, barrel_row_end = self.barrel_rows[0], self.barrel_rows[-1]
        l_endcap_index = w//2 - 5
        r_endcap_index = w//2 + 4

        padded_data = torch.cat((data, torch.zeros_like(data[:, :, :w//2])), dim=2)
        padded_data[:, self.barrel_rows, w:] = data[:, self.barrel_rows, :w//2]

        # Take out the top and bottom endcaps
        top_endcap = data[:, :barrel_row_start, l_endcap_index:r_endcap_index+1]
        bottom_endcap = data[:, barrel_row_end+1: , l_endcap_index:r_endcap_index+1]

        padded_data[:, :barrel_row_start, l_endcap_index+w//2:r_endcap_index+w//2+1] = self.flip_180(top_endcap)
        padded_data[:, barrel_row_end+1:, l_endcap_index+w//2:r_endcap_index+w//2+1] = self.flip_180(bottom_endcap)

        return padded_data


    def double_cover(self, data):
        """
        Specs in double_cover.pdf
        param data: torch.tensor
        returns padded_data: torch.tensor
        """
        w = data.shape[2]
        barrel_row_start, barrel_row_end = self.barrel_rows[0], self.barrel_rows[-1]
        radius_endcap = barrel_row_start//2                                                    # 5
        half_barrel_width, quarter_barrel_width = w//2, w//4                                   # 20, 10

        # Step - 1 : Roll the tensor so that the first quarter is the last quarter
        padded_data = torch.roll(data, -quarter_barrel_width, 2)

        # Step - 2 : Copy the endcaps and paste 3 quarters from the start, after flipping 180
        l1_endcap_index = half_barrel_width - radius_endcap - quarter_barrel_width               #  5
        r1_endcap_index = l1_endcap_index + 2*radius_endcap                                       # 15
        l2_endcap_index = l1_endcap_index+half_barrel_width
        r2_endcap_index = r1_endcap_index+half_barrel_width

        top_endcap = padded_data[:, :barrel_row_start, l1_endcap_index:r1_endcap_index]
        bottom_endcap = padded_data[:, barrel_row_end+1: , l1_endcap_index:r1_endcap_index]

        padded_data[:, :barrel_row_start , l2_endcap_index:r2_endcap_index] = self.flip_180(top_endcap)
        padded_data[:, barrel_row_end+1:, l2_endcap_index:r2_endcap_index] = self.flip_180(bottom_endcap)

        # Step - 3 : Rotate the top and bottom half of barrel and concat them to the top and bottom respectively
        barrel_rows_top, barrel_rows_bottom = np.array_split(self.barrel_rows, 2)
        barrel_top_half, barrel_bottom_half = padded_data[:, barrel_rows_top, :], padded_data[:, barrel_rows_bottom, :]

        concat_order = (self.flip_180(barrel_top_half),
                        padded_data,
                        self.flip_180(barrel_bottom_half))

        padded_data = torch.cat(concat_order, dim=1)

        return padded_data


    def retrieve_event_data(self, item):
        """
        Returns event data from dataset associated with a specific index
        Args:
            item                    ... index of event
        Returns:
            hit_pmts                ... array of ids of hit pmts
            pmt_charge_data         ... array of charge of hits
            pmt_time_data           ... array of times of hits
        """
        data_dict = super().__getitem__(item)

        # construct charge data with barrel array indexing to match endcaps in xyz ordering
        pmt_charge_data = self.process_data(self.event_hit_pmts, self.event_hit_charges).flatten()

        # construct time data with barrel array indexing to match endcaps in xyz ordering
        pmt_time_data = self.process_data(self.event_hit_pmts, self.event_hit_times).flatten()

        return self.event_hit_pmts, pmt_charge_data, pmt_time_data


#!/usr/bin/env python

##
# This module provides some standard interface with DAMASK module.
#
import numpy as np


def read_table(file_path):
    """
    @description: read in an ASCII table with DAMASK format.
    @para:
        out_table: {header1: [data1], header2: [data2], ...}
    """
    f_handle = open(file_path, "r")
    raw_data = f_handle.readlines()
    f_handle.close()
    # remove header
    num_header = int(raw_data.pop(0).split()[0])
    if num_header > 1:
        for i in range(num_header - 1):
            raw_data.pop(0)  # remove command history
    header = raw_data.pop(0).split()
    out_table = {}  # {header:[data_vector]}
    # reading in data
    data = [[float(item) for item in line.split()] for line in raw_data]
    data = np.array(data)  # this allows selecting column
    for i in range(len(header)):
        tmp = data[:, i]
        out_table[header[i]] = list(tmp)
    return out_table
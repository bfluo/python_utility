#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os


def get_filename_list_from_dir(in_dir, suffix="", ignore_suffix_list=[], return_dict=False):
    """
    Get filename list from a given directory
    :param in_dir: input directory
    :param suffix: file name suffixes that you would like to remove
    :param ignore_suffix_list: ignore files with these suffixes
    :param return_dict: return dictionary (if true, return a dictionary using file name as key and True as value)
    :return: list of file names
    """
    res_list = []
    for filename in os.listdir(in_dir):
        for suffix_ignored in ignore_suffix_list:
            if filename.endswith(suffix_ignored):
                continue
            filename = filename[:-len(suffix)] if filename.endswith(suffix) else filename
            res_list.append(filename)
    if return_dict:
        res_dict = {}
        for filename in res_list:
            res_dict[filename] = True
        return res_dict
    else:
        return res_list

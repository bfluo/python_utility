#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
try:
    from scipy.spatial import distance
except ImportError, e:
    print("Warning: Failed to import scipy.spatial.distance", file=sys.stderr)
import six
import random
import math
import os
import time
import nltk
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer()


# General Purpose Utility Functions


""""""""""""""""""""""""""""""" Data Set Related """""""""""""""""""""""""""""""


class WordEmbedding:
    def __init__(self, inpath, init_bound=0.001):
        self.word_to_index = {}
        self.index_to_word = {}
        self.embedding_matrix = []
        self.init_bound = init_bound
        self.vec_len = 0
        self.size = 0
        self.load_from_txt(inpath)

    def load_from_txt(self, inpath):
        """
        Load word vector, generate related data structures
        :param inpath:  input file path, one word per line, separated by tab, first element is word, the rest are numbers
        :return:
        """
        infile = open(inpath, "r")
        for line in infile:
            line_split = line.strip().split()
            word = line_split[0]
            self.word_to_index[word] = self.size
            self.index_to_word[self.size] = word
            self.size += 1
            embedding = map(lambda x: float(x), line_split[1:])
            self.embedding_matrix.append(embedding)
        self.vec_len = len(self.embedding_matrix[0])
        infile.close()

    def add_words(self, word_array):
        """
        Add a list of words, the vector of this words will be randomly initialized.
        Note: words that already exist will be ignored
        :param word_array:  the array of words to be added
        :return:
        """
        for word in word_array:
            if word in self.word_to_index:  # already exists
                continue
            self.word_to_index[word] = self.size
            self.index_to_word[self.size] = word
            self.size += 1
            embedding = np.random.uniform(-self.init_bound, self.init_bound, self.vec_len).tolist()
            self.embedding_matrix.append(embedding)

    def persist(self):
        """
        Tell this object that the words will not change afterwards.
        After this operation, this object is ready to use.
        :return:
        """
        self.embedding_matrix = np.array(self.embedding_matrix)

    def dump_word_index(self, outpath):
        with open(outpath, "w") as outfile:
            for key, value in self.word_to_index.iteritems():
                outfile.writelines("{0}\t{1}\n".format(key, value))

    def find_neighbor_word(self, in_word, out_len=20):
        if in_word not in self.word_to_index:
            return []
        in_index = self.word_to_index[in_word]
        src_vec = self.embedding_matrix[in_index]
        sim_array = []
        for i in xrange(len(self.embedding_matrix)):
            if i == in_index:
                continue
            cosine_sim = 1 - distance.cosine(src_vec, self.embedding_matrix[i])
            sim_array.append(cosine_sim)

        sorted_sim_index = sorted(range(len(sim_array)), key=lambda k: sim_array[k], reverse=True)
        res_array = []
        skipped = False
        for i in xrange(out_len):
            temp_index = sorted_sim_index[i]
            if temp_index == in_index:
                skipped = True
                continue
            res_array.append((self.index_to_word[temp_index], sim_array[temp_index]))

        if skipped:
            temp_index = sorted_sim_index[out_len]
            res_array.append((self.index_to_word[temp_index], sim_array[temp_index]))

        return res_array

    def prompt_neighbor(self):
        while True:
            str_split = raw_input("> ").strip().split()
            out_len = 20
            if len(str_split) == 2:
                out_len = int(str_split[1])
            word = str_split[0]
            res_array = self.find_neighbor_word(word, out_len)
            for ele in res_array:
                print(ele)
            print("")


    @staticmethod
    def load_word_index(inpath):
        word_to_index = {}
        index_to_word = {}
        with open(inpath, "r") as infile:
            for line in infile:
                line = line.strip()
                if line == "":
                    break
                line_split = line.split("\t")
                word = line_split[0]
                index = int(line_split[1])
                word_to_index[word] = index
                index_to_word[index] = word
        return word_to_index, index_to_word


def load_stop_words(inpath):
    """
    Load stop word dictionary.
    :param inpath:  input file path, one word per line
    :return:    {stop_word: True}
    """
    res_dic = {}
    with open(inpath, "r") as infile:
        for line in infile:
            if line.strip() == "":  # EOF
                break
            res_dic[line.strip()] = True
    return res_dic


def sent_to_wid(sent, length, word_to_index, padding_symbol, unknown_symbol):
    """
    Convert a sent to word index array
    :param sent: input sentence (space separated words string)
    :param length: word index length
    :param word_to_index:   word to index mapping
    :param padding_symbol:  padding symbol (string, not index)
    :param unknown_symbol:  unknown symbol (string, not index)
    :return:
    """
    tokens = tokenizer.tokenize(sent)
    wid_array = []
    for token in tokens:
        token = token.lower()
        if token in word_to_index:
            wid_array.append(word_to_index[token])
        else:
            wid_array.append(word_to_index[unknown_symbol])
    wid_array = pad_array(wid_array, length, word_to_index[padding_symbol])
    return wid_array


def summarize_multi_class_prediction(prediction, target, neg_id=-1, in_detail=False):
    """
    Given prediction of a model, and the corresponding target, calculate summary statistics
    :param prediction:  predicted label index array
    :param target:  target label index array
    :param neg_id:   negative target id (used to calculate overall precision and recall), default is -1, then precision
    and recall is the same as accuracy
    :param in_detail:   detail version or not
    :return:    summary dictionary (keys: precision, right_cnt, total_cnt),
                detail version has these statistics and recall, F1 for each label
    """
    assert len(prediction) == len(target)
    summary = dict()
    summary["right_cnt"] = 0
    summary["total_cnt"] = len(prediction)
    summary["posi_right_cnt"] = 0
    summary["pred_cnt"] = 0
    summary["posi_total_cnt"] = 0
    if in_detail:
        summary["label_stats"] = {}

    def ensure_label_existence(label):
        if label not in summary["label_stats"]:
            summary["label_stats"][label] = {}
            summary["label_stats"][label]["right_cnt"] = 0
            summary["label_stats"][label]["pred_cnt"] = 0
            summary["label_stats"][label]["total_cnt"] = 0

    for i in xrange(len(prediction)):
        if prediction[i] == target[i]:
            summary["right_cnt"] += 1
            if prediction[i] != neg_id:
                summary["posi_right_cnt"] += 1
        if prediction[i] != neg_id:
            summary["pred_cnt"] += 1
        if target[i] != neg_id:
            summary["posi_total_cnt"] += 1
        if in_detail:
            ensure_label_existence(prediction[i])
            ensure_label_existence(target[i])
            summary["label_stats"][target[i]]["total_cnt"] += 1
            summary["label_stats"][prediction[i]]["pred_cnt"] += 1
            if prediction[i] == target[i]:
                summary["label_stats"][prediction[i]]["right_cnt"] += 1

    summary["accuracy"] = summary["right_cnt"] / summary["total_cnt"]
    summary["precision"] = safe_divide(summary["posi_right_cnt"], summary["pred_cnt"])
    summary["recall"] = safe_divide(summary["posi_right_cnt"], summary["posi_total_cnt"])
    if in_detail:
        for key, value in summary["label_stats"].iteritems():
            value["precision"] = value["right_cnt"] / value["pred_cnt"] if value["pred_cnt"] else 0
            value["recall"] = value["right_cnt"] / value["total_cnt"] if value["total_cnt"] else 0
    return summary


def summarize_multi_label_prediction(prediction, target, in_detail=False, threshold=0.5):
    """
    Given prediction of a model, and the corresponding target, calculate summary statistics
    :param prediction:  (batch_num, label_num), each element represent the probability of each label
    :param target:  (batch_num, label_num) the element of gold label is one
    :param in_detail:   detail version or not
    :param threshold:   the threshold to determine a positive detection
    :return:    summary dictionary (keys: precision, right_cnt, total_cnt),
                detail version has these statistics and recall, F1 for each label
    """
    assert len(prediction) == len(target)
    summary = dict()
    summary["right_cnt"] = 0
    summary["total_cnt"] = len(prediction)
    if in_detail:
        summary["label_stats"] = {}

    def ensure_label_existence(label):
        if label not in summary["label_stats"]:
            summary["label_stats"][label] = {}
            summary["label_stats"][label]["right_cnt"] = 0
            summary["label_stats"][label]["pred_cnt"] = 0
            summary["label_stats"][label]["total_cnt"] = 0

    for i in xrange(len(prediction)):
        for j in xrange(np.array(prediction[i]).shape[0]):
            if target[i][j] == 1 and prediction[i][j] >= threshold:
                summary["right_cnt"] += 1
            if in_detail:
                ensure_label_existence(j)
                if target[i][j] == 1:
                    summary["label_stats"][j]["total_cnt"] += 1
                if prediction[i][j] >= threshold:
                    summary["label_stats"][j]["pred_cnt"] += 1
                if target[i][j] == 1 and prediction[i][j] >= threshold:
                    summary["label_stats"][j]["right_cnt"] += 1

    summary["accuracy"] = summary["right_cnt"] / summary["total_cnt"]
    if in_detail:
        for key, value in summary["label_stats"].iteritems():
            value["precision"] = value["right_cnt"] / value["pred_cnt"] if value["pred_cnt"] else 0
            value["recall"] = value["right_cnt"] / value["total_cnt"] if value["total_cnt"] else 0
    return summary


def summary_to_string(summary, index_to_class):
    """
    Convert multi-class summary to ready-to-print string
    :param summary:     summary dictionary
    :param index_to_class:  index to class name dictionary
    """
    out_str = ""

    accuracy = safe_divide(summary["right_cnt"], summary["total_cnt"])
    precision = safe_divide(summary["posi_right_cnt"], summary["pred_cnt"])
    recall = safe_divide(summary["posi_right_cnt"], summary["posi_total_cnt"])
    out_str += "Accuracy {0:.4f}({4}/{5}), Precision {1:.4f}({6}/{7}), Recall {2:.4f}({8}/{9}), F1 {3:.4f}\n".format(
        accuracy, precision, recall, cal_F1(precision, recall),
        summary["right_cnt"], summary["total_cnt"],
        summary["posi_right_cnt"], summary["pred_cnt"],
        summary["posi_right_cnt"], summary["posi_total_cnt"],
    )

    for class_index in summary["label_stats"]:
        temp_precision = safe_divide(summary["label_stats"][class_index]["right_cnt"], summary["label_stats"][class_index]["pred_cnt"])
        temp_recall = safe_divide(summary["label_stats"][class_index]["right_cnt"], summary["label_stats"][class_index]["total_cnt"])
        out_str += "\t{0} (index {1}): Precision {2:.4f} ({5}/{6}), Recall {3:.4f} ({7}/{8}), F1 {4:.4f}\n".format(
            index_to_class[class_index], class_index, temp_precision, temp_recall, cal_F1(temp_precision, temp_recall),
            summary["label_stats"][class_index]["right_cnt"], summary["label_stats"][class_index]["pred_cnt"],
            summary["label_stats"][class_index]["right_cnt"], summary["label_stats"][class_index]["total_cnt"],
        )
    return out_str.strip()


""""""""""""""""""""""""""""""" Collection Utility """""""""""""""""""""""""""""""


def safe_initialize_key(dic, key, value):
    """
    If dic doesn't contain key, then dic[key]=value
    :param dic:     input dictionary
    :param key:     key to initialize
    :param value:   initial value
    :return:
    """
    if key not in dic:
        dic[key] = value


def safe_add_one(dic, key):
    """
    If dic contains key, dic[key]=0, else dic[key]+=1
    :param dic: input dicitonary
    :param key: key to add one
    :return:
    """
    safe_initialize_key(dic, key, 0)
    dic[key] += 1


def first_element_of_dic(input_dic):
    """
    Return the first element of a dictionary
    :param input_dic:   input dictionary
    :return:    first element, (key, value)
    """
    return six.next(six.iteritems(input_dic))


def randomize_arrays(*array_list):
    """
    Randomize multiple arrays simultaneously with the same index array
    :param array_list:  a list of arrays to randomize
    :return:    randomized array list, index array
    """
    length = len(array_list[0])
    index_array = range(length)
    random.shuffle(index_array)

    res_array_list = []
    for array in array_list:
        new_array = [array[index_array[i]] for i in xrange(length)]
        res_array_list.append(new_array)

    return res_array_list, index_array


def split_array_with_ratio(array, ratio_array, allow_empty_bin=True):
    """
    Split an array with given ratio.
    :param array:   input array
    :param ratio_array:   array of ratio (e.g [3, 2, 1])
    :param allow_empty_bin: if False, will evenly split the array if empty bin exists
    :return:    list of split array
    """
    length = len(array)
    ratio_sum = np.sum(ratio_array)
    split_point = []
    cumulative_ratio = 0
    for ratio in ratio_array:
        cumulative_ratio += ratio
        split_point.append(int(math.ceil(length * cumulative_ratio / ratio_sum)))

    res_array = []
    start_point = 0
    empty_bin = False
    for end_point in split_point:
        res_array.append(array[start_point: end_point])
        if start_point == end_point:
            empty_bin = True
        start_point = end_point

    if not allow_empty_bin and empty_bin:
        bin_size = int(math.floor(len(array) / len(ratio_array)))
        if bin_size == 0:       # array is too small
            return res_array
        start_point = 0
        res_array = []
        for i in xrange(len(ratio_array)):
            end_point = len(array) if i == len(ratio_array)-1 else start_point + bin_size
            res_array.append(array[start_point: end_point])
            start_point = end_point

    return res_array


def pad_array(array, length, padding):
    if len(array) > length:
        array = array[:length]
    elif len(array) < length:
        array += [padding for i in xrange(length - len(array))]
    return array


def eliminate_padding(array, padding_index_array):
    """
    Eleminate padding elements in an array given padding data indices.
    :param array:   can be python array or numpy ndarray
    :param padding_index_array: array of padding data indices
    :return:    cleaned array
    """
    if type(array) == list:     # python array
        res_array = []
        padding_index_dic = {}
        for padding_index in padding_index_array:
            padding_index_dic[padding_index] = True
        for i in xrange(len(array)):
            if i not in padding_index_dic:
                res_array.append(array[i])
        return res_array
    elif type(array == np.ndarray):     # numpy ndarray
        valid_index_array = []
        start_index = 0
        for i in xrange(len(padding_index_array)):
            valid_index_array += range(start_index, padding_index_array[i])
            start_index = padding_index_array[i] + 1
        valid_index_array += range(start_index, len(array))
        return array[valid_index_array]
    else:
        raise TypeError("Given input array of type {0}, list or numpy.ndarray is required.".format(type(array)))


def load_dic_of_txt(inpath, splitter="\t"):
    """
    Assume each line in input file is separated by "splitter", and has either 1 or 2 columns,
    load the first column as key and second column as value (if only 1 column, value is "True")
    :param inpath:  input file path
    :return     dictionary (key is the first column, value is the second column (or "True" if only 1 column))
    """
    res_dic = {}
    with open(inpath, "r") as infile:
        while True:
            line = infile.readline().strip()
            if line == "":
                break
            line_split = line.split(splitter)
            res_dic[line_split[0]] = line_split[1]
    return res_dic


def get_range_index(lower_bound, upper_bound, additional_symbols=[]):
    """
    Chagne a range of integers to index
    :param lower_bound: lower bound of the range
    :param upper_bound: upper bound of the range
    :param additional_symbols:  additional symbols to add
    :return: integer_to_index, index_to_integer
    """
    integer_array = range(lower_bound, upper_bound)
    integer_to_index = {}
    index_to_integer = {}
    cnt = 0
    for integer in integer_array:
        integer_to_index[integer] = cnt
        index_to_integer[cnt] = integer
        cnt += 1
    for integer in additional_symbols:
        integer_to_index[integer] = cnt
        index_to_integer[cnt] = integer
        cnt += 1
    return integer_to_index, index_to_integer


def get_index_map(in_array):
    """
    Get element_to_index and index_to_element mapping dictionary given an array of elements to map.
    :param in_array:    input array of elements to map
    :return:    element_to_index, index_to_element
    """
    ele_to_index = {}
    index_to_ele = {}
    cnt = 0
    for ele in in_array:
        ele_to_index[ele] = cnt
        index_to_ele[cnt] = ele
        cnt += 1
    return ele_to_index, index_to_ele


def change_encoding(in_obj, in_encoding="unicode", out_encoding="utf8"):
    """
    Convert an string or array of string of any encoding to another encoding
    :param in_obj:   input string or string array
    :param in_encoding:     encoding of the input
    :param out_encoding:    encoding of the output
    :return     utf-8 string array
    """
    if in_encoding == out_encoding:
        return in_obj
    if isinstance(in_obj, str):
        if in_encoding == "unicode":
            return in_obj.encode(out_encoding)
        elif out_encoding == "unicode":
            return in_obj.decode(in_encoding)
        else:
            return in_obj.decode(in_encoding).encode(out_encoding)
    elif isinstance(in_obj, list):
        if in_encoding == "unicode":
            for i in xrange(len(in_obj)):
                in_obj[i] = in_obj[i].encode(out_encoding)
        elif out_encoding == "unicode":
            for i in xrange(len(in_obj)):
                in_obj[i] = in_obj[i].decode(in_encoding)
        else:
            for i in xrange(len(in_obj)):
                in_obj[i] = in_obj[i].decode(in_encoding).encode(out_encoding)
        return in_obj
    else:
        raise TypeError("Expected {0} or {1} of input object, got {2} instead.".format(str, list, type(in_obj)))



""""""""""""""""""""""""""""""" Math Utiliy """""""""""""""""""""""""""""""


def range_overlap(range_1, range_2):
    """
    Judge if two range overlaps
    :param range_1: first range (format: [start, end])
    :param range_2: second range (format: [start, end])
    :return:    True if overlaps, False otherwise
    """
    if range_1[0] <= range_2[0] <= range_1[1]:
        return True
    if range_2[0] <= range_1[0] <= range_2[1]:
        return True
    return False


def safe_divide(dividend, divisor):
    """
    Behave like normal division when divisor is non-zero.
    Return 0 when divisor is zero
    :param dividend:    the number to be divided
    :param divisor:     the number by which the dividend is divided
    :return:
    """
    return dividend / divisor if divisor else 0


""""""""""""""""""""""""""""""" File Utiliy """""""""""""""""""""""""""""""


def load_str_until_empty_line(infile, sp_line_dic={}, default_mark="no_mark"):
    """
    Load the string until encounter an empty line (empty line is read off the input stream)
    If special lines in sp_line_dic values appear, output string will be marked as the corresponding key in sp_line_dic,
    otherwise it will be marked as default_mark
    :param infile:  input stream
    :param sp_line_dic: dictionary of special lines and corresponding keys
    :return:    (string, mark)
    """
    res_str = ""
    res_mark = default_mark
    while True:
        line = infile.readline().strip()
        if line == "":
            break
        res_str += line + "\n"

        if res_mark == default_mark:
            for k, v in sp_line_dic.iteritems():
                if line == v:
                    res_mark = k

    return res_str, res_mark


def cal_F1(precision, recall):
    """
    Calculate F1 from precision and recall
    :param precision:   precision
    :param recall:  recall
    :return:
    """
    return safe_divide(2 * precision * recall, (precision + recall))


def add_suffix_to_file_name(file_name, suffix):
    """
    Add a suffix to a file name (can be a file path), and the extension is retained.
    :param file_name:   raw file name
    :param suffix:  suffix to add
    :return:    new file name
    """
    name_split = file_name.split(".")
    if len(name_split) > 2:
        raise ValueError("The file name is invalid, should only contain at most one dot in tail.\nFile Name: {0}".format(
            file_name
        ))
    new_name = name_split[0] + suffix + "." + name_split[1] if len(name_split) ==2 else name_split[0] + suffix
    return new_name


""""""""""""""""""""""""""""""" String Utiliy """""""""""""""""""""""""""""""


def unicode_to_utf8(in_obj):
    if type(in_obj) == dict:
        return unicode_to_utf8_dic(in_obj)
    elif type(in_obj) == list:
        return unicode_to_utf8_list(in_obj)
    elif type(in_obj) == unicode:
        return in_obj.encode("utf8")
    elif type(in_obj) == str:
        return in_obj
    else:
        raise TypeError("Only support type: dict, list, unicode, str")


def unicode_to_utf8_dic(in_dic):
    out_dic = {}
    for key, value in in_dic.iteritems():
        res_key = key.encode("utf8") if type(key) == unicode else key
        if type(value) == dict:
            res_value = unicode_to_utf8_dic(value)
        elif type(value) == list:
            res_value = unicode_to_utf8_list(value)
        elif type(value) == unicode:
            res_value = value.encode("utf8")
        else:
            res_value = value
        out_dic[res_key] = res_value

    return out_dic


def unicode_to_utf8_list(in_list):
    out_list = []

    for value in in_list:
        if type(value) == unicode:
            res_value = value.encode("utf8")
        elif type(value) == list:
            res_value = unicode_to_utf8_list(value)
        elif type(value) == dict:
            res_value = unicode_to_utf8_dic(value)
        else:
            res_value = value
        out_list.append(res_value)

    return out_list


if __name__ == "__main__":
    pass

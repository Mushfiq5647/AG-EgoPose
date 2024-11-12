# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import argparse
import os
import pickle
import json


class Annotation(object):
    """ Simple vocabulary wrapper """

    def __init__(self):
        self.anns = []

    def __len__(self):
        return len(self.anns)

    def addSequence(path, range):
        anns.append((path, range))


def build_annotation(dataset):
    annotation = Annotation()
    kinect_categories = {"patty1": 1957,
                         "patty2": 1799,
                         "patty5": 2006,
                         "patty26": 2304,
                         "patty27": 934,
                         "patty28": 712,
                         "patty30": 2063,
                         "catch36": 1656,
                         "catch37": 2128,
                         "catch39": 3530,
                         "catch40": 1360,
                         "convo43": 3010,
                         "convo46": 3610,
                         "convo47": 3980,
                         "convo53": 2323,
                         "sport56": 2934,
                         "sport58": 4913}

    cmu_categories = {"1-catch1": 3477,
                      "2-catch2": 3477,
                      "4-convo1": 2594,
                      "5-convo2": 2736,
                      "6-convo3": 2751,
                      "10-hand1": 2738,
                      "13-sports1": 2334}

    sceneego_categories = {"diogo1": 2082,
                           "diogo2": 2174,
                           "jian1": 3748,
                           "jian2": 3972,
                           "pranay2": 2862
                           }
    # "mengyu_new/out6": 3500
    # # "zhili_new/kitchen1": 3200,
    # "zhili_new/kitchen2": 3500,
    # "zhili_new/office1": 3500,
    # "zhili_new/office2": 3400,
    # "zhili_new/rountunda4": 1400

    egopw_categories = {
                        "binchen/kitchen": 1600, "binchen/office": 3000, "binchen/out": 2100,
                        "binchen/rountunda1": 2400, "binchen/rountunda2": 2300,
                        "chao/kitchen1": 1800, "chao/kitchen2": 2000, "chao/rountunda1": 2000,
                        "chao/rountunda2": 2000, "chao/rountunda3": 1400, "chao_new/office1": 2800, "chao_new/office2": 2300,
                        "lingjie/rountunda1": 3800, "mengyu_new/kitchen1": 3700, "mengyu_new/kitchen2": 4200,
                        "mengyu_new/office1": 4000,
                        "mengyu_new/office2": 3300,
                        "mengyu_new/out1": 3600,
                        "mengyu_new/out4": 4200,
                        "mengyu_new/rest2": 2400, "mengyu_new/rountunda1": 2800, "mengyu_new/rountunda2": 2900,
                        "soshi_new/kitchen": 2000, "soshi_new/kitchen2": 1900, "soshi_new/office1": 3200,
                        "soshi_new/office2": 2900, "soshi_new/out": 3200, "soshi_new/rountunda1": 2300,
                         "soshi_new/rountunda2": 2200,
                        "zhili_new/out1": 3800,
                         "zhili_new/out2": 3600,
                        "zhili_new/rest1": 2400, "zhili_new/rest2": 2600, "zhili_new/rountunda1": 2500,
                        "zhili_new/rountunda2": 2000, "zhili_new/rountunda3": 1100}

    if dataset == 'cmu':
        categories = cmu_categories
    elif dataset == 'sceneego':
        categories = sceneego_categories
    elif dataset == 'egopw':
        categories = egopw_categories
    else:
        categories = kinect_categories

    for cat in categories:
        count = 0
        count = categories[cat]
        while (count - 256) > 0:
            annotation.anns.append((cat, count))
            count -= 32
        annotation.anns.append((cat, 257))
    return annotation


def main(args):
    anns = build_annotation(args.dataset)
    print("num sequences:", len(anns))
    annotation_path = args.annotation_path
    with open(annotation_path, 'wb') as f:
        pickle.dump(anns, f)

    with open(annotation_path, 'rb') as f:
        train_annotations = pickle.load(f)

    annotations_dict = {
        "annotations": train_annotations.anns
    }

    with open('annotations.json', 'w') as f:
        json.dump(annotations_dict, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', type=str, required=True, help='path to base dir for all captures')
    parser.add_argument('--dataset', type=str, required=True, help='kinect or cmu')
    args = parser.parse_args()
    main(args)

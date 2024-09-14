# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import argparse
import os
import pickle
import json


class TestAnnotation(object):
    """ Simple vocabulary wrapper """

    def __init__(self):
        self.test_anns = []

    def __len__(self):
        return len(self.test_anns)


def build_test_annotation():
    test_annotation = TestAnnotation()
    test_categories = {"patty1":1957,
                       "patty2":1799,
                       "patty5":2006,
                       "patty32":2657,
                       "patty34":1787,
                       "patty35":1421,
                       "catch55":2536,
                       "convo53":2323,
                       "convo54":2808,
                       "convo59":2887,
                       "sport57":3732}

    for cat in test_categories:
        count = 0
        count = test_categories[cat]
        while (count - 256) > 0:
            test_annotation.test_anns.append((cat, count))
            count -= 256
        test_annotation.test_anns.append((cat, 257))
    return test_annotation


def main(args):
    test_anns = build_test_annotation()
    print("num sequences:", len(test_anns))
    test_annotation_path = args.test_annotation_path
    with open(test_annotation_path, 'wb') as f:
        pickle.dump(test_anns, f)

    with open(test_annotation_path, 'rb') as f:
        test_annotations = pickle.load(f)

    annotations_dict = {
        "annotations": test_annotations.test_anns
    }

    with open('test_annotations.json', 'w') as f:
        json.dump(annotations_dict, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_annotation_path', type=str, required=True, help='path to base dir for all captures')
    args = parser.parse_args()
    main(args)


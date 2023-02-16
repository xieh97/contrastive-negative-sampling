import dbm
import json
import os
import shelve
from ast import literal_eval
from dbm import dumb

import numpy as np
import pandas as pd

dbm._defaultmod = dumb
dbm._modules = {"dbm.dumb": dumb}

# Trial info
trial_base = "~"
trial_series = "~"
trial_name = "~"
ckp_dir = "~"

# Model checkpoint directory
ckp_fpath = os.path.join(trial_base, trial_series, trial_name, ckp_dir)

# Load trial parameters
conf_fpath = os.path.join(trial_base, trial_series, trial_name, "params.json")
with open(conf_fpath, "rb") as store:
    conf = json.load(store)
print("Load", conf_fpath)

data_conf = conf["data_conf"]


def calculate_metrics(qid2items):
    # Retrieval metrics over sample queries
    # i.e., recall@{1, 5, 10}, mAP.
    qid_R1s, qid_R5s, qid_R10s, qid_mAPs = [], [], [], []

    # Recommendation metrics over samples
    # i.e., hit rate@{1, 5, 10}.
    qid_HR1s, qid_HR5s, qid_HR10s = [], [], []

    for qid in qid2items:
        items = qid2items[qid]

        objects = [i[0] for i in items]
        scores = np.array([i[1] for i in items])
        targets = np.array([i[2] for i in items])

        desc_indices = np.argsort(scores, axis=-1)[::-1]
        targets = np.take_along_axis(arr=targets, indices=desc_indices, axis=-1)

        # Recall at cutoff K
        R1 = np.sum(targets[:1], dtype=float) / np.sum(targets, dtype=float)
        R5 = np.sum(targets[:5], dtype=float) / np.sum(targets, dtype=float)
        R10 = np.sum(targets[:10], dtype=float) / np.sum(targets, dtype=float)

        qid_R1s.append(R1)
        qid_R5s.append(R5)
        qid_R10s.append(R10)

        # Mean average precision
        positions = np.arange(1, len(targets) + 1, dtype=float)[targets > 0]

        if len(positions) > 0:
            precisions = np.divide(np.arange(1, len(positions) + 1, dtype=float), positions)
            avg_precision = np.sum(precisions, dtype=float) / np.sum(targets, dtype=float)
            qid_mAPs.append(avg_precision)
        else:
            qid_mAPs.append(0.0)

        # Hit rate at cutoff K
        HR1 = np.sum(targets[:1], dtype=float) > 0.
        HR5 = np.sum(targets[:5], dtype=float) > 0.
        HR10 = np.sum(targets[:10], dtype=float) > 0.

        qid_HR1s.append(HR1)
        qid_HR5s.append(HR5)
        qid_HR10s.append(HR10)

    print("mAP: {:.3f}".format(np.mean(qid_mAPs)),
          "R1: {:.3f}".format(np.mean(qid_R1s)),
          "R5: {:.3f}".format(np.mean(qid_R5s)),
          "R10: {:.3f}".format(np.mean(qid_R10s)),
          "HR1: {:.3f}".format(np.mean(qid_HR1s)),
          "HR5: {:.3f}".format(np.mean(qid_HR5s)),
          "HR10: {:.3f}".format(np.mean(qid_HR10s)), end="\n")


for name in data_conf:
    params = data_conf[name]
    name = name.replace("_data", "")

    # Load text data
    text_fpath = os.path.join(params["dataset"], params["text_data"])
    text_data = pd.read_csv(text_fpath, converters={"tokens": literal_eval})
    print("Load", text_fpath)

    # Process text data
    tid2fid, tid2text, fid2fname = {}, {}, {}
    for idx in text_data.index:
        item = text_data.iloc[idx]
        tid2fid[item["tid"]] = item["fid"]
        tid2text[item["tid"]] = item["text"]
        fid2fname[item["fid"]] = item["fname"]

    # Load cross-modal scores
    score_fpath = os.path.join(ckp_fpath, f"{name}_xmodal_scores.db")
    with shelve.open(filename=score_fpath, flag="r", protocol=2) as stream:
        fid2items, tid2items = {}, {}
        for fid in stream:
            group_scores = stream[fid]

            # Audio2Text retrieval
            fid2items[fid] = [(tid, group_scores[tid], tid2fid[tid] == fid) for tid in group_scores]

            # Text2Audio retrieval
            for tid in group_scores:
                if tid2items.get(tid, None) is None:
                    tid2items[tid] = [(fid, group_scores[tid], tid2fid[tid] == fid)]
                else:
                    tid2items[tid].append((fid, group_scores[tid], tid2fid[tid] == fid))
        # Print info
        print("Audio2Text retrieval")
        calculate_metrics(fid2items)

        print("Text2Audio retrieval")
        calculate_metrics(tid2items)

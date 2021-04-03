import pickle
import sys
from pathlib import Path

import pandas as pd

# add "group" variable to train to create folds that
# don't contain segments from the same audio file
if __name__ == "__main__":
    DATA_DIR = Path(sys.argv[1])

    train = pd.read_csv(
        DATA_DIR / "train_ground_truth.csv",
        header=None,
        names=["id", "labels"],
        index_col="id",
    )

    train["group"] = 0

    with open("./distribution-train-out.txt") as f:
        f.readline()  # skip header
        line = f.readline()
        group = 0
        while line:
            idxs = [int(i) for i in line.strip().split(",")]
            train.loc[idxs, "group"] = group
            group += 1
            line = f.readline()

    with open(Path("/wdata/train-groups.pkl"), "wb") as f:
        pickle.dump(train, f)

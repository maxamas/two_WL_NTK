import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from utils import load_kernelmatrix_from_blocks
import config


def kernel_SVM(kernel_matrix, ys):
    C_list = np.logspace(-2, 4, 120)
    svc = SVC(kernel="precomputed", cache_size=16000, max_iter=1000000)
    clf = GridSearchCV(
        svc,
        {"C": C_list},
        verbose=0,
        return_train_score=True,
        n_jobs=6,
    )
    clf.fit(kernel_matrix, ys)

    return clf, pd.DataFrame(clf.cv_results_)


if __name__ == "__main__":

    dataset_names = [
        "MUTAG",
        "PTC_MR",
        "COX2",
        "DHFR",
    ]

    results = df = pd.DataFrame(
        columns=["Dataset Name", "NN Type", "mean_test_score", "std_test_score"]
    )

    for dataset_name in config.dataset_names:
        for nn_type in config.nn_types:

            kernel_path = f"Data/Kernels/{dataset_name}/{nn_type}/L_10"

            kernel_matrix, ys = load_kernelmatrix_from_blocks(kernel_path)
            ys = np.array(ys, "int32")

            # estimate kernel svm
            svc, r = kernel_SVM(kernel_matrix, ys)
            r = r[r["mean_test_score"] == r["mean_test_score"].max()][
                ["mean_test_score", "std_test_score"]
            ]
            r = r.drop_duplicates()
            r["Dataset Name"] = dataset_name
            r["NN Type"] = nn_type

            results = results.append(r)

            print(results)

    results.to_csv("Results/NTK_L_10.csv")

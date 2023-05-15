"""
Experiment code
===============

This module contains the code for running the experiment that can test the alignment quality.
"""


def align_experiment(csv_path: str) -> None:
    """
    Run the align experiment on one csv.

    Parameters
    ----------
    csv_path : str
        Path to the csv file containing the segments.

    Returns
    -------
    (left, right) : tuple
    """
    import pandas as pd

    from kaldialign import align

    EPS = "*"

    df = pd.read_csv(csv_path)

    lefts, rights = [], []
    for i, row in df.iterrows():
        alignment = align(row["text"], row["model_output"], EPS)
        left = "".join([x[0] if x[1] != " " else " " for x in alignment])
        right = "".join([x[1] if x[0] != " " else " " for x in alignment])
        lefts.append(left)
        rights.append(right)

    return (lefts, rights)


def align_experiment_folder(folder_path: str, verbose=False, quiet=False) -> None:
    """
    Run the align experiment on all csv files in a folder.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing the csv files.

    verbose : bool
        If True will print for debugging.

    quiet : bool
        Do not print at all.

    Returns
    -------
    None
    """
    from glob import glob

    filenames = [x for x in glob(folder_path + "/**/segments.csv", recursive=True)]
    for filename in filenames:
        if "Den2Radio" in filename:
            continue
        i = 0
        bad_alignment = 0
        good_alignment = 0
        bad_model = 0
        outliers = 0
        lefts, rights = align_experiment(filename)
        for left, right in zip(lefts, rights):
            text_eps_ratio = left.count("*") / len(left)
            model_eps_ratio = right.count("*") / len(right)

            if text_eps_ratio > 0.1 and model_eps_ratio < 0.1:
                if verbose and not quiet:
                    print("Example of bad alignment:")
                bad_alignment += 1
            elif text_eps_ratio < 0.1 and model_eps_ratio < 0.1:
                if verbose and not quiet:
                    print("Example of good alignment:")
                good_alignment += 1
            elif text_eps_ratio < 0.1 and model_eps_ratio > 0.1:
                if verbose and not quiet:
                    print("Example of bad model:")
                bad_model += 1
            else:
                if verbose and not quiet:
                    print("Outlier:")
                outliers += 1

            i += 1
            if verbose and not quiet:
                print("Text:  " + left)
                print("Model: " + right)
                print(f"Number of '*' in text:  {left.count('*')}")
                print(f"Epsilon ratio in text:  {text_eps_ratio:.2f}")
                print(f"Number of '*' in model: {right.count('*')}")
                print(f"Epsilon ratio in model: {model_eps_ratio:.2f}")
                print()
        if not quiet:
            print(f"Bad alignment:  {bad_alignment/i:.2f} %")
            print(f"Good alignment: {good_alignment/i:.2f} %")
            print(f"Bad model:      {bad_model/i:.2f} %")
            print(f"Outliers:       {outliers/i:.2f} %")

"""
Main way of interacting with this package.
"""

from rifsalignment.base import BaseAlignment
from rifsalignment.datamodels import TimedSegment

from typing import List


def align_csv(
    data_path: str,
    align_method: BaseAlignment,
    model: str,
    target_path: str = None,
    verbose: bool = False,
    quiet: bool = False,
    max_duration: float = 15,
):
    """
    Adapter fucntion. Primarily serves to make rifsalignment compatible with the rifs CLI.

    Parameters
    ----------
    data_path: str
        The path to the csv file containing the audio and text files to align. Should of the rifs format.
    align_method: BaseAlignment
        The method to use for alignment. Can be one of the following: CTC
    model: str
        The path to the model to use for alignment. Can be a huggingface model or a local path.
    target_path: str
        The path to the folder to save the aligned files to.
        If None, the files will be saved to the same folder as the csv file.
    verbose: bool
        Whether to print the alignments progress with steps.
    quiet: bool
        Prints nothing.
    max_duration: float
        The maximum length of a segment in seconds.
    """
    import os
    import pandas as pd

    from glob import glob

    # Load csv file
    all_csv = pd.read_csv(os.path.join(data_path, "all.csv"), header=0)

    if target_path is None:
        target_path = os.path.join(data_path, "alignments")
        os.makedirs(target_path, exist_ok=True)

    if "id" in all_csv:
        # Retrieve audio and text files and parse to align
        for i, row in all_csv.iterrows():
            if os.path.exists(os.path.join(target_path, row["id"])):
                if verbose and not quiet:
                    print(f"Skipping {row['id']} as it already exists")
                continue

            if verbose and not quiet:
                print(f"Aligning {row['id']}")

            alignments = align_method.align(
                audio_file=os.path.join(data_path, "audio", row["id"] + ".wav"),
                text_file=os.path.join(data_path, "text", row["id"] + ".txt"),
                model=model,
                verbose=verbose,
                quiet=quiet,
                max_duration=max_duration,
                # TODO: Parse kwargs from environment variables
            )

            save_alignments(
                data_path=data_path,
                target_path=target_path,
                id=row["id"],
                alignments=alignments,
            )

    else:

        all_wav_files = glob(os.path.join(data_path, "audio/**/*.wav"), recursive=True)

        for wav_file_path in all_wav_files:
            id = wav_file_path.replace(os.path.join(data_path, "audio/"), "").replace(
                ".wav", ""
            )

            if os.path.exists(os.path.join(target_path, id)):
                if verbose and not quiet:
                    print(f"Skipping {id} as it already exists")
                continue

            if verbose and not quiet:
                print(f"Aligning {id}")

            alignments = align_method.align(
                audio_file=os.path.join(data_path, "audio", id + ".wav"),
                text_file=os.path.join(data_path, "text", id + ".txt"),
                model=model,
                verbose=verbose,
                quiet=quiet,
                max_duration=max_duration,
                # TODO: Parse kwargs from environment variables
            )

            save_alignments(
                data_path=data_path,
                target_path=target_path,
                id=id,
                alignments=alignments,
            )


def save_alignments(
    data_path: str,
    target_path: str,
    id: str,
    alignments: List[TimedSegment],
):
    """
    Saves a list of alignments to a csv file in the target location.

    Parameters
    ----------
    data_path: str
        The path to the csv file containing the audio and text files.
    target_path: str
        The path to the folder to save the aligned files to.
    id: str
        The id of the file to save.
    alignments: List[TimedSegment]
        The alignments to save.
    """
    import os
    import librosa
    import soundfile as sf
    from rifsalignment.datamodels import placeholder_text

    # Load the audio file
    audio, sr = librosa.load(
        os.path.join(data_path, "audio", f"{id}.wav"), sr=16_000, mono=True
    )

    os.makedirs(os.path.join(target_path, id), exist_ok=True)
    with open(os.path.join(target_path, id, "segments.csv"), "w+") as f:
        f.write("file,start,end,text,model_output\n")

        for segment in alignments:

            # Skip placeholder text
            if segment.text == placeholder_text:
                continue

            f.write(
                f"segment_{str(segment.start)}.wav,{segment.start},{segment.end},{segment.text},{segment.model_output}\n"  # noqa: E501
            )

            sf.write(
                os.path.join(target_path, id, f"segment_{str(segment.start)}.wav"),
                audio[int(segment.start * sr) : int(segment.end * sr)],  # noqa: E203
                sr,
                subtype="PCM_24",
            )

"""
Algorithms for alignment.
"""

from rifsalignment.base import BaseAlignment
from rifsalignment.datamodels import TimedSegment
from rifsalignment.preprocess import prepare_text

from rifsstatemachine.functions import wav_to_utterances
from rifsstatemachine.base_predictor import Predictor

from typing import List

from Levenshtein import ratio
import numpy as np
import librosa
import time


class CTC(BaseAlignment):
    """
    CTC alignment algorithm.
    """

    @staticmethod
    def align(
        audio_file: str,
        text_file: str,
        model: str,
        verbose: bool = False,
        quiet: bool = False,
    ) -> List[TimedSegment]:

        """
        Align the source and target audio files.

        Parameters
        ----------
        audio_file: str
            The path to the source audio wav file.
        text_file: str
            The path to the source text file.
        model: str
            The path to the model to use for alignment. Can be a huggingface model or a local path.
        verbose: bool
            Whether to print the alignments progress with steps.
        quiet: bool
            Whether to print anything.

        Returns
        -------
        List
            The aligned audio and text.
        """

        assert model, "Model must be specified for CTC alignment."

        print("WARNING! CTC alignment requires a very large GPU.")

        from transformers import (
            Wav2Vec2CTCTokenizer,
            Wav2Vec2Processor,
            Wav2Vec2ForCTC,
        )

        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model)
        processor = Wav2Vec2Processor.from_pretrained(model)
        model = Wav2Vec2ForCTC.from_pretrained(model)

        # Load the audio file
        audio_input, sr = librosa.load(audio_file, sr=16_000, mono=True)

        # Load and prepare text
        with open(text_file, "r") as f:
            transcripts = f.readlines()
        transcripts = prepare_text(transcripts, prepend_placeholder=True)

        alignments = CTC.ctc_align_with_transcript(
            audio=audio_input,
            transcripts=transcripts,
            tokenizer=tokenizer,
            processor=processor,
            model=model,
        )

        return alignments

    @staticmethod
    def ctc_align_with_transcript(
        audio: np.ndarray,
        transcripts: List[str],
        tokenizer,
        processor,
        model,
        sr: int = 16_000,
    ) -> List[TimedSegment]:
        """
        Align the audio with the transcript.

        Parameters
        ----------
        audio: np.ndarray
            The audio to align.
        transcripts: str
            The transcript to align.
        tokenizer: Wav2Vec2CTCTokenizer
            The tokenizer to use.
        processor: Wav2Vec2Processor
            The processor to use.
        model: Wav2Vec2ForCTC
            The model to use.
        sr: int
            The sampling rate of the audio. Defaults to 16_000.

        Returns
        -------
        List
            The aligned audio and text.
        """

        import ctc_segmentation
        import torch

        inputs = processor(audio, sampling_rate=16_000, return_tensors="pt")

        model.eval()

        with torch.no_grad():
            logits = model(inputs.input_values).logits[0]
            probs = torch.nn.functional.softmax(logits, dim=-1)

        vocab = tokenizer.get_vocab()
        inv_vocab = {v: k for k, v in vocab.items()}

        # TODO: [UNK] might not be universal for other models, but it works for now
        unk_id = vocab["[UNK]"]

        tokens = []
        for transcript in transcripts:
            assert len(transcript) > 0
            tok_ids = tokenizer(transcript.lower())["input_ids"]
            tok_ids = np.array(tok_ids, dtype=int)
            tokens.append(tok_ids[tok_ids != unk_id])

        char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
        config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
        config.index_duration = inputs.input_values.shape[1] / probs.size()[0] / sr

        ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_token_list(
            config, tokens
        )
        timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(
            config, probs.numpy(), ground_truth_mat
        )
        segments = ctc_segmentation.determine_utterance_segments(
            config, utt_begin_indices, char_probs, timings, transcripts
        )

        timed_segments = [
            TimedSegment(text=t, start=p[0], end=p[1])
            for t, p in zip(transcripts, segments)
        ]

        return timed_segments


class StateMachineForLevenshtein(BaseAlignment):
    """
    State machine alignment algorithm with Levenshtein.
    """

    @staticmethod
    def align(
        audio_file: str,
        text_file: str,
        model: str,
        verbose: bool = False,
        quiet: bool = False,
        **kwargs,
    ) -> List[TimedSegment]:
        """
        Align the source and target audio files.

        Parameters
        ----------
        audio_file: str
            The path to the source audio wav file.
        text_file: str
            The path to the source text file.
        model: str
            The path to the model to use for alignment. Can be a huggingface model or a local path.
        verbose: bool
            Whether to print verbose output. Defaults to False.
        quiet: bool
            Whether to print any output. Defaults to False.
        max_depth: int
            The maximum depth of the permutations. Defaults to 10.
        """

        if verbose and not quiet:
            print(
                "This model does not take a model parameter. Defaulting to 'Alvenir/wav2vec2-base-da-ft-nst'."
            )
        # TODO: Parse model path to this predictor.
        predictor = Predictor()

        # Load the audio file
        audio_input, sr = librosa.load(audio_file, sr=16_000, mono=True)
        assert sr == 16_000

        # Load and prepare text
        with open(text_file, "r") as f:
            transcripts = f.readlines()
        transcripts = prepare_text(transcripts, prepend_placeholder=False)

        start = time.time()
        all_predictions = wav_to_utterances(audio_input, model=predictor)
        all_predictions_list = [pred for pred in all_predictions]
        all_predictions_text = [pred.transcription for pred in all_predictions_list]
        end = time.time()

        if verbose and not quiet:
            print(f"Finished predicting with state machine. Total time: {end - start}")

        # Generate all possible permutations of the predictions
        start = time.time()
        max_depth = kwargs.get("max_depth", 10)
        all_permutations = []
        for i in range(len(all_predictions_text)):
            for j in range(i, len(all_predictions_text)):
                all_permutations.append(
                    TimedSegment(
                        start=all_predictions_list[i].start / sr,
                        end=all_predictions_list[j - 1].end / sr,
                        text=" ".join(all_predictions_text[i:j]),
                    )
                )
                if j == i + max_depth - 1:
                    break
        end = time.time()
        if verbose and not quiet:
            print(f"Finished generating all permutations. Total time: {end - start}")

        # Align the audio with the transcript
        start = time.time()
        alignments = []
        for i, true_transcript in enumerate(transcripts):
            all_sims = []
            for pred in all_permutations:
                sim = ratio(pred.text.upper(), true_transcript.upper())
                all_sims.append(sim)
            best_alignment = all_permutations[np.argmax(all_sims)]

            if verbose and not quiet:
                print(
                    f"Best alignment for {true_transcript} is {best_alignment.text} with score {np.max(all_sims)}"
                )
                print(
                    f"True start: {best_alignment.start}, true end: {best_alignment.end}"
                )
                print()

            alignments.append(
                TimedSegment(
                    start=best_alignment.start,
                    end=best_alignment.end,
                    text=true_transcript,
                )
            )
        end = time.time()
        if verbose and not quiet:
            print(f"Finished aligning with Levenshtein. Total time: {end - start}")

        return alignments


# TODO: Add a class for the state machine unsupervised (without text).
class StateMachineUnsupervised(BaseAlignment):
    """
    Unsupervised alignment with state machine
    """

    @staticmethod
    def align(
        audio_file: str,
        text_file: str,
        model: str,
        verbose: bool = False,
        quiet: bool = False,
    ) -> List[TimedSegment]:
        """
        Align a single audio file with no text.

        Parameters
        ----------
        audio_file: str
            The path to the source audio wav file.
        text_file: str
            The path to the source text file.
        model: str
            The path to the model to use for alignment. Can be a huggingface model or a local path.
        verbose: bool
            Whether to print verbose output. Defaults to False.
        quiet: bool
            Whether to print any output. Defaults to False.
        """

        if verbose and not quiet:
            print(
                "This algorithm does not take a model parameter. Defaulting to 'Alvenir/wav2vec2-base-da-ft-nst'."
            )
        # TODO: Parse model path to this predictor.
        predictor = Predictor()

        # Load the audio file
        audio_input, sr = librosa.load(audio_file, sr=16_000, mono=True)
        assert sr == 16_000

        all_predictions = wav_to_utterances(audio_input, model=predictor)
        all_predictions_list = [pred for pred in all_predictions]

        alignments = []
        for pred in all_predictions_list:
            alignments.append(
                TimedSegment(
                    start=pred.start / sr,
                    end=pred.end / sr,
                    text=pred.transcription,
                )
            )

        return alignments


class StateMachineUnsupervisedNoModel(BaseAlignment):
    """
    Unsupervised alignment with state machine
    """

    @staticmethod
    def align(
        audio_file: str,
        text_file: str,
        model: str,
        verbose: bool = False,
        quiet: bool = False,
    ) -> List[TimedSegment]:
        """
        Align a single audio file with no text.

        Parameters
        ----------
        audio_file: str
            The path to the source audio wav file.
        text_file: str
            The path to the source text file.
        model: str
            The path to the model to use for alignment. Can be a huggingface model or a local path.
        verbose: bool
            Whether to print verbose output. Defaults to False.
        quiet: bool
            Whether to print any output. Defaults to False.
        """

        if verbose and not quiet:
            print(
                "This algorithm does not take a model parameter."
            )

        # Load the audio file
        audio_input, sr = librosa.load(audio_file, sr=16_000, mono=True)
        assert sr == 16_000

        all_predictions = wav_to_utterances(audio_input)
        all_predictions_list = [pred for pred in all_predictions]

        alignments = []
        for pred in all_predictions_list:
            alignments.append(
                TimedSegment(
                    start=pred.start / sr,
                    end=pred.end / sr,
                    text="",
                )
            )

        return alignments
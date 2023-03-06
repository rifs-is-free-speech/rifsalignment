"""
Algorithms for alignment.
"""

from rifsalignment.base import BaseAlignment
from rifsalignment.datamodels import TimedSegment
from rifsalignment.preprocess import prepare_text

from typing import List
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)
import ctc_segmentation
import numpy as np
import librosa
import torch


class CTC(BaseAlignment):
    """
    CTC alignment algorithm.
    """

    @staticmethod
    def align(
        audio_file: str,
        text_file: str,
        model: str,
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

        Returns
        -------
        List
            The aligned audio and text.
        """

        print("WARNING! CTC alignment requires a very large GPU.")

        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model)
        processor = Wav2Vec2Processor.from_pretrained(model)
        model = Wav2Vec2ForCTC.from_pretrained(model)

        # Load the audio file
        audio_input, sr = librosa.load(audio_file, sr=16_000, mono=True)

        # Load and prepare text
        with open(text_file, "r") as f:
            transcripts = f.readlines()
        transcripts = prepare_text(transcripts, prepend_placeholder=True)

        alignments = CTC._ctc_align_with_transcript(
            audio=audio_input,
            transcripts=transcripts,
            tokenizer=tokenizer,
            processor=processor,
            model=model,
        )

        return alignments

    @staticmethod
    def _ctc_align_with_transcript(
        audio: np.ndarray,
        transcripts: List[str],
        tokenizer: Wav2Vec2CTCTokenizer,
        processor: Wav2Vec2Processor,
        model: Wav2Vec2ForCTC,
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

        # timed_segments = [{"text": t, "start": p[0], "end": p[1], "conf": p[2]} for t, p in zip(transcripts, segments)]

        return timed_segments

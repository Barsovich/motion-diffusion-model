from torch.utils.data import Dataset
import json
from os import path, listdir
import numpy as np
import bisect
import spacy
import torch
import math


class TrinityDataset(Dataset):
    def __init__(self,
                 split,
                 num_frames,
                 max_text_length=50,
                 clip_length=5,
                 frame_rate=30,
                 motion_dir='./dataset/genea_2022/trn/npz/',
                 transcripts_dir='./dataset/genea_2022/trn/json/',
                 audio_dir='dataset/genea_2022/trn/audio/'):

        super().__init__()

        self.split = split
        self.num_frames = num_frames
        self.motion_dir = motion_dir
        self.transcripts_dir = transcripts_dir
        self.clip_length = clip_length
        self.frame_rate = frame_rate
        self.max_text_length = max_text_length
        self.num_frames_per_clip = clip_length * frame_rate
        self.nlp = spacy.load("en_core_web_sm")

        with open(path.join(transcripts_dir, "processed_words_into_sections_5s_offset0s.json")) as f:
            self.transcripts_0_offset = json.load(f)
        with open(path.join(transcripts_dir, "processed_words_into_sections_5s_offset25s.json")) as f:
            self.transcripts_2_5_offset = json.load(f)

        motion_files = [path.join(motion_dir, f) for f in listdir(
            motion_dir) if path.isfile(path.join(motion_dir, f))]
        motion_files.sort()
        self.num_motion_files = len(motion_files)
        self.motion_start_indices = []
        self.motions = []
        current_motion_start_index = 0
        for i in range(self.num_motion_files):
            motion = np.load(motion_files[i])
            if motion_files[i][-3:] == "npz":
                filename = next(iter(motion.keys()))
                motion = motion[filename]
            num_frames = motion.shape[0]
            num_frames = num_frames - (num_frames % (self.num_frames_per_clip))
            motion = motion[:num_frames]

            self.motions.append(motion)
            self.motion_start_indices.append(current_motion_start_index)
            current_motion_start_index += num_frames // self.num_frames_per_clip

        audio_files = [path.join(audio_dir, f) for f in listdir(
            audio_dir) if path.isfile(path.join(audio_dir, f))]
        audio_files.sort()
        self.audios = []
        for i in range(self.num_motion_files):
            audio = np.load(audio_files[i])
            num_frames = audio.shape[0]
            num_frames = num_frames - (num_frames % (self.num_frames_per_clip))
            audio = audio[:num_frames]
            self.audios.append(audio)

        motion_stacked = np.concatenate(self.motions, axis=0)
        self.mean = np.mean(motion_stacked, axis=0)[None, :]
        self.std = np.std(motion_stacked, axis=0)[None, :]
        self.zero_std = np.squeeze(self.std) < 1e-10
        self.num_clips = current_motion_start_index

    def __len__(self):
        # The first self.num_clips many items are the clips at offset 0
        # The next self.num_clips many items are the clips at offset (self.num_frames_per_clip // 2)
        # Instead of the clip of each motion which is actually half size, we give the last full motion
        return self.num_clips * 2

    def __getitem__(self, index):
        motion_offset = 0
        if index >= self.num_clips:
            # We serve one of the clips with offset
            index -= self.num_clips
            motion_offset = self.num_frames_per_clip // 2

        motion_index = bisect.bisect_right(
            self.motion_start_indices, index) - 1
        index_within_motion = index
        if motion_index > 0:
            index_within_motion -= self.motion_start_indices[motion_index]

        if motion_offset > 0 and index_within_motion == len(self.motions[motion_index]) // self.num_frames_per_clip - 1:
            # This is the last motion of the sequence with offset. We serve the last complete motion
            motion_offset = 0

        motion_start_index = index_within_motion * \
            self.num_frames_per_clip + motion_offset
        motion = self.motions[motion_index][motion_start_index:
                                            motion_start_index + self.num_frames_per_clip]
        audio = self.audios[motion_index][motion_start_index:
                                          motion_start_index + self.num_frames_per_clip]
        motion = motion - self.mean
        motion_norm = motion / self.std
        motion_norm[:, self.zero_std] = motion[:, self.zero_std]

        transcript = self.transcripts_0_offset[motion_index] if index < self.num_clips else self.transcripts_2_5_offset[motion_index]
        text = transcript[index_within_motion]["words"] if index_within_motion < len(
            transcript) else []

        # There are some words that are registered as NaN for some reason
        text = list(filter(lambda word: isinstance(word, str), text))

        parsed_text = self.nlp(" ".join(text))

        tokens = []
        for token in parsed_text:
            word = token.text
            split_by_dot = word.split('.')
            word = split_by_dot[0]
            if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
                word = token.lemma_
            tokens.append(word + "/" + token.pos_)
            if len(split_by_dot) == 2:
                # There was a dot at the end of the word
                tokens.append("eos/OTHER")
        tokens += ["unk/OTHER"] * (self.max_text_length - len(tokens))

        return {
            "text": text,
            "tokens": tokens,
            "motion": motion_norm,
            "audio": audio
        }


def trinity_collate(batch):
    sample_motion = batch[0]["motion"]
    sample_audio = batch[0]["audio"]
    clip_frame_count = sample_motion.shape[0]
    motion_feature_count = sample_motion.shape[1]
    audio_feature_count = sample_audio.shape[1]
    batch_size = len(batch)

    motion = torch.empty(
        (batch_size, motion_feature_count, 1, clip_frame_count))
    for i in range(batch_size):
        motion[i, :, 0, :] = torch.tensor(batch[i]["motion"].T)

    audio = torch.empty(
        (batch_size, clip_frame_count, audio_feature_count))
    for i in range(batch_size):
        audio[i] = torch.tensor(batch[i]["audio"])

    lengths = torch.tensor([clip_frame_count] * batch_size)
    mask = torch.ones((batch_size, 1, 1, clip_frame_count))

    text = []
    tokens = []
    for i in range(batch_size):
        text.append(" ".join(batch[i]["text"]))
        tokens.append("_".join(batch[i]["tokens"]))

    cond = {'y': {
        "lengths": lengths,
        "mask": mask,
        "text": text,
        "tokens": tokens,
        "audio": audio,
    }}

    return motion, cond

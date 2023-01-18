from torch.utils.data import Dataset
import json
from os import path, listdir
import numpy as np
import bisect
import spacy
import torch


class TrinityDataset(Dataset):
    def __init__(self,
                 split,
                 num_frames,
                 max_text_length=50,
                 clip_length=5,
                 frame_rate=20,
                 motion_dir='./dataset/Trinity/motion/',
                 transcripts_dir='dataset/Trinity/TranscriptsProcessed/'):

        super().__init__()

        # import pdb
        # pdb.set_trace()
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
            num_frames = motion.shape[0]
            num_frames = num_frames - (num_frames % (self.num_frames_per_clip))
            motion = motion[:num_frames]

            self.motions.append(motion)
            self.motion_start_indices.append(current_motion_start_index)
            current_motion_start_index += num_frames // self.num_frames_per_clip

        self.num_clips = current_motion_start_index

    def __len__(self):
        return self.num_clips

    def __getitem__(self, index):
        # import pdb
        # pdb.set_trace()
        motion_index = bisect.bisect_left(self.motion_start_indices, index) - 1
        index_within_motion = index
        if motion_index > 0:
            index_within_motion -= self.motion_start_indices[motion_index]

        motion = self.motions[motion_index][index_within_motion *
                                            self.num_frames_per_clip:(index_within_motion + 1) * self.num_frames_per_clip]
        text = self.transcripts_0_offset[motion_index][index_within_motion]["words"]
        parsed_text = self.nlp(" ".join(text))
        tokens = [str(i) + "/" + i.pos_ for i in parsed_text]
        tokens += ["unk/OTHER"] * (self.max_text_length - len(tokens))

        return {
            "text": text,
            "tokens": tokens,
            "motion": motion,
        }


def trinity_collate(batch):
    import pdb
    pdb.set_trace()
    sample_motion = batch[0]["motion"]
    clip_frame_count = sample_motion.shape[0]
    motion_feature_count = sample_motion.shape[1]
    batch_size = len(batch)

    motion = torch.empty(
        (batch_size, motion_feature_count, 1, clip_frame_count))
    for i in range(batch_size):
        motion[i, :, 0, :] = torch.tensor(batch[i]["motion"].T)

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
        "tokens": tokens
    }}

    return motion, cond

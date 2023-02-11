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
                 data_dir = './dataset/genea_2022/'):

        super().__init__()

        self.split = split
        self.num_frames = num_frames
        self.motion_dir = path.join(data_dir, split, 'npz')
        self.audio_dir = path.join(data_dir, split, 'audio')
        self.transcripts_dir = path.join(data_dir, split, 'json')

        self.clip_length = clip_length
        self.frame_rate = frame_rate
        self.max_text_length = max_text_length
        self.num_frames_per_clip = clip_length * frame_rate

        self.num_offsets = 10
        self.transcripts = []
        for i in range(self.num_offsets):
            with open(path.join(self.transcripts_dir, f"text_5s_offset_{i}_half_s.json")) as f:
                self.transcripts.append(json.load(f))

        motion_files = [path.join(self.motion_dir, f) for f in listdir(
            self.motion_dir) if path.isfile(path.join(self.motion_dir, f))]
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

        audio_files = [path.join(self.audio_dir, f) for f in listdir(
            self.audio_dir) if path.isfile(path.join(self.audio_dir, f))]
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

        audio_stacked = np.concatenate(self.audios, axis=0)
        self.audio_mean = np.mean(audio_stacked, axis=0)[None, :]
        self.audio_std = np.std(audio_stacked, axis=0)[None, :]
        self.audio_zero_std = np.squeeze(self.audio_std) < 1e-10


    def __len__(self):
        # The first self.num_clips many items are the clips at offset 0
        # The next self.num_clips many items are the clips at offset (self.num_frames_per_clip // 2)
        # Instead of the clip of each motion which is actually half size, we give the last full motion
        return self.num_clips * self.num_offsets

    def __getitem__(self, index):
        motion_offset = 0
        # We serve one of the clips with offset
        offset = index // self.num_clips
        index_within_motion = index % self.num_clips
        motion_offset = self.num_frames_per_clip * (offset // self.num_offsets)              

        motion_index = bisect.bisect_right(self.motion_start_indices, index_within_motion) - 1
        if motion_index > 0:
            index_within_motion -= self.motion_start_indices[motion_index]

        if motion_offset > 0 and index_within_motion == len(self.motions[motion_index]) // self.num_frames_per_clip - 1:
            # This is the last motion of the sequence with offset. We serve the last complete motion
            motion_offset = 0
            offset = 0

        motion_start_index = index_within_motion * \
            self.num_frames_per_clip + motion_offset
        motion = self.motions[motion_index][motion_start_index:
                                            motion_start_index + self.num_frames_per_clip]
        audio = self.audios[motion_index][motion_start_index:
                                          motion_start_index + self.num_frames_per_clip]
        motion = motion - self.mean
        motion_norm = motion / self.std
        motion_norm[:, self.zero_std] = motion[:, self.zero_std]

        audio = audio - self.audio_mean
        audio_norm = audio / self.audio_std
        audio_norm[:, self.audio_zero_std] = audio[:, self.audio_zero_std]

        transcript = self.transcripts[offset][motion_index]
        text = transcript[index_within_motion]["section_text"] if index_within_motion < len(
            transcript) else ""
        text_indices = transcript[index_within_motion]["indices"] if index_within_motion < len(
            transcript) else []
        
        return {
            "text": text,
            "text_indices": text_indices,
            "motion": motion_norm,
            "audio": audio_norm
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
    text_indices = []
    for i in range(batch_size):
        text.append(batch[i]["text"])
        text_indices.append(batch[i]["text_indices"])

    cond = {'y': {
        "lengths": lengths,
        "mask": mask,
        "text": text,
        "audio": audio,
        "text_indices": text_indices,
    }}

    return motion, cond

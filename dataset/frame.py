#!/usr/bin/env python3

"""
File containing classes related to the frame datasets.
"""

#Standard imports
from util.io import load_json, load_text
import os
import random
import numpy as np
import copy
import torch
from torch.utils.data import Dataset
import torchvision
from tqdm import tqdm
import pickle
import math
from tqdm import tqdm

#Local imports


#Constants

# Pad the start/end of videos with empty frames
DEFAULT_PAD_LEN = 5
FPS_SN = 25


class ActionSpotDataset(Dataset):

    def __init__(
            self,
            classes,                    # dict of class names to idx
            label_file,                 # path to label json
            frame_dir,                  # path to frames
            store_dir,                  # path to store files (with frames path and labels per clip)
            store_mode,                 # 'store' or 'load'
            modality,                   # [rgb]
            clip_len,                   # Number of frames per clip
            dataset_len,                # Number of clips
            stride=1,                   # Downsample frame rate
            overlap=1,                  # Overlap between clips (in proportion to clip_len)
            radi_displacement=0,        # Radius of displacement for labels
            mixup=False,                # Mixup usage
            pad_len=DEFAULT_PAD_LEN,    # Number of frames to pad the start
                                        # and end of videos
            dataset = 'soccernetball',     # Dataset name
            event_team = False           # Include team in event label
    ):
        self._src_file = label_file
        self._labels = load_json(label_file)
        self._split = label_file.split('/')[-1].split('.')[0]
        self._class_dict = classes
        self._video_idxs = {x['video']: i for i, x in enumerate(self._labels)}
        self._dataset = dataset
        self._store_dir = store_dir
        self._store_mode = store_mode
        assert store_mode in ['store', 'load']
        self._clip_len = clip_len
        assert clip_len > 0
        self._stride = stride
        assert stride > 0
        if overlap != 1:
            self._overlap = int((1-overlap) * clip_len)
        else:
            self._overlap = 1
        assert overlap >= 0 and overlap <= 1
        if self._dataset == 'soccernet':
            if self._overlap % 2 == 1:
                self._overlap += 1 # To ensure that the overlap is even to ensure frames exist (extracted with stride 2)
        self._dataset_len = dataset_len
        assert dataset_len > 0
        self._pad_len = pad_len
        assert pad_len >= 0

        # Label modifications
        self._radi_displacement = radi_displacement
        assert radi_displacement >= 0
        self._event_team = event_team

        #Mixup
        self._mixup = mixup        

        #Frame reader class
        self._frame_reader = FrameReader(frame_dir, modality, dataset = dataset)

        #Variables for SN & SNB label paths if datastes
        if (self._dataset == 'soccernet') | (self._dataset == 'soccernetball'):
            global LABELS_SN_PATH
            global LABELS_SNB_PATH
            LABELS_SN_PATH = load_text(os.path.join('data', 'soccernet', 'labels_path.txt'))[0]
            LABELS_SNB_PATH = load_text(os.path.join('data', 'soccernetball', 'labels_path.txt'))[0]

        #Store or load clips
        if self._store_mode == 'store':
            self._store_clips()
        elif self._store_mode == 'load':
            self._load_clips()

        self._total_len = len(self._frame_paths)

    def _store_clips(self):
        #Initialize frame paths list
        self._frame_paths = []
        self._labels_store = []
        if self._radi_displacement > 0:
            self._labelsD_store = []
        for video in tqdm(self._labels):
            video_len = int(video['num_frames'])

            #Different label file for SoccerNet (and we require the half for frames):
            if self._dataset == 'soccernet':
                video_half = int(video['video'][-1])
                labels_file = load_json(os.path.join(LABELS_SN_PATH, "/".join(video['video'].split('/')[:-1]) + '/Labels-v2.json'))['annotations']
            elif self._dataset == 'soccernetball':
                video_half = 1
                labels_file = load_json(os.path.join(LABELS_SNB_PATH, video['video'] + '/Labels-ball.json'))['annotations']

            for base_idx in range(-self._pad_len * self._stride, max(0, video_len - 1 + (2 * self._pad_len - self._clip_len) * self._stride), self._overlap):

                frames_paths = self._frame_reader.load_paths(video['video'], base_idx, base_idx + self._clip_len * self._stride, stride=self._stride)

                labels = []
                if self._radi_displacement > 0:
                    labelsD = []
                
                for event in labels_file:    
                    #For SoccerNet dataset different label file
                    event_half = int(event['gameTime'][0])
                    if event_half == video_half:
                        event_frame = int(int(event['position']) / 1000 * FPS_SN) #miliseconds to frames
                        label_idx = (event_frame - base_idx) // self._stride

                        if self._radi_displacement >= 0:
                            if (label_idx >= -self._radi_displacement and label_idx < self._clip_len + self._radi_displacement):
                                label = event['label']
                                if self._event_team:
                                    if (self._dataset == 'soccernet') & (event['team'] == 'not applicable'):
                                        label = label + '-' + 'left' # if not applicable left by default (to suppress from team loss later)
                                    else:
                                        label = label + '-' + event['team']
                                label = self._class_dict[label]
                                for i in range(max(0, label_idx - self._radi_displacement), min(self._clip_len, label_idx + self._radi_displacement + 1)):
                                    labels.append({'label': label, 'label_idx': i})
                                    if self._radi_displacement > 0:
                                        labelsD.append({'displ': i - label_idx, 'label_idx': i})

                if frames_paths[1] != -1: #in case no frames were available

                    #For SoccerNet only include clips with events
                    if self._dataset == 'soccernet':
                        if len(labels) > 0:
                            self._frame_paths.append(frames_paths)
                            self._labels_store.append(labels)
                            if self._radi_displacement > 0:
                                self._labelsD_store.append(labelsD)
                    else:
                        self._frame_paths.append(frames_paths)
                        self._labels_store.append(labels)
                        if self._radi_displacement > 0:
                            self._labelsD_store.append(labelsD)

        #Save to store
        store_path = os.path.join(self._store_dir, 'LEN' + str(self._clip_len) + 'DIS' + str(self._radi_displacement) + 'SPLIT' + self._split)

        if not os.path.exists(store_path):
            os.makedirs(store_path)

        with open(store_path + '/frame_paths.pkl', 'wb') as f:
            pickle.dump(self._frame_paths, f)
        with open(store_path + '/labels.pkl', 'wb') as f:
            pickle.dump(self._labels_store, f)
        if self._radi_displacement > 0:
            with open(store_path + '/labelsD.pkl', 'wb') as f:
                pickle.dump(self._labelsD_store, f)
        print('Stored clips to ' + store_path)
        return
    
    def _load_clips(self):
        store_path = os.path.join(self._store_dir, 'LEN' + str(self._clip_len) + 'DIS' + str(self._radi_displacement) + 'SPLIT' + self._split)
        
        with open(store_path + '/frame_paths.pkl', 'rb') as f:
            self._frame_paths = pickle.load(f)
        with open(store_path + '/labels.pkl', 'rb') as f:
            self._labels_store = pickle.load(f)
        if self._radi_displacement > 0:
            with open(store_path + '/labelsD.pkl', 'rb') as f:
                self._labelsD_store = pickle.load(f)
        print('Loaded clips from ' + store_path)
        return

    def _get_one(self):
        #Get random index
        idx = random.randint(0, self._total_len - 1)

        #Get frame_path and labels dict
        frames_path = self._frame_paths[idx]
        dict_label = self._labels_store[idx]
        if self._radi_displacement > 0:
            dict_labelD = self._labelsD_store[idx]

        #Load frames
        frames = self._frame_reader.load_frames(frames_path, pad=True, stride=self._stride)

        #Process labels
        labels = np.zeros(self._clip_len, np.int64)
        if self._event_team:
            labels_team = np.zeros(self._clip_len, np.int64) - 1
        for label in dict_label:
            if not self._event_team:
                labels[label['label_idx']] = label['label']
            else:
                labels[label['label_idx']] = math.ceil(label['label'] / 2)
                if label['label'] != 0:
                    if (self._dataset == 'soccernet') & (math.ceil(label['label'] / 2) == 9):
                        labels_team[label['label_idx']] = -1 # -1 as it is not applicable
                    else:
                        labels_team[label['label_idx']] = (label['label']+1) % 2 # -1 if background, 0 if left, 1 if right

        data = {'frame': frames, 'contains_event': int(np.sum(labels) > 0), 'label': labels}

        if self._radi_displacement > 0:
            labelsD = np.zeros(self._clip_len, np.int64)
            for label in dict_labelD:
                labelsD[label['label_idx']] = label['displ']

            data['labelD'] = labelsD
        
        if self._event_team:
            data['labelT'] = labels_team

        return data

    def __getitem__(self, unused):
        ret = self._get_one()
        
        if self._mixup:
            mix = self._get_one()    # Sample another clip
            
            ret['frame2'] = mix['frame']
            ret['contains_event2'] = mix['contains_event']
            ret['label2'] = mix['label']
            if self._radi_displacement > 0:
                ret['labelD2'] = mix['labelD']
            if self._event_team:
                ret['labelT2'] = mix['labelT']

        return ret

    def __len__(self):
        return self._dataset_len

    def print_info(self):
        _print_info_helper(self._src_file, self._labels)

    def get_paths_labels_dict(self, dataset: str):
        """
        Creates a dictionary mapping frame paths to their corresponding labels
        
        Returns:
            Dict: {frame_path: label}
        """
        framepath2label = {}
    
        # Process each clip
        for idx in tqdm(range(self._total_len)):
            frames_path = self._frame_paths[idx]
            dict_label = self._labels_store[idx]
            
            # Extract path information
            base_path = frames_path[0]
            start_frame = frames_path[1]
            pad_start = frames_path[2]
            pad_end = frames_path[3]
            length = frames_path[5]
            
            # Skip if no valid start frame was found
            if start_frame == -1:
                continue
            
            # Calculate actual frame indices, skipping padding
            actual_length = length - pad_start - pad_end
            
            # Create frame paths
            frame_paths = []
            if self._dataset == 'soccernet' or self._dataset == 'soccernetball':
                for i in range(actual_length):
                    frame_num = start_frame + i * self._stride
                    frame_path = os.path.join(base_path, f"frame{frame_num}.jpg")
                    # Only add if file exists
                    if os.path.exists(frame_path):
                        frame_paths.append((frame_path, pad_start + i))
                        
            # Map labels to frame paths
            for frame_path, frame_idx in frame_paths:
                # Default to background (0)
                label = 0
                
                # Check if this frame has any labels
                for label_item in dict_label:
                    if label_item['label_idx'] == frame_idx:
                        label = label_item['label']
                        break
                        
                # Add to dictionary
                framepath2label[frame_path] = label
        
        # Print statistics
        print(f"Created mapping with {len(framepath2label)} frames")
        
        # Save dictionary
        save_path = os.path.join(self._store_dir, f'framepath2label_{dataset}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(framepath2label, f)
        print(f"Saved mapping to {save_path}")
        
        return framepath2label

class FrameReader:

    def __init__(self, frame_dir, modality, dataset):
        self._frame_dir = frame_dir
        self.modality = modality
        self.dataset = dataset

    def read_frame(self, frame_path):
        img = torchvision.io.read_image(frame_path) #.float() / 255 -> into model normalization / augmentations
        return img
    
    def load_paths(self, video_name, start, end, stride=1, source_info = None):

        if (self.dataset == 'soccernetball') | (self.dataset == 'soccernet'):
            path = os.path.join(self._frame_dir, video_name)

        found_start = -1
        pad_start = 0
        pad_end = 0
        for frame_num in range(start, end, stride):

            if frame_num < 0:
                pad_start += 1
                continue

            if pad_end > 0:
                pad_end += 1
                continue
            
            if (self.dataset == 'soccernet') | (self.dataset == 'soccernetball'):
                frame = frame_num
                frame_path = os.path.join(path, 'frame' + str(frame) + '.jpg')
                base_path = path
                ndigits = -1
            
            exist_frame = os.path.exists(frame_path)
            if exist_frame & (found_start == -1):
                found_start = frame

            if not exist_frame:
                pad_end += 1

        ret = [base_path, found_start, pad_start, pad_end, ndigits, (end-start) // stride]

        return ret
    
    def load_frames(self, paths, pad=False, stride=1):
        base_path = paths[0]
        start = paths[1]
        pad_start = paths[2]
        pad_end = paths[3]
        ndigits = paths[4]
        length = paths[5]

        ret = []
        if ndigits == -1:
            path = os.path.join(base_path, 'frame')
            _ = [ret.append(self.read_frame(path + str(start + j * stride) + '.jpg')) for j in range(length - pad_start - pad_end)]

        ret = torch.stack(ret, dim=int(len(ret[0].shape) == 4))

        # Always pad start, but only pad end if requested
        if pad_start > 0 or (pad and pad_end > 0):
            ret = torch.nn.functional.pad(
                ret, (0, 0, 0, 0, 0, 0, pad_start, pad_end if pad else 0))

        return ret
    
    def get_frame_paths(self, paths, pad=False, stride=1):
        base_path = paths[0]
        start = paths[1]
        pad_start = paths[2]
        pad_end = paths[3]
        ndigits = paths[4]
        length = paths[5]
        frame_paths = []

        if ndigits == -1:
            path = os.path.join(base_path, 'frame')

        for i in range(length - pad_start - pad_end):
            frame_paths.append(path + str(start + i * stride) + '.jpg')
        
        return frame_paths   

class ActionSpotVideoDataset(Dataset):

    def __init__(
            self,
            classes,
            label_file,
            frame_dir,
            modality,
            clip_len,
            overlap_len=0,
            stride=1,
            pad_len=DEFAULT_PAD_LEN,
            dataset = 'soccernetball',
            event_team = False
    ):
        self._src_file = label_file
        self._labels = load_json(label_file)
        self._class_dict = classes
        self._video_idxs = {x['video']: i for i, x in enumerate(self._labels)}
        self._clip_len = clip_len
        self._stride = stride
        self._dataset = dataset
        self._event_team = event_team

        self._frame_reader = FrameReaderVideo(frame_dir, modality, dataset = dataset)

        self._clips = []
        for l in self._labels:
            has_clip = False
            for i in range(
                -pad_len * self._stride,
                max(0, l['num_frames'] - (overlap_len * stride)), \
                # Need to ensure that all clips have at least one frame
                (clip_len - overlap_len) * self._stride
            ):
                has_clip = True
                
                self._clips.append((l['video'], i))
            assert has_clip, l

        #Variables for SN & SNB label paths if datastes
        if (self._dataset == 'soccernet') | (self._dataset == 'soccernetball'):
            global LABELS_SN_PATH
            global LABELS_SNB_PATH
            LABELS_SN_PATH = load_text(os.path.join('data', 'soccernet', 'labels_path.txt'))[0]
            LABELS_SNB_PATH = load_text(os.path.join('data', 'soccernetball', 'labels_path.txt'))[0]

    def __len__(self):
        return len(self._clips)

    def __getitem__(self, idx):
        
        video_name, start = self._clips[idx]

        frames = self._frame_reader.load_frames(
            video_name, start, start + self._clip_len * self._stride, pad=True,
            stride=self._stride)

        return {'video': video_name, 'start': start // self._stride,
                'frame': frames}

    def get_labels(self, video):
        meta = self._labels[self._video_idxs[video]]
        if self._dataset == 'soccernet':
            labels_file = load_json(os.path.join(LABELS_SN_PATH, "/".join(meta['video'].split('/')[:-1]) + '/Labels-v2.json'))['annotations']
            labels_half = int(video[-1])
        elif self._dataset == 'soccernetball':
            labels_file = load_json(os.path.join(LABELS_SNB_PATH, meta['video'] + '/Labels-ball.json'))['annotations']
            labels_half = 1
        
        num_frames = meta['num_frames']
        num_labels = math.ceil(num_frames / self._stride) 

        labels = np.zeros(num_labels, np.int64)
        for event in labels_file:
            if (self._dataset == 'soccernet') | (self._dataset == 'soccernetball'):
                frame = int(int(event['position']) / 1000 * FPS_SN)
                half = int(event['gameTime'][0])

            if (half == labels_half):
                if (frame < num_frames):
                    label = event['label']
                    if self._event_team:
                        label = label + '-' + event['team']
                    labels[frame // self._stride] = self._class_dict[label]
                else:
                    print('Warning: {} >= {} is past the end {}'.format(
                        frame, num_frames, meta['video']))
        return labels

    @property
    def videos(self):
        if (self._dataset == 'soccernet') | (self._dataset == 'soccernetball'):
            return sorted([
                (v['video'], math.ceil(v['num_frames'] / self._stride),
                FPS_SN / self._stride) for v in self._labels])

    @property
    def labels(self):
        assert self._stride > 0
        if self._stride == 1:
            return self._labels
        else:
            labels = []
            for x in self._labels:
                x_copy = copy.deepcopy(x)
                
                if (self._dataset == 'soccernet') | (self._dataset == 'soccernetball'):
                    x_copy['fps'] = FPS_SN / self._stride

                x_copy['num_frames'] //= self._stride

                if self._dataset == 'soccernet':
                    labels_file = load_json(os.path.join(LABELS_SN_PATH, "/".join(x_copy['video'].split('/')[:-1]) + '/Labels-v2.json'))['annotations']
                    for e in labels_file:
                        half = int(e['gameTime'][0])
                        if half == int(x_copy['video'][-1]):
                            e['frame'] = int(int(e['position']) / 1000 * FPS_SN) // self._stride
                    x_copy['events'] = labels_file

                elif self._dataset == 'soccernetball':
                    labels_file = load_json(os.path.join(LABELS_SNB_PATH, x_copy['video'] + '/Labels-ball.json'))['annotations']
                    for e in labels_file:
                        e['frame'] = int(int(e['position']) / 1000 * FPS_SN) // self._stride
                    x_copy['events'] = labels_file

                labels.append(x_copy)
            return labels

    def print_info(self):
        num_frames = sum([x['num_frames'] for x in self._labels])

        print('{} : {} videos, {} frames ({} stride)'.format(
            self._src_file, len(self._labels), num_frames, self._stride)
        )

class FrameReaderVideo:

    def __init__(self, frame_dir, modality, dataset):
        self._frame_dir = frame_dir
        self._modality = modality
        assert self._modality == 'rgb'
        self._dataset = dataset

    def read_frame(self, frame_path):
        img = torchvision.io.read_image(frame_path) #/ 255 -> modified for ActionSpotVideoDataset (to be compatible with train reading without / 255)
        return img

    def load_frames(self, video_name, start, end, pad=False, stride=1, source_info = None):
        ret = []
        n_pad_start = 0
        n_pad_end = 0

        for frame_num in range(start, end, stride):

            if frame_num < 0:
                n_pad_start += 1
                continue

            if self._dataset == 'soccernet':
                frame_path = os.path.join(
                    self._frame_dir, video_name, 'frame' + str(frame_num) + '.jpg'
                    #self._frame_dir, video_name, str(frame_num) + '.jpg'
                )

            elif self._dataset == 'soccernetball':
                frame_path = os.path.join(self._frame_dir, video_name, 'frame' + str(frame_num) + '.jpg')
                
            try:
                img = self.read_frame(frame_path)
                ret.append(img)
            except RuntimeError:
                # print('Missing file!', frame_path)
                n_pad_end += 1

        if len(ret) == 0:
            return -1 # Return -1 if no frames were loaded

        # In the multicrop case, the shape is (B, T, C, H, W)
        ret = torch.stack(ret, dim=int(len(ret[0].shape) == 4))

        # Always pad start, but only pad end if requested
        if n_pad_start > 0 or (pad and n_pad_end > 0):
            ret = torch.nn.functional.pad(
                ret, (0, 0, 0, 0, 0, 0, n_pad_start, n_pad_end if pad else 0))
        return ret
    


def _print_info_helper(src_file, labels):
        num_frames = sum([x['num_frames'] for x in labels])
        #num_events = sum([len(x['events']) for x in labels])
        print('{} : {} videos, {} frames'.format(
            src_file, len(labels), num_frames))
        
        #print('{} : {} videos, {} frames, {:0.5f}% non-bg'.format(
        #    src_file, len(labels), num_frames,
        #    num_events / num_frames * 100))

class ActionSpotDatasetJoint(Dataset):

    def __init__(
            self,
            dataset1,
            dataset2
    ):
        self._dataset1 = dataset1
        self._dataset2 = dataset2

        self._radi_displacement = self._dataset1._radi_displacement
        self._event_team = self._dataset1._event_team
        

    def __getitem__(self, unused):

        if random.random() < 0.5:
            data = self._dataset1.__getitem__(unused)
            data['dataset'] = 1
            return data
        else:
            data = self._dataset2.__getitem__(unused)
            data['dataset'] = 2
            return data
        
        

    def __len__(self):
        return self._dataset1.__len__() + self._dataset2.__len__()
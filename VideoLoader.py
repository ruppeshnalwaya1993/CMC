import random
import cv2
import numpy as np
import os
import torchvision.datasets as datasets
import traceback

class VideoLoader(object):

    def __init__(self, num_frames=100, step_size=1, diff_stride=0, samples_per_video=1,
                random_offset=True, path_to_offset=None):
        assert(samples_per_video > 0)
        assert(num_frames > 0)
        assert(step_size > 0)
        assert(diff_stride < step_size) # so that there is no overlap
        self.samples_per_video = samples_per_video
        self.num_frames = num_frames
        self.step_size = step_size
        self.random_offset = random_offset
        self.path_to_offset = path_to_offset
        self.diff_stride = diff_stride
        self.dummyoutput = [np.zeros((500,500,3)).astype(np.uint8)]*( num_frames*(1+(diff_stride>0)) )

    def load_single_clip(self, path):
      offset = 0
      cap = cv2.VideoCapture(path)
      frames_list = None
      try:
        step_size = random.randint(0, self.step_size)
        num_frames_video = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        num_frames_necessary = (self.num_frames-1) * step_size + 1
        diff =  num_frames_video - num_frames_necessary
        if diff > 0:
            if self.random_offset:
                offset = random.randint(0, diff)
            else:
                offset = diff // 2
        if self.path_to_offset != None:
            offset = self.path_to_offset[path]
        cap.set(cv2.CAP_PROP_POS_FRAMES, offset)

        frames_list = []
        to_use_diff = self.diff_stride > 0
        if to_use_diff > 0:
            frames_list2 = []
        idx = 0
        break_flag = False
        while(cap.isOpened() and idx < self.num_frames):
            for j in range((idx>0)*(step_size - (self.diff_stride * to_use_diff))):
                if not cap.grab():
                    break_flag = True
                    break
            if break_flag:
                break
            ret, frame = cap.retrieve()
            if not ret:
                break
            frames_list.append(frame)
            idx += 1
            if self.diff_stride > 0:
                for j in range(self.diff_stride):
                    if not cap.grab():
                        break_flag = True
                        break
                if break_flag:
                    break
                ret, frame = cap.retrieve()
                if not ret:
                    break
                frames_list2.append(frame)
        # if video is shorter than necessary
        if to_use_diff:
            # make sure that frame_list2 and frame_list are of same len
            for i in range(self.num_frames):
                if len(frames_list2) >= len(frames_list):
                    break
                if len(frames_list2) > 0:
                    frames_list2.append(frames_list2[i])
                else:
                    frames_list2.append(frames_list[i])
        for i in range(self.num_frames):
            if len(frames_list) >= self.num_frames:
                break
            frames_list.append(frames_list[i])
            if to_use_diff:
                frames_list2.append(frames_list2[i])
        cap.release()
      except Exception as err:
        cap.release()
        print(path)
        print('error loading data')
        print('exception = ',str(err))
        print(traceback.print_tb(err.__traceback__))
        with open('error_paths.txt','a') as fpt:
            fpt.write(path+'\n')
        with open('error_paths_with_info.txt','a') as fpt:
            fpt.write('%s, offset,%d,step,%d,diff,%d,numframes,%d\n'%(path,offset,step_size,self.diff_stride,self.num_frames))
        #os.remove(path)
        return (self.dummyoutput, 0)
      if to_use_diff:
        retlist = frames_list+frames_list2
      else:
        retlist = frames_list
      if random.choice([True, False]):
          retlist.reverse()
      return (retlist, offset)


    def __call__(self, path):
        if self.samples_per_video == 1:
                return self.load_single_clip(path)

        try:
            step_size = self.step_size
            per_clip_frame_indices_list = [] # list of frames indices for every clip
            cap = cv2.VideoCapture(path)
            num_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            num_frames_necessary = (self.num_frames-1) * step_size + 1
            num_diff = max(0, num_frames_video - num_frames_necessary)
            step = num_diff / (self.samples_per_video - 1)

            for j in range(self.samples_per_video):
                frame_start =  min(num_frames_video-1,int(j*step))
                frame_end = min(num_frames_video, frame_start + num_frames_necessary)
                num_frames_sample = frame_end - frame_start
                diff =  num_frames_sample - num_frames_necessary
                offset = 0
                if diff > 0:
                    if self.random_offset:
                        offset = random.randint(0, diff)
                    else:
                        offset = diff // 2
                # set target frames to be loaded
                frames_indices = list(range(offset + frame_start, offset + frame_start + num_frames_necessary,
                                    self.step_size))
                # if video is shorter than necessary
                for i in range(self.num_frames):
                    if len(frames_indices) >= self.num_frames:
                        break
                    frames_indices.append(frames_indices[i])
                per_clip_frame_indices_list.append(frames_indices)

            unique_frame_indices = set([item for sublist in per_clip_frame_indices_list for item in sublist])
            unique_index_to_frame = {}
            idx = 0
            frames_collected = 0
            frames_to_be_collected = len(unique_frame_indices)
            while(cap.isOpened() and frames_collected < frames_to_be_collected):
                if idx in unique_frame_indices:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    unique_index_to_frame[idx] = frame
                    frames_collected += 1
                else:
                    if not cap.grab():
                        break
                idx += 1

            clips_list = []
            for indices_list in per_clip_frame_indices_list:
                clips_list.append([ unique_index_to_frame[idx] for idx in indices_list])

            cap.release()
            return (clips_list, 0)
        except Exception as err:
            cap.release()
            print(path)
            print('error loading data')
            print('exception = ',str(err))
            print(traceback.print_tb(err.__traceback__))
            with open('error_paths.txt','a') as fpt:
                fpt.write(path+'\n')
            return ([self.dummyoutput]*self.samples_per_video, 0)



class MyVideoFolder(datasets.DatasetFolder):
    def __init__(self, root, loader, extensions=['.mp4'], transform=None, target_transform=None, error_paths=None, data_file_paths=None):
        super(MyVideoFolder, self).__init__(root=root, loader=loader, extensions=extensions, transform=transform, target_transform=target_transform)
        if data_file_paths is not None:
            data_file_paths = set(data_file_paths)
            new_samples = [e for e in self.samples if e[0] in data_file_paths]
            print('orignal dataset len =', len(self.samples))
            print('new dataset len = ', len(new_samples))
            self.samples = new_samples
            self.targets = [e[1] for e in self.samples] 
        if error_paths is not None:
            error_paths = set(error_paths)
            new_samples = [e for e in self.samples if e[0] not in error_paths]
            print('orignal dataset len =', len(self.samples))
            print('new dataset len = ', len(new_samples))
            self.samples = new_samples
            self.targets = [e[1] for e in self.samples]

    def __getitem__(self, index):
        return super(MyVideoFolder, self).__getitem__(index), index#return image path



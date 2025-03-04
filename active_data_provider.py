import numpy as np
import h5py
import torch
from perspective_transformation import crop_all, flip_all
from augmentation_utils import draw_shadow
import os
from datetime import datetime
import cv2

class ActiveDataProvider:

    def __init__(self,h5_files,shadow_max_gamma=0,shadow_darkening_ratio=0.66,debug_flag=False, mode='train'):
        self.debug_flag = debug_flag
        self.shadow_max_gamma = shadow_max_gamma
        self.shadow_darkening_ratio = shadow_darkening_ratio

        self._load_h5_files(h5_files, mode)
        self._curvature_scaling = 1000.0


    ''' Loads the h5 files that contain the recorded driving into the memory
        Input:
            @data_path: Path where the h5 files for the active test are located or a single h5 file '''
    def _load_h5_files(self,h5_files, mode):
        self.index = 0
        self.save_path = '/home/user/8T/caizewu/science_robotics_code/training_scripts/NCP-Data/'
        if not os.path.exists(self.save_path + mode):
            os.makedirs(self.save_path + mode + '/image/')
            os.makedirs(self.save_path + mode + '/target/')
        self.data_x = None

        h5_files = sorted(h5_files)
        if(len(h5_files) == 0):
            raise ValueError('No .h5 files found!')

        for f in h5_files:
            # Open h5 file
            h5f = h5py.File(f,'r')
            h5_name = f.split('/')[-1].split('.')[0]
            # Convert int64 array to uint8 array, to reduce memory footprint
            cam_raw = np.array(h5f['camera_front'])
            print("Raw cam shape: ",str(cam_raw.shape))

            timestamp = torch.tensor(np.array(h5f['timestamp'], dtype=np.float64))
            y_raw = torch.tensor(np.array(h5f['inverse_r']))

            # Crop images immediately after loading, because we won't do augmentation (and therefore don't need the full images)
            cam_cropped = torch.tensor(crop_all(cam_raw))

            if(self.data_x is None):
                # Create new buffer
                self.data_x = cam_cropped
                self.data_y = y_raw
                self.data_ts = timestamp
            else:
                # Append to existing buffer
                self.data_x = torch.cat((self.data_x,cam_cropped), dim=0)
                self.data_y = torch.cat((self.data_y,y_raw),dim=0)
                self.data_ts = torch.cat((self.data_ts,timestamp),dim=0)

            file_date = datetime.fromtimestamp(int(timestamp[0]))

            print("Loaded file '{}' from {} containing {} images ({:0.2f} GB)".format(
                f,
                file_date.strftime("%Y-%b-%d"),
                cam_cropped.shape[0],
                cam_cropped.shape[0]*cam_cropped.shape[1]*cam_cropped.shape[2]*cam_cropped.shape[3]*1/(1024.0*1024.0*1024.0),
            ))

            # If debug flag is set, load just one file to reduce startup time
            if(self.debug_flag):
                break
        self.data_x = self.data_x.permute(0, 3, 1, 2)

    def _sample_random_shadow_gamma(self):
        gamma = np.random.uniform(low=0,high=self.shadow_max_gamma)
        gamma = 1+gamma
        # if coin flip is > than the ratio of darkened shadows (66%) -> make it lighter
        if(np.random.rand()>self.shadow_darkening_ratio):
            gamma = 1.0/gamma
        return gamma

    def _augment_with_shadow(self,img):
        thickness = np.random.randint(10,100)
        kernel_sizes = [3,5,7]
        blur = kernel_sizes[np.random.randint(0,len(kernel_sizes))]
        angle = np.random.uniform(low=0,high=np.pi)
        offset_x = np.random.randint(-100,100)
        offset_y = np.random.randint(-30,30)
        gamma = self._sample_random_shadow_gamma()
        img_merged = draw_shadow(img,thickness,blur,angle,offset_x,offset_y,gamma)
        return img_merged
        
    # Counts how many iterations with the given batch_size and sequence length
    # are needed to iterate over all data
    def count_epoch_size(self, batch_size, seq_len):
        return self.data_y.shape[0]//(batch_size*seq_len)

    ''' Prints information about the loaded data '''
    def summary(self,set_name="dataset"):
        print("----------------------------------------------")
        print("Summary of {}".format(set_name))

        frameskips = self.data_ts[1:] - self.data_ts[:-1]
        sampling_T = np.median(frameskips)

        total_images = self.data_x.shape[0]
        total_seconds = int(sampling_T*total_images)
        print('Total number of samples: {} ({:02d}:{:02d} at {:0.0f} Hz)'.format(
            total_images,
            total_seconds // 60,
            total_seconds % 60,
            1.0/sampling_T,
        ))

        total_memory= total_images*self.data_x.shape[1]*self.data_x.shape[2]*self.data_x.shape[3]*1/(1024.0*1024.0*1024.0)
        print('Total memory footprint of images: {0:.2f} GB'.format(total_memory))

        print("Curvature distribution (mean: {:0.2f})".format(np.mean(self.data_y.numpy())*self._curvature_scaling))
        hist,bin_edges=np.histogram(self.data_y*self._curvature_scaling, bins=[-60,-15,-5,0,5,15,60])
        hist = hist/np.sum(hist)
        for i in range(len(hist)):
            print("[{:0.2f}, {:0.2f}]: {:0.2f}%".format(
                bin_edges[i],bin_edges[i+1],
                100*hist[i]
            ))

        print("----------------------------------------------")

    ''' Shuffles the training data and iterates over it in mini-batches
        Use this function to train Feed-forward networks  '''
    def iterate_shuffled_train(self,batch_size):
        # Shuffle complete data
        p = np.random.permutation(np.arange(self.data_x.shape[0]))

        iterations_per_epoch = self.data_x.shape[0]//batch_size

        for j in range(iterations_per_epoch):
            sample_inds = np.arange(j*batch_size, batch_size*(j+1))
            inds = np.sort(p[sample_inds])
            samples = self.data_x[inds].astype(np.float32)/255.0
            labels = self.data_y[inds]
            if(self.shadow_max_gamma > 0.0):
                for s in range(samples.shape[0]):
                    samples[s] = self._augment_with_shadow(samples[s])

            # cv2.imshow("f",samples[0]/np.max(samples[0]))
            # cv2.waitKey(100)
            yield(samples,labels*self._curvature_scaling)

    ''' Creats a batch of training sequences, use this function to train RNNs
        Input: 
            @batch_size: Size of the batch 
            @sequence_length: Sequence length of each item 
        Output:
            @batched_x: numpy array of size [sequence_length,batch_size,...]
                        (Time major)
            @batched_y: numpy array of size [sequence_length,batch_size,1]
                        (Time major)
             '''

    def create_sequenced_batch(self, batch_size=32, sequence_length=16):
        """
        生成序列化的批次数据，用于训练 RNN。
        输出:
            batched_x: shape 为 [batch_size, sequence_length, C, H, W]
            batched_y: shape 为 [batch_size, sequence_length, 1]
        """
        # 创建空的批次缓冲区
        batched_x = np.zeros(
            [batch_size, sequence_length, self.data_x.shape[1], self.data_x.shape[2], self.data_x.shape[3]])
        batched_y = np.zeros([batch_size, sequence_length, 1])

        # 填充批次缓冲区
        for i in range(batch_size):
            # 随机采样一个起始点
            uniform_sample = np.random.randint(0, self.data_y.shape[0] - sequence_length)

            # 检查样本是否有效（是否包含帧跳跃）
            is_sample_valid = False
            while not is_sample_valid:
                sequence_indices = np.arange(uniform_sample, uniform_sample + sequence_length)
                timestamps = self.data_ts[sequence_indices]
                ts_diffs = timestamps[1:] - timestamps[:-1]

                if np.max(ts_diffs.numpy()) >= 5.0:
                    # 如果帧跳跃超过 5 秒，则重新采样
                    uniform_sample = np.random.randint(0, self.data_y.shape[0] - sequence_length)
                else:
                    is_sample_valid = True

            # 将有效样本复制到批次缓冲区
            # 将 data_x 的 shape 从 [sequence_length, H, W, C] 转换为 [sequence_length, C, H, W]
            sequence_images = self.data_x[sequence_indices].float() / 255.0  # [sequence_length, H, W, C]
            # sequence_images = np.transpose(sequence_images, (0, 3, 1, 2))  # [sequence_length, C, H, W]
            batched_x[i] = sequence_images  # [sequence_length, C, H, W]

            # 将 data_y 复制到批次缓冲区
            batched_y[i] = self.data_y[sequence_indices]

            # 如果需要，对序列中的每一帧进行数据增强
            if self.shadow_max_gamma > 0.0:
                for t in range(sequence_length):
                    # 将图像从 [C, H, W] 转换回 [H, W, C] 进行增强
                    img = np.transpose(batched_x[i, t], (1, 2, 0))  # [H, W, C]
                    img = self._augment_with_shadow(img)
                    # 将增强后的图像转换回 [C, H, W]
                    batched_x[i, t] = np.transpose(img, (2, 0, 1))  # [C, H, W]

        # 转换为 PyTorch 张量
        batched_x = torch.from_numpy(batched_x).float().to(0)  # [batch_size, sequence_length, C, H, W]
        batched_y = torch.from_numpy(batched_y).float().to(0)  # [batch_size, sequence_length, 1]

        return batched_x, batched_y * self._curvature_scaling
    ''' Iterates over all the data split '''
    def iterate_as_single_sequence(self,max_seq_len=16):
        # Count in how much mini-sequnces we can split the data
        number_of_chunks = self.data_x.shape[0]//max_seq_len
        # Flag indicating if there was a frameskip
        frameskip = True
        for i in range(number_of_chunks):
            sequence_index = np.arange(i*max_seq_len,(i+1)*max_seq_len)

            timestamps = self.data_ts[sequence_index]
            ts_diffs = timestamps[1:]-timestamps[:-1]

            # Remove frameskips
            if(np.max(ts_diffs.numpy()) >= 5.0):
                frameskip = True
                continue

            y = self.data_y[sequence_index].to(0)
            x = self.data_x[sequence_index].to(0)

            yield (x,y*self._curvature_scaling,frameskip)
            frameskip = False
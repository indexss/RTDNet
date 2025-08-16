
import torch.utils.data as data
from PIL import Image
import torch
import numpy as np
import os
import cv2


class EyediapLoaderTest(data.Dataset):
    def __init__(self, source_path, config, transforms=None, person_id=None, no_padding=False):
        self.transforms = transforms

        self.config = config
        self.source_path = source_path
        self.no_padding = no_padding

        self.all_eyediap_mapping = {}

        all_label_files_list = [f for f in os.listdir(source_path + '/Label_test/') if
                                f.endswith('label')]

        if person_id is not None:
            all_label_files_list = [p for p in all_label_files_list if p.split('.')[0] in person_id]

        for ff in all_label_files_list:

            pid = ff.split('.')[0]

            with open(source_path + '/Label_test/' + ff) as infile:
                lines = infile.readlines()
                header = lines.pop(0)

            eyediap_img_mapping = {}

            for K in lines:
                face_path = os.path.join(source_path, '/Image', K.split(' ')[0])

                orig_path = K.strip().split(' ')[3]
                session_str = "_".join(orig_path.split('_')[:-1])
                timestamp = int(orig_path.split('_')[-1])

                # assert (timestamp - 1) % 15 == 0

                if session_str not in eyediap_img_mapping:
                    eyediap_img_mapping[session_str] = {}
                    eyediap_img_mapping[session_str]['timestamps'] = []
                    time_sections = []

                label = K.strip().split(' ')[4].split(",")
                label = np.array(label).astype('float')

                head_3d = K.strip().split(' ')[5].split(",")
                head_3d = np.array(head_3d).astype('float')

                revc = K.strip().split(' ')[8].split(",")
                revc = np.array(revc).astype('float')

                sevc = K.strip().split(' ')[9].split(",")
                sevc = np.array(sevc).astype('float')

                origin = K.strip().split(' ')[10].split(",")
                origin = np.array(origin).astype('float')

                label2 = K.strip().split(' ')[6].split(",")
                label2 = np.array(label2).astype('float')[::-1]  # originally [yaw,pitch], store as [pitch,yaw]

                head_label2 = K.strip().split(' ')[7].split(",")
                head_label2 = np.array(head_label2).astype('float')[::-1]  # originally [yaw,pitch], store as [pitch,yaw]

                time_sections.append(timestamp)
                if len(time_sections) == 30:
                    eyediap_img_mapping[session_str]['timestamps'].append(time_sections)
                    time_sections = []

                eyediap_img_mapping[session_str][timestamp] = {'face_path': face_path, 
                                                               '2D_gaze': label2,
                                                               '3D_gaze': label, 
                                                               '2D_head': head_label2,
                                                               '3D_head': head_3d,
                                                               'revc': revc,
                                                               'sevc': sevc,
                                                               'origin': origin}
                

            self.all_eyediap_mapping[pid] = eyediap_img_mapping

        self.meta_data = []

        for person_id, data1 in self.all_eyediap_mapping.items():
            for session, data2 in data1.items():
                for tt in data2['timestamps']:
                    self.meta_data.append({'session': session,
                                           'timestamp_window': tt,
                                           'person': person_id})
    def preprocess_frames(self, frames):
        if self.no_padding:
            return frames
        if self.transforms is not None:
            imgss = [self.transforms(frames[k]) for k in range(frames.shape[0])]
            return torch.stack(imgss, dim=0)
        else:
            # Expected input:  N x H x W x C
            # Expected output: N x C x H x W
            frames = np.transpose(frames, [0, 3, 1, 2])
            # frames = np.transpose(frames, [2, 0, 1])
            frames = frames.astype(np.float32)
            # frames *= 1.0 / 255.0
            frames *= 2.0 / 255.0
            frames -= 1.0
            return frames

    def __getitem__(self, index):
        meta_source = self.meta_data[index]

        all_timestamps = meta_source['timestamp_window']
        session_ = meta_source['session']
        person_ = meta_source['person']

        entry_data = self.all_eyediap_mapping[person_][session_]

        subentry = {}

        source_video = []
        validity = []
        validity_pog = []
        gaze_pitchyaw = []
        gaze_xyz = []
        head_pitchyaw = []
        head_xyz = []
        revc = []
        sevc = []
        origin = []
        revc_matrix = []

        count = 0
        for i, timestep in enumerate(all_timestamps):
            if timestep in entry_data.keys():
                face_path = self.source_path + entry_data[timestep]['face_path']
                img = Image.open(face_path).convert('RGB')
                img = img.resize((self.config.face_size[0], self.config.face_size[1]))
                img = img.resize((self.config.face_size[0], self.config.face_size[1]))
                source_video.append(np.array(img))
                validity.append(1)
                validity_pog.append(1)
                gaze_pitchyaw.append(entry_data[timestep]['2D_gaze'])
                gaze_xyz.append(entry_data[timestep]['3D_gaze'])
                head_pitchyaw.append(entry_data[timestep]['2D_head'])
                head_xyz.append(entry_data[timestep]['3D_head'])
                revc.append(entry_data[timestep]['revc'])
                # print(entry_data[timestep]['revc'].shape)
                revcmat = cv2.Rodrigues(entry_data[timestep]['revc'])[0]
                revc_matrix.append(revcmat)
                sevc.append(entry_data[timestep]['sevc'])
                origin.append(entry_data[timestep]['origin'])
                count += 1
            else:
                if count == 0:
                    continue
                else:
                    source_video.append(source_video[-1].copy())
                    validity.append(0)
                    validity_pog.append(0)
                    gaze_pitchyaw.append(gaze_pitchyaw[-1].copy())
                    gaze_xyz.append(gaze_xyz[-1].copy())
                    head_pitchyaw.append(head_pitchyaw[-1].copy())
                    head_xyz.append(head_xyz[-1].copy())
                    revc.append(revc[-1].copy())
                    revc_matrix.append(revc_matrix[-1].copy())
                    sevc.append(sevc[-1].copy())
                    origin.append(origin[-1].copy())
                    count += 1

        if len(source_video) == 0:
            return None

        source_video = np.array(source_video).astype(np.float32)
        gaze_pitchyaw = np.array(gaze_pitchyaw).astype(np.float32)
        gaze_xyz = np.array(gaze_xyz).astype(np.float32)
        head_pitchyaw = np.array(head_pitchyaw).astype(np.float32)
        head_xyz = np.array(head_xyz).astype(np.float32)
        revc = np.array(revc).astype(np.float32)
        revc_matrix = np.array(revc_matrix).astype(np.float32)
        sevc = np.array(sevc).astype(np.float32)
        origin = np.array(origin).astype(np.float32)
        validity = np.array(validity).astype(int)
        validity_pog = np.array(validity_pog).astype(int)

        source_video = self.preprocess_frames(source_video)

        subentry['face_patch'] = source_video
        subentry['face_g_tobii'] = gaze_pitchyaw
        subentry['face_g_tobii_validity'] = validity
        subentry['face_g_tobii_3d'] = gaze_xyz
        subentry['face_h'] = head_pitchyaw
        subentry['face_h_3d'] = head_xyz
        subentry['face_revc'] = revc
        subentry['face_revc_matrix'] = revc_matrix
        subentry['face_sevc'] = sevc
        subentry['face_o'] = origin
        subentry['face_PoG_tobii_validity'] = validity_pog

        for key, value in subentry.items():
            if value.shape[0] < self.config.max_sequence_len:
                pad_len = self.config.max_sequence_len - value.shape[0]

                if pad_len > 0:
                    subentry[key] = np.pad(
                        value,
                        pad_width=[(0, pad_len if i == 0 else 0) for i in range(value.ndim)],
                        mode='constant',
                        constant_values=(False if value.dtype is np.bool else 0.0),
                    )

        torch_entry = dict([
            (k, torch.from_numpy(a)) if isinstance(a, np.ndarray) else (k, a)
            for k, a in subentry.items()
        ])

        return torch_entry

    def __len__(self):
        return len(self.meta_data)

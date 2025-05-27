from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import os
from torch.autograd import Variable
import pandas as pd
from itertools import combinations

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 创建成功")


def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask==0))
    return np_mask

GENDER_LIST  = ['男 Male', '女 Female']
GENDER_INDEX = {gender: idx for idx, gender in enumerate(GENDER_LIST)}
GENDER_MATRIX = torch.tensor([
    [0.3, 1.0],
    [1.0, 0.6]
], dtype=torch.float)


SCHOOL_LIST = ["理工学院 SSE", "数据科学学院 SDS", "经管学院 SME", 
                "人文社科学院 SHSS", "金融工程 FE", "音乐学院 MUS", 
                "医学院非临床 MED(nonclinical)", "医学院临床 CMED"]
SCHOOL_INDEX = {school: idx for idx, school in enumerate(SCHOOL_LIST)}
SCHOOL_MATRIX = torch.tensor([
    [1.0, 0.8, 0.5, 0.2, 0.7, 0.1, 0.6, 0.3],
    [0.8, 1.0, 0.5, 0.2, 0.7, 0.1, 0.6, 0.3],
    [0.5, 0.5, 1.0, 0.6, 0.8, 0.1, 0.4, 0.3],
    [0.2, 0.2, 0.5, 1.0, 0.2, 0.1, 0.2, 0.2],
    [0.7, 0.7, 0.8, 0.2, 1.0, 0.1, 0.5, 0.2],
    [0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1],
    [0.6, 0.6, 0.4, 0.2, 0.5, 0.1, 1.0, 0.5],
    [0.3, 0.3, 0.3, 0.2, 0.2, 0.1, 0.5, 1.0]
], dtype=torch.float)


COLLEGE_LIST = ['逸夫书院 Shaw', '祥波书院 Harmonia', '学勤书院 Diligentia',
                '厚含书院 Minerva', '思廷书院 Muse', '道扬书院 Ling', '第七书院 The Seventh']
COLLEGE_INDEX = {college: idx for idx, college in enumerate(COLLEGE_LIST)}
COLLEGE_MATRIX = torch.tensor([
    [1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 
    [0.1, 1.0, 0.4, 0.3, 0.5, 0.6, 0.1],
    [0.1, 0.3, 1.0, 0.5, 0.6, 0.4, 0.1],
    [0.1, 0.3, 0.5, 1.0, 0.4, 0.3, 0.5],
    [0.1, 0.5, 0.6, 0.4, 1.0, 0.6, 0.1],
    [0.1, 0.6, 0.4, 0.3, 0.6, 1.0, 0.1],
    [0.1, 0.1, 0.1, 0.5, 0.1, 0.1, 1.0]
], dtype=torch.float)


MBTI_LIST = [
    'ISTJ', 'ISTP', 'ESTP', 'ESTJ', 'ISFJ', 'ISFP', 'ESFP', 'ESFJ',
    'INFJ', 'INFP', 'ENFP', 'ENFJ', 'INTJ', 'INTP', 'ENTP', 'ENTJ'
]
MBTI_INDEX = {mbti: idx for idx, mbti in enumerate(MBTI_LIST)}
MBTI_MATRIX = torch.tensor([
    [1.00, -0.06, 0.33, 0.43, -0.28, -0.49, -0.37, 0.29, -0.36, -0.48, -0.35, -0.03, -0.02, -0.24, -0.09, 0.35],
    [-0.06, 1.00, 0.06, 0.17, 0.14, 0.55, 0.24, 0.14, -0.30, -0.10, -0.33, -0.34, -0.44, -0.19, -0.66, 0.40],
    [0.33, 0.06, 1.00, 0.71, -0.66, -0.29, 0.14, -0.01, 0.65, -0.69, -0.45, -0.34, -0.32, 0.41, 0.07, 0.12],
    [0.43, 0.17, 0.71, 1.00, -0.63, -0.47, 0.09, 0.27, 0.58, -0.77, -0.68, 0.19, 0.06, -0.36, 0.25, 0.43],
    [-0.28, 0.17, -0.66, -0.63, 1.00, 0.57, 0.26, 0.41, 0.26, 0.47, 0.18, -0.19, -0.27, -0.16, -0.36, 0.47],
    [-0.49, 0.55, -0.29, -0.47, 0.57, 1.00, 0.67, 0.49, 0.12, 0.22, 0.03, -0.19, -0.53, -0.27, -0.62, -0.73],
    [-0.37, 0.24, 0.14, 0.09, 0.26, 0.67, 1.00, 0.41, -0.48, -0.09, -0.03, -0.52, -0.75, -0.62, -0.27, -0.60],
    [0.29, 0.14, -0.01, 0.27, 0.41, 0.49, 0.41, 1.00, 0.12, 0.01, 0.14, -0.36, -0.45, 0.38, -0.30, 0.50],
    [-0.36, -0.30, 0.65, 0.58, 0.26, 0.12, -0.48, 0.12, 1.00, 0.70, 0.55, 0.54, 0.64, 0.75, 0.11, 0.01],
    [-0.48, -0.10, -0.69, -0.77, 0.47, 0.22, -0.09, 0.01, 0.70, 1.00, 0.61, 0.16, 0.19, 0.49, -0.02, -0.34],
    [-0.35, -0.33, -0.45, -0.68, 0.18, 0.03, -0.03, 0.14, 0.55, 0.61, 1.00, 0.27, 0.03, 0.32, 0.14, -0.25],
    [-0.03, -0.34, -0.34, 0.19, -0.19, -0.19, -0.52, -0.36, 0.54, 0.16, 0.27, 1.00, 0.65, 0.56, 0.24, 0.52],
    [-0.02, -0.44, -0.32, 0.06, -0.27, -0.53, -0.75, -0.45, 0.64, 0.19, 0.03, 0.65, 1.00, 0.74, 0.38, 0.64],
    [-0.24, -0.19, 0.41, -0.36, -0.16, -0.27, -0.62, 0.38, 0.75, 0.49, 0.32, 0.56, 0.74, 1.00, 0.33, 0.52],
    [-0.09, -0.66, 0.07, 0.25, -0.36, -0.62, -0.27, -0.30, 0.11, -0.02, 0.14, 0.38, 0.38, 0.33, 1.00, 0.52],
    [0.35, 0.40, 0.12, 0.43, 0.47, -0.73, -0.60, 0.50, 0.01, -0.34, -0.25, 0.52, 0.64, 0.52, 0.52, 1.00]
], dtype=torch.float)
MBTI_MATRIX = (MBTI_MATRIX + 1) / 2


class MatchingDataset(Dataset):
    def __init__(self,path):
        df=pd.read_excel(path)
        df=df[['gender','school','college','MBTI']].dropna().reset_index(drop=True)
        self.gender_map={'男 Male':0,'女 Female':1}
        ALL_SCHOOLS = ['理工学院 SSE', '数据科学学院 SDS', '经管学院 SME', 
                       '人文社科学院 SHSS', '金融工程 FE', '音乐学院 MUS', 
                       '医学院非临床 MED(Nonclinical)','医学院临床 CMED']
        self.school_map = {school: i for i, school in enumerate(ALL_SCHOOLS)}
        ALL_COLLEGES = ['逸夫书院 Shaw', '祥波书院 Harmonia', '学勤书院 Diligentia',
                        '厚含书院 Minerva', '思廷书院 Muse', '道扬书院 Ling','第七书院 The Seventh']
        self.college_map = {college: i for i, college in enumerate(ALL_COLLEGES)}
        ALL_MBTI_TYPES = [
            'ISTJ', 'ISFJ', 'INFJ', 'INTJ', 
            'ISTP', 'ISFP', 'INFP', 'INTP', 
            'ESTP', 'ESFP', 'ENFP', 'ENTP', 
            'ESTJ', 'ESFJ', 'ENFJ', 'ENTJ'
        ]
        self.mbti_map = {mbti: i for i, mbti in enumerate(ALL_MBTI_TYPES)}
        self.mbti_unknow_id =  len(ALL_MBTI_TYPES)

        self.users = []
        for _, row in df.iterrows():
            gender = torch.tensor(self.gender_map[row['gender']], dtype=torch.long)
            school = torch.tensor(self.school_map[row['school']], dtype=torch.long)
            college = torch.tensor(self.college_map[row['college']], dtype=torch.long)
            
            mbti_str = str(row['MBTI']).upper().strip()
            mbti_idx = self.mbti_map.get(mbti_str, self.mbti_unknow_id)
            mbti = torch.tensor(mbti_idx, dtype=torch.long)
            self.users.append({
                'gender': gender,
                'school': school, 
                'college': college, 
                'mbti': mbti,
                'mbti_str': mbti_str
            })

        self.pairs = []
        for userA, userB in combinations(self.users, 2):
            g1, g2 = userA['gender'], userB['gender']
            gender_score = GENDER_MATRIX[GENDER_MATRIX[g1], GENDER_MATRIX[g2]]
            
            s1, s2 = userA['school'], userB['school']
            school_score = SCHOOL_MATRIX[SCHOOL_MATRIX[s1], SCHOOL_MATRIX[s2]] 

            c1, c2 = userA['college'], userB['college']
            college_score = COLLEGE_MATRIX[COLLEGE_MATRIX[c1], COLLEGE_MATRIX[c2]]

            m1, m2 = userA['mbti_str'], userB['mbti_str']
            mbti_score = MBTI_MATRIX[MBTI_INDEX[m1], MBTI_INDEX[m2]] if m1 in MBTI_INDEX and m2 in MBTI_INDEX else 0.0
            
            label = torch.tensor([gender_score, school_score, college_score, mbti_score], dtype=torch.float)
            self.pairs.append((
                {
                    'gender': userA['gender'],
                    'school': userA['school'],
                    'college': userA['college'],
                    'mbti': userA['mbti']
                },
                {
                    'gender': userB['gender'],
                    'school': userB['school'],
                    'college': userB['college'],
                    'mbti': userB['mbti']
                },
                label
            ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def custom_collate(batch):
    userA_list, userB_list, label_list = zip(*batch)

    def collate_dict(list_of_dicts):
        return {
            key: torch.stack([d[key] for d in list_of_dicts])
            for key in list_of_dicts[0]
        }

    batch_userA = collate_dict(userA_list)
    batch_userB = collate_dict(userB_list)
    labels = torch.stack(label_list)

    return batch_userA, batch_userB, labels


def create_matching_dataloader(excel_path, batch_size=16, shuffle=True):
    dataset = MatchingDataset(excel_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate)



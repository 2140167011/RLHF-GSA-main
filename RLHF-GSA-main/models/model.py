import torch
import torch.nn as nn
import torch.nn.functional as F

class MatchingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gender_emb = nn.Embedding(2, 4)
    
        self.college_emb = nn.Embedding(7, 64)
        
        self.school_emb = nn.Embedding(8, 64)

        self.interest_fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Softmax(dim=1)  
        )

        self.mbti_emb = nn.Embedding(17, 512)

        self.final_fc = nn.Linear(4, 1)  
        self.weight = nn.Parameter(torch.tensor([1, 1, 1, 1],dtype=torch.float))  

    def forward(self, userA, userB, interestA=None, interestB=None):
        gA,sA,cA,mA = userA['gender'],userA['school'],userA['college'],userA['mbti']
        gB,sB,cB,mB = userB['gender'],userB['school'],userB['college'],userB['mbti']
        gender_sim = F.cosine_similarity(
            self.gender_emb(gA),
            self.gender_emb(gB),
            dim=-1
        )

        college_sim = F.cosine_similarity(
            self.college_emb(cA),
            self.college_emb(cB),
            dim=-1
        )
        school_sim = F.cosine_similarity(
            self.school_emb(sA),
            self.school_emb(sB),
            dim=-1
        )

        mbti_sim = F.cosine_similarity(
            self.mbti_emb(mA),
            self.mbti_emb(mB),
            dim=-1
        )

        sims = [gender_sim, college_sim, school_sim, mbti_sim]

        if interestA is not None and interestB is not None:
            interestA_vec = self.interest_fc(interestA)
            interestB_vec = self.interest_fc(interestB)
            interest_sim = (interestA_vec * interestB_vec).sum(dim=-1)
            sims.append(interest_sim)

        all_features = torch.stack(sims, dim=1)
        
        weighted = all_features * self.weight
        final_score = self.final_fc(weighted)
        
        return torch.sigmoid(final_score)  



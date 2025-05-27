import os
from queue import PriorityQueue
import pandas as pd
import json
import heapq

os.chdir("C:\\Users\\23835\\Desktop\\Project\\RLHF-GSA")
INF = 1e9
num_map = {2: 6, 3: 8, 4: 10}
mentor = pd.read_excel(".//mentor.xlsx")
mentee = pd.read_excel(".//mentee.xlsx")
num_mentor = len(mentor)
num_mentee = len(mentee)

female_mentee = mentee[mentee["gender"] == '女 Female'].reset_index()
male_mentee = mentee[mentee["gender"] == '男 Male'].reset_index()
num_female_mentee = len(female_mentee)
num_male_mentee = len(male_mentee)

dist = {}
head = {}
cur = {}
for i in range(num_mentee):
    id = int(mentee.at[i, "id"])
    dist[id] = [[0, INF, -1]]
    head[id] = 0

quota_male = {}
quota_female = {}
quota_all = {}
pq = {}
res = {}
mentor["num"] += 1

pq = {mentor.at[i, "gid"]: [] for i in range(num_mentor)}
for i in range(num_mentor):
    quota_all[mentor.at[i, "gid"]] = num_map[mentor.at[i, "num"]]
    quota_male[mentor.at[i, "gid"]] = num_map[mentor.at[i, "num"]] // 2
    res[int(mentor.at[i, "gid"])] = []

print("# of mentor = {}, # of mentee = {}".format(num_mentor, num_mentee))
print("# male mentee = {}, # of female mentee = {}".format(num_male_mentee, num_female_mentee))
print("2 mentors take {} mentees, 3 mentors take {} mentees, 4 mentors take {} mentees. ".format(num_map[2], num_map[3], num_map[4]))


#############################TO-DO#######################################
# calculate how many mentees could be accepted. If not, raise a warning #
#########################################################################


print("============ preparation done =================")


def distance(school_mentee, college_mentee, interest_mentee, interest_mentor, idx):
    ##### info of mentors loaded #####
    num_mentor_in_group = mentor.at[idx, "num"]
    schools = []; colleges = []
    for i in range(num_mentor_in_group):
        schools.append(mentor.at[idx, "school"+str(i+1)])
        colleges.append(mentor.at[idx, "college"+str(i+1)])
    ##### calculate distance #####
    weights = {"school": 100, "college": 50, "interest": 1}
    distance_school = 1 if school_mentee in schools else 0
    #print(schools, school_mentee)
    #########################TO-DO#################################
    # introduce a matrix for better digitalize distance of school #
    ###############################################################

    distance_college = 1 if college_mentee in colleges else 0
    ##################### TO-DO here as well ######################

    distance_interest = 0
    for i in range(len(interest_mentee)):
        distance_interest += interest_mentee[i] + interest_mentor[i]
    distance_interest/=200

    #  print(idx, distance_school, distance_college, distance_interest)
    return distance_school*weights["school"] + distance_college*weights["college"] + distance_interest*weights["interest"]


def calculate_distance():
    column = ["(运动 Sports)", "(艺术 Arts)", "(社交 Social Contact)", "(游戏 Game (on- or offline))", "(学习 Study)", 
                  "(书法与绘画 Calligraphy and Painting)", "(音乐与乐器 Music and Musical Instruments)", "(播音、主持、演讲、朗诵 Broadcasting, hosting, speaking, reciting)", "(摄影（含图片处理，音视频的摄制、剪辑）Photography (including image processing, audio and video recording and editing))", "(棋类 Chess)", "(舞蹈 Dancing)", "(二次元 ACGN)", "(烹饪 Cooking)", "(化妆护肤与着装（如汉服）Makeup, skin care and clothing)", 
                  "(虚拟游戏（含手游、网游等）Virtual games)", "(线下游戏（如剧本杀、密室逃脱、棋牌等）Offline games)", 
                  "(阅读、观影与写作等 Reading, vedio-watching, writing, etc.)", "(与专业相关的学术 Academics in majors)", "(其他学术（如非专业的哲学、金融理财）Non-major academic)"]

    interest_mentor = []
    for i in range(num_mentor):
        for j in column:
            interest_mentor.append(9-mentor.at[i, j])

    for i in range(num_mentee):
        id = int(mentee.at[i, "id"])
        school_mentee = mentee.at[i, "school"]; college_mentee = mentee.at[i, "college"]
        interest_mentee = []
        for k in column:
            interest_mentee.append(9-mentee.at[i, k])
        for j in range(num_mentor):
            cur_dist = distance(school_mentee, college_mentee, interest_mentee, interest_mentor, j)
            dist[id].append([mentor.at[j, "gid"], cur_dist, -1])
            q = head[id]; p = dist[id][head[id]][2]
            while dist[id][p][2] != -1 and dist[id][p][1] > dist[id][-1][1]:
                q = p
                p = dist[id][p][2]
            dist[id][-1][2] = p
            dist[id][q][2] = len(dist[id])-1


def put_in(mentee_id, quota_dict):
    while True:
        if cur[mentee_id] == -1:  # 偏好列表已耗尽
            return False

        current_pref = dist[mentee_id][cur[mentee_id]]
        mentor_gid = current_pref[0]

        # 使用最大堆维护导师组优先级（距离越大优先级越高）
        if quota_dict[mentor_gid] > 0:
            heapq.heappush(pq[mentor_gid], (-current_pref[1], mentee_id))  # 存储负值实现最大堆
            quota_dict[mentor_gid] -= 1
            return True
        else:
            # 比较当前学员与组内最低优先级学员
            if pq[mentor_gid]:
                min_priority, min_mentee = heapq.heappop(pq[mentor_gid])
                if -min_priority < current_pref[1]:  # 当前学员更优
                    heapq.heappush(pq[mentor_gid], (-current_pref[1], mentee_id))
                    # 被替换学员重新申请
                    cur[min_mentee] = dist[min_mentee][cur[min_mentee]][2]
                    if not put_in(min_mentee, quota_dict):
                        return False
                    return True
                else:
                    heapq.heappush(pq[mentor_gid], (min_priority, min_mentee))
   
            # 尝试下一个偏好
            cur[mentee_id] = dist[mentee_id][cur[mentee_id]][2]


def solve(mentee_lst, num, quota):
    for i in range(num):
        id = int(mentee_lst.at[i, "id"])
        put_in(id, quota)


def matching():
    cnt = 0
    flag = True

    solve(male_mentee, num_male_mentee, quota_male)
    for i in range(num_mentor):
        gid = mentor.at[i, "gid"]
        quota_female[gid] = quota_all[gid] - len(res[gid])  # 动态计算女性配额
        while pq[gid]: 
            cur_priority, cur_mentee = heapq.heappop(pq[gid])
            cur_res = int(cur_mentee)
            if cur_res in res[gid]:
                flag = False
            res[gid].append(cur_res)
            cnt += 1

    solve(female_mentee, num_female_mentee, quota_female)
    for i in range(num_mentor):
        gid = mentor.at[i, "gid"]
        while pq[gid]: 
            cur_priority, cur_mentee = heapq.heappop(pq[gid])
            cur_res = int(cur_mentee)
            if cur_res in res[gid]:
                flag = False
            res[gid].append(cur_res)
            cnt += 1
    print("all the mentees have been divided into mentor groups:", cnt == num_mentee and flag, num_mentee-cnt, "remains")


calculate_distance()
for i in range(num_mentee):
    id = int(mentee.at[i, "id"])
    cur[id] = dist[id][head[id]][2]


matching()
print("done")
print("=================================")


with open('data1.json', 'w') as json_file:
    json.dump(res, json_file)

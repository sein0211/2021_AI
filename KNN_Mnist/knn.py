import numpy as np

def calculate_distance(a, b):
    return (sum((a - b) ** 2)) # 유클리드 거리 공식을 이용한 거리 구하기


def obtain_k_nearest_neighbor(distance, k):  # 60,000개의 거리 중에 가장 가까운 k개의 거리를 구하는 함수이다.
    sort_distance = sorted(distance, key=lambda x: x[1])  # lamba를 이용해 거리를 오름차순으로 정렬하였다.
    return sort_distance[:k]  # 오름차순으로 정렬한 리스트를 k번째까지 자르고 return한다.


def weight_majority_vote(test, train_data, t_train, k):
    distance = list([i] for i in range(len(train_data)))  # train_data와 test와의 거리를 나타내기 위한 2차원 리스트 생성

    for i in range(len(train_data)):
        distance[i].append(calculate_distance(test, train_data[i])) # 60,000개의 학습 데이터와의 거리를 구한다.

    k_distance = obtain_k_nearest_neighbor(distance, k)

    for i in range(k):
        k_distance[i][1] = 1 / (k_distance[i][1])  # 거리를 d라고 하면 가중치를 (1/d)로 두었다. 이는 거리가 가까울수록 값이 더 커지게 가중치를 둔 것이다.

    k_target = [0 for i in range(10)]  # 정답의 후보인 0~9까지에 해당하는 거리값의 합을 나타내는 리스트이다.
    for i in k_distance:
        k_target[t_train[i[0]]] += i[1]  # 위에서 가중치를 둔 거리값을 해당하는 인덱스에 더한다.

    return k_target.index(max(k_target))  # 거리값의 합이 제일 큰 인덱스값을 return한다.


def hand_craft(train_data): # 28*28인 input feature을 56으로 가공하기 위한 함수이다.
    dim_train_data = list(0 for i in range(56)) # 크기가 56인 리스트를 만든다.

    for i in range(28): # 앞의 28개에는 행에서 배경이 아닌 숫자에 해당하는 픽셀 수를 넣었다.
        sum = 0
        for j in range(i * 28, (i + 1) * 28):
            if train_data[j] != 0: # 배경이 아니면
                sum += 1 # 1을 증가
        dim_train_data[i] = sum

    for i in range(28): # 뒤의 28개에는 열에서 배경이 아닌 숫자에 해당하는 픽셀 수를 넣었다.
        sum = 0
        for j in range(i, 28 * 28, 28):
            if train_data[j] != 0: # 배경이 아니면
                sum += 1 # 1을 증가
        dim_train_data[i + 28] = sum

    return np.array(dim_train_data) # 위의 knn 알고리즘 함수를 이용하려고 리스트를 numpy 배열로 바꿨다.
import sys, os
sys.path.append(os.pardir) # 부모 디렉토리에서 import할 수 있도록 설정

import numpy as np
from dataset.mnist import load_mnist # mnist data load할 수 있는 함수 import

import time
import knn

(x_train, t_train), (x_test, t_test)=load_mnist(flatten=True, normalize=False)
x_train=np.int_(x_train) # uint8타입을 int32로 바꿔주었다
x_test=np.int_(x_test) # uint8타입을 int32로 바꿔주었다
# training data, test data
# flatten: 이미지를 1차원 배열로 읽음
# normalize: 0~1 실수로. 그렇지 않으면 0~255

print("k: ",end="")
k = int(input())  # k를 입력받음

print("size: ",end="")
size = int(input())  # 테스트할 케이스 수를 입력받음

label_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
sample = np.random.randint(0, t_test.shape[0], size) # 10,000개의 테스트 케이스 중에 랜덤으로 size개를 고르기

count0 = 0 # weight knn으로 테스트했을 때 맞은 개수
count1 = 0 # hand-craft knn으로 테스트했을 때 맞은 개수

# start = time.time()
# for i in sample:
#     real_answer = t_test[i] # 실제 정답
#     my_answer = knn.weight_majority_vote(x_test[i], x_train, t_train, k)  # 내가 구한 정답
#
#     if real_answer == my_answer:
#         count0 += 1 # 내가 구한 정답이 맞으면 카운트수를 1 증가시킨다
#     print(my_answer, real_answer)
# print("accuracy =", count0 / size) # 정확도 계산
# print("time :", time.time() - start)

new_x_train = np.zeros((60000, 56), dtype='int32') # 원래 28*28인 학습 데이터의 input feature을 56으로 가공하기 위한 새로운 numpy 배열
for i in range(len(x_train)):
    new_x_train[i] = knn.hand_craft(x_train[i]) # 가공

new_x_test = np.zeros((10000, 56), dtype='int32') # 원래 28*28인 테스트 데이터의 input feature을 56으로 가공하기 위한 새로운 numpy 배열
for i in range(len(x_test)):
    new_x_test[i] = knn.hand_craft(x_test[i]) # 가공

# start1 = time.time()
for i in sample:
    real_answer = t_test[i] # 실제 정답
    my_answer = knn.weight_majority_vote(new_x_test[i], new_x_train, t_train, k)  # 내가 구한 정답. 가공한 데이터로 weighted knn 알고리즘을 이용하였다.

    if real_answer == my_answer:
        count1 += 1 # 내가 구한 정답이 맞으면 카운트수를 1 증가시킨다
    print(i, "th data", " result", my_answer, " label", real_answer)
print("accuracy =", count1 / size) # 정확도 계산
# print("time :", time.time() - start1)
def calculate_distance(i0, i1,X):  # 유클라디안 거리 공식을 이용해 두 점 사이의 거리를 구하는 함수이다.
    # 유클라디안 거리 공식에 대한 설명은 리포트에서 3.알고리즘에 대한 설명란에 써놓았다.
    a = X[i0]
    b = X[i1]
    return (sum((a - b) ** 2))


def obtain_k_nearest_neighbor(distance, k):  # 140개의 거리 중에 가장 가까운 k개의 거리를 구하는 함수이다.
    sort_distance = sorted(distance, key=lambda x: x[1])  # lamba를 이용해 거리를 오름차순으로 정렬하였다.

    return sort_distance[:k]  # 오름차순으로 정렬한 리스트를 k번째까지 자르고 return한다.


def majority_vote(test, train_data, k,X,y):
    distance = list([i] for i in train_data)  # train_data와 test와의 거리를 나타내기 위한 2차원 리스트 생성

    for i in range(len(train_data)):
        distance[i].append(
            calculate_distance(test, train_data[i],X))  # [[인덱스, 거리],[인덱스, 거리],...,[인덱스, 거리]] 이런식으로 2차원 리스트를 채운다.

    k_distance = obtain_k_nearest_neighbor(distance, k)  # 가장 가까운 k개의 [인덱스, 거리]가 나온다.

    k_target = [0 for i in range(3)]  # y값 0,1,2가 k개 중에 각각 몇 개가 있는지 나타내는 리스트이다.(0,0,0으로 초기화)

    for i in k_distance:
        k_target[y[i[0]]] += 1

    return k_target.index(max(k_target))  # 가장 많은 개수를 가진 y값을 return한다.


def weight_majority_vote(test, train_data, k,X,y):
    distance = list([i] for i in train_data)  # train_data와 test와의 거리를 나타내기 위한 2차원 리스트 생성

    for i in range(len(train_data)):
        distance[i].append(calculate_distance(test, train_data[i],X))

    k_distance = obtain_k_nearest_neighbor(distance, k)

    for i in range(k):
        k_distance[i][1] = 1 / (k_distance[i][1] ** 2 + 1)  # 거리를 d라고 하면 가중치를 1/(d^2+1)로 한다. 이는 거리가 가까울수록 값이 더 커지게 가중치를 둔 것이다.

    k_target = [0 for i in range(3)] #y값 0,1,2에 해당하는 거리값의 합을 나타내는 리스트이다.(0,0,0으로 초기화)
    for i in k_distance:
        k_target[y[i[0]]] += i[1] #앞에서 가중치를 곱한 거리값을 해당하는 인덱스에 더한다.

    return k_target.index(max(k_target)) #거리값의 합이 제일 큰 y값을 return한다.
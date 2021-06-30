import knn
from sklearn.datasets import load_iris

iris=load_iris()

X=iris.data #iris꽃의 sepal length(cm), sepal width(cm), petal length(cm), petal width(cm) data

y=iris.target #iris꽃 이름을 나타내는 정수형 데이터 (0은 setosa를, 1은 versicolor를, 2는 virginica를 나타낸다)

y_name=iris.target_names #iris꽃 이름 데이터 (setosa,versicolor,virginica)

print("k: ",end="")
k = int(input())  # k를 입력받음
t = 15

test_data = [i for i in range(len(iris.data)) if (i + 1) % 15 == 0]  # 150개의 iris data 150개를 15로 나누어 15번째의 각 data -> [14, 29, 44, 59, 74, 89, 104, 119, 134, 149]

train_data = [i for i in range(len(iris.data)) if i not in test_data]  # 150개의 data중 test_data를 뺀 나머지 data


print("<majority_vote result>")  # 아래 for문은 majority_vote에 대한 result를 구하는 것이다
for i in range(len(test_data)):
    answer = y_name[y[test_data[i]]]  # 진짜 정답
    my_answer = y_name[knn.majority_vote(test_data[i], train_data, k, X, y)]  # 내가 구한 정답

    if answer == my_answer:
        print("Test Data Index:", i, "Computed class:", my_answer, "True class:", answer)
    else:  # 내가 구한 답이 정답과 다르면 ->WRONG ANSWER을 출력한다
        print("Test Data Index:", i, "Computed class:", my_answer, "->WRONG ANSWER", "True class:", answer)

print("<weight_majority_vote result>")  # 아래 for문은 weight_majority_vote에 대한 result를 구하는 것이다
for i in range(len(test_data)):
    answer = y_name[y[test_data[i]]]  # 진짜 정답
    my_answer = y_name[knn.weight_majority_vote(test_data[i], train_data, k, X, y)]  # 내가 구한 정답

    if answer == my_answer:
        print("Test Data Index:", i, "Computed class:", my_answer, "True class:", answer)
    else:  # 내가 구한 답이 정답과 다르면 ->WRONG ANSWER을 출력한다
        print("Test Data Index:", i, "Computed class:", my_answer, "->WRONG ANSWER", "True class:", answer)
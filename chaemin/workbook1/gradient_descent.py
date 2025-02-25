import numpy as np
import matplotlib.pyplot as plt

# 목표 함수 f(x) = x^2
def f(x):
    return x**2

# 미분 (기울기) df(x) = 2x
def df(x):
    return 2*x

# 파라미터 설정
learning_rate = 0.1
x_for = 5  # 초기값 설정 (For loop용)
x_while = 5  # 초기값 설정 (While loop용)
iterations = 20  # 반복 횟수
threshold = 1e-6  # 종료 조건 (While loop)

history_for = []  # For loop 경로 저장
history_while = []  # While loop 경로 저장
 

## While loop 버전
while True:
    gradient = df(x_while)
    new_x = x_while - learning_rate * gradient
    history_while.append(new_x)

    if abs(new_x - x_while) < threshold:  # 종료 조건
        break
    x_while = new_x

## For Loop 버전
for i in range(iterations):
    gradient = df(x_for)
    x_for = x_for - learning_rate * gradient
    history_for.append(x_for)


# 그래프 시각화
x_vals = np.linspace(-6, 6, 100)
y_vals = f(x_vals)

plt.plot(x_vals, y_vals, label="f(x) = x^2")  # 함수 그래프
plt.scatter(history_for, [f(x) for x in history_for], color='red', label="For loop steps")  # For loop 과정
plt.scatter(history_while, [f(x) for x in history_while], color='blue', label="While loop steps")  # While loop 과정
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.title("Gradient Descent Simulation (For loop vs While loop)")
plt.show()

print("최적 x 값 (For loop):", x_for)
print("최적 x 값 (While loop):", x_while)

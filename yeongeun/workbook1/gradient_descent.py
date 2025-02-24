import matplotlib.pyplot as plt

weight = 0
bias = 0
threshold = 0.001
learning_rate = 0.008
epoch = 1000

data = [
    [-2.51, -0.65],
    [4.64, 19.92],
    [-6.88, -15.75],
    [-8.84, -19.85],
    [2.02, 12.34],
    [2.24, 9.86],
    [-4.16, -9.89],
    [-0.88, -1.10],
    [-6.01, -15.81],
    [1.85, 8.93]
]

x, y = zip(*data)
plt.scatter(x, y, color='skyblue')
plt.show()

# 예측 모델
def predict_model(x):
    return weight * x + bias

# mse함수로 오차 구하고 미분한 값 반환
def compute_gradients(x, y):
    pred = predict_model(x)
    real = y

    dw = 2 * (pred - y) * x
    db = 2 * (pred - y)

    return dw, db

for i in range(epoch):
    total_dw, total_db = 0, 0

    for d in data:
        dw, db = compute_gradients(d[0], d[1]) #d[0]=x, d[1]=y
        total_dw += dw
        total_db += db

    avg_dw = total_dw / len(x)
    avg_db = total_db / len(x)

    weight -= learning_rate * avg_dw
    bias -= learning_rate * avg_db

    if abs(avg_dw) < threshold and abs(avg_db) < threshold:
        print(f"epoch {i+1}, 종료")
        break
    if i == epoch - 1:
        print(f"epoch {i+1}, 종료")

print(f"최종 가중치: w = {weight:.4f}, b = {bias:.4f}")

plt.scatter(x, y, color='skyblue')

x_range = list(range(-10, 10))
y_pred = [weight * xi + bias for xi in x_range]

plt.plot(x_range, y_pred, color='red')
plt.show()
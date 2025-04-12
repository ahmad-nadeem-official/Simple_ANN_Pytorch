import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


train = pd.read_csv('/content/train.csv')
test = pd.read_csv('/content/test.csv')

train.head()
test.head()

train['price_range'].unique()
# array([1, 2, 3, 0])

train.info()

train.isnull().sum(), test.isnull().sum()
#no null value

test.describe()

train.duplicated().sum(), test.duplicated().sum()
# (np.int64(0), np.int64(0))

'''visualization'''

plt.figure(figsize=(10,8))
sns.heatmap(train.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

train.hist(figsize=(16, 12), bins=20, color='skyblue', edgecolor='black')
plt.suptitle("Feature Distributions of train data", fontsize=16)
plt.show()

test.hist(figsize=(16, 12), bins=20, color='skyblue', edgecolor='black')
plt.suptitle("Feature Distributions of test data", fontsize=16)
plt.show()


plt.figure(figsize=(18, 6))
sns.boxplot(data=train, palette="Set2")
plt.xticks(rotation=45)
plt.title("Boxplot of Features")
plt.show()

plt.figure(figsize=(18, 6))
sns.boxplot(data=test, palette="Set2")
plt.xticks(rotation=45)
plt.title("Boxplot of Features")
plt.show()

train.shape, test.shape

x = train.drop('price_range', axis=1)
y = train['price_range']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Convert DataFrames to NumPy
x_train_np = x_train.to_numpy()
x_test_np = x_test.to_numpy()
y_train_np = y_train.to_numpy()
y_test_np = y_test.to_numpy()

# Convert NumPy to PyTorch tensors with correct dtype
x_train = torch.from_numpy(x_train_np).float()
x_test = torch.from_numpy(x_test_np).float()
y_train = torch.from_numpy(y_train_np).float()
y_test = torch.from_numpy(y_test_np).float() 

class SimpleANN:
    def __init__(self, x):
        # Weight matrix: [num_features x 20]
        self.weight = torch.randn(x.shape[1], 20, dtype=torch.float32, requires_grad=True)
        
        self.bias = torch.randn(20, dtype=torch.float32, requires_grad=True) 

    def forward(self, x):
        # Linear transformation + sigmoid activation
        z = torch.matmul(x, self.weight) + self.bias
        y_pred = torch.sigmoid(z) 
        return y_pred

    def loss(self, y_pred, y):
        epsilon = 1e-15
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)

        loss = -torch.mean(y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred))
        return loss
    
learning_rate = 0.01
epochs = 10

for epoch in range(epochs):
    model = SimpleANN(x_train)
    y_pred = model.forward(x_train)
    loss = model.loss(y_pred, y_train)
    loss.backward()
    with torch.no_grad():
        model.weight -= learning_rate * model.weight.grad
        model.bias -= learning_rate * model.bias.grad
    model.weight.grad.zero_()
    model.bias.grad.zero_()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")


with torch.no_grad():
    y_pred = model.forward(x_test)
    y_pred = (y_pred > 0.5).float()
    accuracy = (y_pred == y_test).float().mean()
    print(f"Test Accuracy: {accuracy.item()}")    
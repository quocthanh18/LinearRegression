import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
X_train = train.iloc[:, :-1]    # Dataframe (chứa 10 đặc trưng huấn luyện)
y_train = train.iloc[:, -1]     # Series    (chứa 1 giá trị mục tiêu kiểm tra)
X_test = test.iloc[:, :-1]      # Dataframe (chứa 10 đặc trưng kiểm tra)
y_test = test.iloc[:, -1]    
# Set the random seed for reproducibility

# Shuffle the dataset indices
shuffled_indices = np.arange(len(X_train))
np.random.shuffle(shuffled_indices)

# Initialize the k-fold cross-validation object
kfold = KFold(n_splits=5, shuffle=False)  # Note: shuffle=False
for j, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
        print("Fold", j + 1)
        print("Training indices:", train_idx.shape)
        print("Validation indices:", val_idx.shape)
        print()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Assuming you have a DataFrame called 'data' with 'Degree' and 'Salary' columns
# 'Degree' should have values 1 for Bachelor and 2 for Master

# Map degree values to labels
data = pd.read_csv('train.csv')
data['Degree_Label'] = data['Degree'].map({1: 'Bachelor', 2: 'Master'})

# Create a bar plot
plt.figure(figsize=(8, 6))
sns.barplot(x='Degree_Label', y='Salary', data=data)
plt.xlabel('Degree')
plt.ylabel('Salary')
plt.title('Bachelor vs. Master Salary')
plt.show()

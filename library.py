import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('test.csv')

# Group data by degree and calculate average salary
degree_salary = data.groupby('Degree')['Salary'].mean()

# Create a bar plot
plt.figure(figsize=(8, 6))
degree_salary.plot(kind='bar', color='skyblue')
plt.xlabel('Degree')
plt.ylabel('Average Salary')
plt.title('Average Salary by Degree')
plt.xticks(rotation=0)
plt.show()

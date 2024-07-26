# Video Game Market Segmentation
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-%20blue?style=plastic)
![Machine_Learning](https://img.shields.io/badge/Machine%20Learning-%20blue?style=plastic)
![Data_Science](https://img.shields.io/badge/Data%20Science-%20blue?style=plastic)
![License](https://img.shields.io/badge/license%20-%20MIT%20-%20darkblue?style=plastic)
![Pandas](https://img.shields.io/badge/Pandas-2.0.3-%20blue?style=plastic)
![Numpy](https://img.shields.io/badge/Numpy-1.24.4-%20blue?style=plastic)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7.2-%20blue?style=plastic)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12.2-%20blue?style=plastic)
![Build_Status](https://img.shields.io/badge/build-passing-brightgreen)
![Open_Issues](https://img.shields.io/badge/Issues%20-%200%20-%20orange?style=plastic)

## Introduction
This project focuses on market segmentation for video games using a dataset collected via Google Forms and QR codes. The goal is to segment users based on their gaming preferences and behaviors using machine learning techniques.

## Dataset
<p align="center">
  <img src="Famous-Video-Game-Logos.png" alt="Famous-Video-Game-Logos" width="500"/>
</p>

- The dataset used for this project is available directly in the repository.
- This dataset contains 222 records of video game preferences and habits. 

## Column Names

- **Username**: Username
- **Age**: How old are you?
- **Gender**: Kindly select your gender?
- **Games-Lovers**: How much do you like playing video games?
- **Machine-Setup**: Where do you usually play video games? (PC, Laptop, Console, etc.)
- **Game-type**: What type of video games do you like to play?
- **Games**: What games do you play?
- **Play-Games H/W**: How much time in a week do you spend on video games (in hours)?
- **?PlayInWeekend**: Do you play in the weekends?
- **?PlayInBusyTimes**: Do you play video games even if you are very busy?
- **?GameBenefit**: Do you believe video games teach you something?

## Loading Libraries and Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv('Data.csv')
df.head()
```

## Data Cleaning
```pyhon
# Drop unnecessary columns
df = df.drop(columns=["Username", "?PlayInWeekend"], axis=1)

# Rename column for clarity
df = df.rename(columns={'PC | Laptop | Console': 'Machine-Setup'})

# Remove duplicates
df = df.drop_duplicates()

# Fill missing values
play_hours_med = df["Play-Games H/W"].median()
df["Play-Games H/W"] = df["Play-Games H/W"].fillna(play_hours_med)
df = df.dropna()

# Clip values for hours played
df['Play-Games H/W'] = df['Play-Games H/W'].clip(0, 50)
```
## Data Exploration
```pyhon
# Statistical summary
df.describe()

# Visualizations
plt.figure(figsize=(8, 6))
df.boxplot(column=['Play-Games H/W'], grid=False)
plt.title('"Hours Playing" Statistics', fontsize=20)

# Machine setup distribution
pc_count = df["Machine-Setup"].str.contains('PC').sum()
mob_count = df["Machine-Setup"].str.contains('Mobile').sum()
cons_count = df["Machine-Setup"].str.contains('Console').sum()

plt.figure(figsize=(8, 6))
plt.bar(['PC', 'Mobile', 'Console'], [pc_count, mob_count, cons_count])
plt.xlabel("Machine Setup", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.title("Machine Setup Distribution", fontsize=20)

# Play hours mean by gender
gender_play_mean = df.groupby('Gender')['Play-Games H/W'].mean()

plt.figure(figsize=(8, 6))
plt.bar(gender_play_mean.index, gender_play_mean.values)
plt.xlabel("Gender", fontsize=15)
plt.ylabel("Play Hours Mean", fontsize=15)
plt.title("Play Hours Mean by Gender", fontsize=20)
```
## Categorical Data Encoding
```python
# Perform label encoding
df_relabeled = df.copy()
categ_col = ['Age', 'Gender', 'Games-Lovers', 'Machine-Setup', 'Game-type', 'Games', '?PlayInBusyTimes', '?GameBenefit']
ordinal_encoder = OrdinalEncoder()
df_relabeled[categ_col] = ordinal_encoder.fit_transform(df_relabeled[categ_col])
```
## Clustering and PCA
```python
# Determine the number of clusters using the Elbow method
scores = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit_predict(df_relabeled)
    scores.append(kmeans.inertia_)

plt.figure(figsize=(20, 10))
plt.plot(range(2, 10), scores, marker='o')
plt.xlabel("Number of clusters (K)", fontsize=20)
plt.ylabel("Distance", fontsize=20)
plt.title("Elbow Method for Optimal K", fontsize=30)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
df_2D = pca.fit_transform(df_relabeled.values)

plt.figure(figsize=(20, 10))
plt.scatter(df_2D[:, 0], df_2D[:, 1], s=50, color='b')
plt.title("Dataset after Dimensionality Reduction", fontsize=30)

# Clustering with K-means
k = 4
kmeans = KMeans(n_clusters=k)
y_pred = kmeans.fit_predict(df_relabeled)
centers = kmeans.cluster_centers_

# Visualize clusters
centers_2D = pca.transform(centers)
plt.figure(figsize=(20, 10))
plt.scatter(df_2D[:, 0], df_2D[:, 1], s=50, color='b', label='Datapoint')
plt.scatter(centers_2D[:, 0], centers_2D[:, 1], s=150, color='r', label='Cluster Center')
plt.title("Clusters and Centers after Dimensionality Reduction", fontsize=30)
plt.legend(fontsize=15)
```
# Determine the number of clusters using the Elbow method
```python
scores = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit_predict(df_relabeled)
    scores.append(kmeans.inertia_)

plt.figure(figsize=(20, 10))
plt.plot(range(2, 10), scores, marker='o')
plt.xlabel("Number of clusters (K)", fontsize=20)
plt.ylabel("Distance", fontsize=20)
plt.title("Elbow Method for Optimal K", fontsize=30)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
df_2D = pca.fit_transform(df_relabeled.values)

plt.figure(figsize=(20, 10))
plt.scatter(df_2D[:, 0], df_2D[:, 1], s=50, color='b')
plt.title("Dataset after Dimensionality Reduction", fontsize=30)

# Clustering with K-means
k = 4
kmeans = KMeans(n_clusters=k)
y_pred = kmeans.fit_predict(df_relabeled)
centers = kmeans.cluster_centers_

# Visualize clusters
centers_2D = pca.transform(centers)
plt.figure(figsize=(20, 10))
plt.scatter(df_2D[:, 0], df_2D[:, 1], s=50, color='b', label='Datapoint')
plt.scatter(centers_2D[:, 0], centers_2D[:, 1], s=150, color='r', label='Cluster Center')
plt.title("Clusters and Centers after Dimensionality Reduction", fontsize=30)
plt.legend(fontsize=15)
```
## Business Recommendations
- Customized Advertising: Target ads and services to each user segment based on their gaming preferences.
- Product Development: Develop new products tailored to the interests of different segments.
- Customer Recommendations: Suggest products to new customers based on the cluster they belong to.

## License
- This project is licensed under the MIT License.










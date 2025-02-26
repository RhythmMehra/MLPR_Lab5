# MLPR_Lab5
# Distance-Based Classification Algorithms

## 1. Common Distance Metrics
- Euclidean Distance
- Manhattan Distance
- Minkowski Distance
- Mahalanobis Distance
- Cosine Distance
- Chebyshev Distance

## 2. Real-World Applications
- **Medical Diagnosis**: Identifying diseases based on patient symptom similarity.
- **Recommender Systems**: Suggesting products based on user preferences.
- **Fraud Detection**: Identifying fraudulent transactions by comparing them to normal patterns.
- **Image Recognition**: Classifying objects based on feature distances.
- **Customer Segmentation**: Grouping customers with similar buying behaviors.

## 3. Explanation of Various Distance Metrics
- **Euclidean Distance**:  
  \[ d = \sqrt{\sum (x_i - y_i)^2} \]
- **Manhattan Distance**:  
  \[ d = \sum |x_i - y_i| \]
- **Minkowski Distance**:  
  \[ d = \left(\sum |x_i - y_i|^p
ight)^{1/p} \]
- **Cosine Similarity**:  
  \[ \cos(	heta) = rac{A \cdot B}{||A|| \cdot ||B||} \]
- **Mahalanobis Distance**:  
  \[ d = \sqrt{(X - \mu)^T S^{-1} (X - \mu)} \]

## 4. Role of Cross Validation in Model Performance
Cross-validation aids in assessing a model's generalizability by repeatedly dividing data into training and validation sets. It ensures that the model performs effectively on unseen data and prevents overfitting. Common techniques include:

- **k-Fold Cross-Validation**: Splitting the dataset into k parts and training on k-1 while testing on the remaining.
- **Leave-One-Out Cross-Validation (LOO-CV)**: A special case where each sample is tested individually.

## 5. Variance and Bias in Terms of KNN
- **Bias**: A high bias in KNN can occur when the value of k is large.
- **Variance**: The algorithm's sensitivity to fluctuations in the training data.

https://www.kaggle.com/code/rhythmmehra/rhythmlab5
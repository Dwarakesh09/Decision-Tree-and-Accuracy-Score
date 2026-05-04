# Decision Tree and Accuracy Score

A comprehensive Python implementation for building decision tree models, evaluating their performance, and understanding classification metrics.

## 📚 Overview

This repository contains a complete guide and implementation for:
- **Decision Tree Classification**: Build trees from scratch
- **Accuracy Evaluation**: Measure model performance accurately
- **Performance Metrics**: Comprehensive evaluation tools
- **Practical Examples**: Real-world use cases and demonstrations

Decision trees are one of the most interpretable machine learning algorithms, making them ideal for both beginners and experienced practitioners.

## 🎯 Key Features

✅ **Decision Tree Implementation**
- Pure Python implementation from scratch
- Support for both binary and multi-class classification
- Customizable splitting criteria (Gini Impurity, Information Gain/Entropy)
- Hyperparameter tuning options

✅ **Accuracy Evaluation**
- Accurate accuracy score calculation
- Confusion matrix generation
- Precision, recall, and F1-score metrics
- Cross-validation support

✅ **Educational Value**
- Well-documented code
- Step-by-step algorithm explanation
- Multiple example use cases
- Detailed comments throughout

## 🚀 Getting Started

### Prerequisites

- **Python** 3.6 or higher
- **numpy**: Numerical computations
- **pandas**: Data manipulation and analysis
- **matplotlib**: Visualization (optional)
- **scikit-learn**: For comparison and validation (optional)

### Installation

```bash
# Clone the repository
git clone https://github.com/Dwarakesh09/Decision-Tree-and-Accuracy-Score.git
cd Decision-Tree-and-Accuracy-Score

# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install numpy pandas matplotlib scikit-learn
```

## 📖 Quick Start

### Basic Usage

```python
from decision_tree import DecisionTree
from accuracy_score import accuracy_score

# Load your data
X_train, y_train = load_training_data()
X_test, y_test = load_test_data()

# Create and train the decision tree
tree = DecisionTree(max_depth=5, criterion='gini')
tree.fit(X_train, y_train)

# Make predictions
predictions = tree.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.4f}")
```

### Advanced Usage

```python
from decision_tree import DecisionTree
from sklearn.model_selection import cross_val_score
from metrics import precision_score, recall_score, f1_score

# Train with custom parameters
tree = DecisionTree(
    max_depth=10,
    min_samples_split=5,
    criterion='entropy'
)
tree.fit(X_train, y_train)

# Make predictions
y_pred = tree.predict(X_test)

# Evaluate with multiple metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
```

## 📁 Project Structure

```
Decision-Tree-and-Accuracy-Score/
├── README.md                      # This file
├── requirements.txt               # Project dependencies
├── decision_tree.py              # Main decision tree implementation
├── accuracy_score.py             # Accuracy calculation functions
├── metrics.py                    # Additional evaluation metrics
├── utils.py                      # Helper functions
├── examples/
│   ├── basic_classification.py   # Simple example
│   ├── iris_dataset.py           # Iris dataset example
│   ├── performance_evaluation.py # Evaluation metrics example
│   └── visualization.py          # Tree visualization example
└── data/
    └── sample_data.csv           # Sample dataset (if included)
```

## 🔍 Algorithm Details

### Decision Tree Algorithm

The implementation uses the **Recursive Partitioning** approach:

1. **Feature Selection**
   - Evaluates all features to find the best split
   - Uses Gini Impurity or Information Gain (Entropy) as criteria
   - Selects feature that maximizes information gain

2. **Data Splitting**
   - Partitions data based on selected feature threshold
   - Creates left and right child nodes
   - Recursively applies to each subset

3. **Stopping Criteria**
   - Maximum tree depth reached
   - Minimum samples to split threshold met
   - Pure node (all samples same class)
   - No improvement in information gain

4. **Leaf Node Classification**
   - Assigns most frequent class label in leaf
   - Stores class probabilities for uncertainty estimation

### Accuracy Score Calculation

```
Accuracy = (Correct Predictions) / (Total Predictions) × 100%
```

**Formula:**
```
Accuracy = TP + TN / (TP + TN + FP + FN)
```

Where:
- **TP** (True Positives): Correctly predicted positive instances
- **TN** (True Negatives): Correctly predicted negative instances
- **FP** (False Positives): Incorrectly predicted as positive
- **FN** (False Negatives): Incorrectly predicted as negative

## ⚙️ Parameters

### DecisionTree Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_depth` | int | None | Maximum tree depth (None = unlimited) |
| `min_samples_split` | int | 2 | Minimum samples required to split node |
| `min_samples_leaf` | int | 1 | Minimum samples required at leaf node |
| `criterion` | str | 'gini' | Split criterion ('gini' or 'entropy') |
| `random_state` | int | None | Random seed for reproducibility |

## 📊 Performance Characteristics

| Metric | Value |
|--------|-------|
| **Training Time Complexity** | O(n log n) average, O(n²) worst case |
| **Space Complexity** | O(n) |
| **Prediction Time** | O(log n) balanced, O(n) worst case |
| **Memory Usage** | Moderate |

### Strengths
✅ Highly interpretable and easy to visualize
✅ Requires minimal data preprocessing
✅ Handles both categorical and numerical data
✅ No scaling required
✅ Fast predictions

### Limitations
❌ Prone to overfitting (especially without constraints)
❌ Biased towards high-cardinality features
❌ Unstable with small data changes
❌ Poor performance on linearly separable data

## 💡 Use Cases

- **Medical Diagnosis**: Patient classification based on symptoms
- **Credit Approval**: Loan eligibility decisions
- **Customer Segmentation**: Dividing customers into categories
- **Spam Detection**: Email classification
- **Fraud Detection**: Transaction analysis
- **Game AI**: Decision making in games

## 📈 Examples

### Example 1: Iris Dataset

```python
from sklearn.datasets import load_iris
from decision_tree import DecisionTree
from accuracy_score import accuracy_score

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train model
model = DecisionTree(max_depth=5)
model.fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Iris Dataset Accuracy: {accuracy:.4f}")
```

### Example 2: Cross-Validation

```python
from sklearn.model_selection import cross_val_score
from decision_tree import DecisionTree

model = DecisionTree(max_depth=5)
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean():.4f}")
```

## 🤝 Contributing

Contributions are welcome! To contribute:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Contribution Ideas
- Add additional evaluation metrics
- Implement pruning algorithms
- Create visualization tools
- Add more example datasets
- Improve documentation
- Performance optimizations

## 📝 License

This project is open-source and available under the **MIT License**. See the LICENSE file for details.

## 👨‍💻 Author

**Dwarakesh09**
- GitHub: [@Dwarakesh09](https://github.com/Dwarakesh09)

## 🙏 Acknowledgments

- Inspired by classical machine learning decision tree algorithms
- References: C4.5, ID3, and CART algorithms
- Built with foundational ML principles
- Community feedback and contributions

## 📞 Support & Contact

For questions, issues, or suggestions:

- **Open an Issue**: [GitHub Issues](https://github.com/Dwarakesh09/Decision-Tree-and-Accuracy-Score/issues)
- **Create a Discussion**: [GitHub Discussions](https://github.com/Dwarakesh09/Decision-Tree-and-Accuracy-Score/discussions)
- **Email**: Check your GitHub profile for contact info

## 🔗 Useful Resources

- [Decision Trees Explained](https://scikit-learn.org/stable/modules/tree.html)
- [Information Gain & Entropy](https://en.wikipedia.org/wiki/Information_gain_(decision_tree))
- [Gini Impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity)
- [Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Overfitting Prevention](https://en.wikipedia.org/wiki/Overfitting)

## 🎓 Learning Path

1. **Understand**: Read the algorithm explanation
2. **Explore**: Check example notebooks
3. **Experiment**: Modify parameters and observe results
4. **Build**: Create your own classification projects
5. **Contribute**: Share improvements with the community

---

**Last Updated**: May 2026

**Status**: ✅ Active Development

**Maintained by**: [@Dwarakesh09](https://github.com/Dwarakesh09)

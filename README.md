# Decision Tree and Accuracy Score

A Python implementation for building decision trees and evaluating their accuracy on classification tasks.

## Overview

This project provides tools for:
- Building decision tree models from scratch
- Training on classification datasets
- Computing accuracy scores and performance metrics
- Visualizing decision boundaries and tree structures

## Features

✨ **Core Features**
- Decision Tree Classification implementation
- Accuracy Score calculation
- Support for both binary and multi-class classification
- Easy-to-use API for model training and prediction

## Installation

### Prerequisites
- Python 3.6 or higher
- Required libraries: numpy, pandas, scikit-learn (optional for comparison)

### Setup

```bash
# Clone the repository
git clone https://github.com/Dwarakesh09/Decision-Tree-and-Accuracy-Score.git

# Navigate to the project directory
cd Decision-Tree-and-Accuracy-Score

# Install dependencies (if applicable)
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from decision_tree import DecisionTree
from accuracy_score import accuracy_score

# Load your data
X_train, y_train = load_training_data()
X_test, y_test = load_test_data()

# Create and train the decision tree
tree = DecisionTree()
tree.fit(X_train, y_train)

# Make predictions
predictions = tree.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.4f}")
```

## Project Structure

```
Decision-Tree-and-Accuracy-Score/
├── README.md
├── requirements.txt
├── decision_tree.py          # Main decision tree implementation
├── accuracy_score.py         # Accuracy scoring functions
└── examples/
    └── sample_usage.py       # Example usage scripts
```

## Algorithm Details

### Decision Tree
The implementation uses a recursive approach to build the tree:
1. **Feature Selection**: Chooses the best feature to split on (based on information gain or Gini impurity)
2. **Splitting**: Recursively partitions the data
3. **Stopping Criteria**: Halts when reaching pure nodes or maximum depth
4. **Leaf Assignment**: Assigns class labels to leaf nodes

### Accuracy Score
Calculates the percentage of correct predictions:
```
Accuracy = (Number of Correct Predictions) / (Total Number of Predictions)
```

## Parameters

Key parameters for DecisionTree:
- `max_depth`: Maximum depth of the tree (default: None)
- `min_samples_split`: Minimum samples required to split a node (default: 2)
- `criterion`: Splitting criterion - 'gini' or 'entropy' (default: 'gini')

## Performance

Typical performance metrics:
- Training time: O(n log n) average case
- Space complexity: O(n)
- Prediction time: O(log n) for balanced trees

## Examples

Check the `examples/` directory for sample notebooks and scripts demonstrating:
- Basic classification
- Performance evaluation
- Cross-validation
- Hyperparameter tuning

## Contributing

Contributions are welcome! Please feel free to:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is open source and available under the MIT License. See LICENSE file for details.

## Author

**Dwarakesh09**

## Acknowledgments

- Inspired by classical machine learning decision tree algorithms
- Built with reference to foundational ML principles

## Contact & Support

For questions or issues, please open an issue on the [GitHub repository](https://github.com/Dwarakesh09/Decision-Tree-and-Accuracy-Score).

---

**Last Updated**: May 2026

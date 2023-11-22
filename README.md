This is a machine learning library developed by Pablo Sauma for CS5350/6350 in University of Utah.

# Usage

## Execution

For the purpose on the assignments, please execute the `bash run.sh` available in the root directory, I will update it for each assignement and all the pathing should work.

## Generalities

### Data Descriptor

Some of the following structures require what is called a "data descriptor" which contain key-value pairs that provide the structure with information on the data to process. Each of the structures specify which of these descriptor attributes it requires, if any. The descriptor is a dictionary which may contain values stored in the following keys:

1. `target`: Name of the column of the output values in the data.
2. `columns`: List of column names in their respective order for the introduced data.
3. `categorical`: Set of column names which contain categorical data.
4. `numerical`: Set of column names which contain numerical data.
5. `weight`: Name of the column for weights if the data samples are weighted.

### Call

All structures perform their function in two ways:

1. Construction: The object is created with the specified parameters.
2. Callable: The object interacts with new data (e.g: data to be predicted) through a call to the object (e.g: `object(params...)`).

## Preprocessing

### CSVLoader

This preprocessor method loads the data from a CSV file. Returns a matrix-like object with the data.

**Parameters:**

1. `path`: String with the path of the file to load.
2. `descriptor`: (Optional) Specifies a descriptor for the data, numerical data will be loaded as float values.
3. `separator`: (Optional) Specifies the separator used in the file to separate the data. `,` by default.

### Replace Missing With Majority

This preprocessor takes a dataset and replaces its missing values with those of the majority attribute for data samples that share its label (for labeled data) or just the majority attribute for that column (for unlabeled data).

**Constructor parameters:**

1. `data`: Original data to use to construct the preprocessor. Data is not modified, just used as reference for future transformations.
2. `descriptor`: Must contain at least `target`, `columns` and `categorical`. Optional: `weight`. 
3. `missing`: Value representing the missing value in the data.
4. `columns`: (Optional) List which specifies only certain columns to which to apply the preprocessing. If left empty then the processor will apply to all categorical columns.

**Call:**

The preprocessor is called with a matrix-like `data` parameter which follows the original `descriptor` specifications and a secondary `labeled` boolean parameter to specify if the data is labeled or not (`False` by default).

### DiscretizeNumericalAtMedian

This preprocessor takes a dataset and discritizes the numerical values within it into `<=` and `>` with respect to the median.

**Constructor parameters:**

1. `data`: Original data to use to construct the preprocessor. Data is not modified, just used as reference for future transformations.
2. `descriptor`: Must contain at least `columns` and `numerical`. Optional: `weight`.
3. `columns`: (Optional) List which specifies only certain columns to which to apply the preprocessing. If left empty then the processor will apply to all numerical columns.

## Postprocessing

### Metrics

`get_y(x, descriptor)`: Extracts the `target` columns from the dataset `x` into a single array.

`accuracy(y, prediction)`: Calculates the accuracy of prediction for a given target array `y` and a prediction array `prediction`.

`square_error(y, prediction)`: Calculates the square error of the predictions for a given target array `y` and a prediction array `prediction`.

## Decision Trees

### ID3

Allows the classification of data through the construction of a tree. 

**Constructor parameters:**

1. `data`: Original data to use to construct the tree, this data must have passed through the appropriate pre-processing.
2. `descriptor`: Must contain at least `target`, `columns`, `categorical` and `numerical`. Optional: `weight`. The descriptor must have passed through the appropriate pre-processing.
3. `criterion`: (Optional) May be `information_gain`/`entropy` (default), `gini_index`/`gini` or `majority_error`/`majority`.
4. `max_depth`: (Optional, default: 0) Specified the maximum depth of the tree, if zero then it does not has a maximum depth.
5. `preprocess`: (Optional) List of preprocessor objects to be applied to the data before prediction in this tree. They will be applied in the given order, only the data parameter is sent to the call so preprocessors must assume unlabeled data.
6. `subsample_attributes`: (Optional, default: 0) Number of attributes to subsample before deciding the best split in each node. Useful for Random Forest. If 0 then it does not perform subsampling.
7. `seed`: (Optional, default: None) Seed to utilize for the randomness of the subsampling of attributes. Allows reproducibility of experiments.

**Call:**

The decision tree is called with a matrix-like `data` parameter of samples to be classified. The function returns an array with a label for each of the samples provided in their respective order.

## Ensemble Learning

### AdaBoost

Allows the classification of data through ensamble learning with AdaBoost.

**Constructor parameters:**

1. `data`: Original data to use to construct the tree, this data must have passed through the appropriate pre-processing.
2. `descriptor`: Must contain at least `target`, `columns`, `categorical` and `numerical`.
3. `T`: Number of decision trees to use as part of this ensamble.
4. `treeCls`: (Optional, default: ID3) Class of the trees to use as part of the ensamble.
5 `**kwargs`: (Optional) Additional parameters to send to the trees of the ensamble when constructing them.

**Call:**

The ensamble is called with a matrix-like `data` parameter of samples to be classified. The function returns an array with a label for each of the samples provided in their respective order.

### Bagging

Allows the classification of data through ensamble learning with Bagging.

**Constructor parameters:**

1. `data`: Original data to use to construct the tree, this data must have passed through the appropriate pre-processing.
2. `descriptor`: Must contain at least `target`, `columns`, `categorical` and `numerical`.
3. `T`: Number of decision trees to use as part of this ensamble.
4. `m`: Number of data samples to sample (with replacement) from the dataset for each of the trees in the ensamble.
5. `seed`: (Optional, default: None) Seed value for the randomness used to sample from the dataset, allows reproducibility of experiments.
6. `treeCls`: (Optional, default: ID3) Class of the trees to use as part of the ensamble.
7 `**kwargs`: (Optional) Additional parameters to send to the trees of the ensamble when constructing them.

**Call:**

The ensamble is called with a matrix-like `data` parameter of samples to be classified. The function returns an array with a label for each of the samples provided in their respective order.

### Random Forest

Allows the classification of data through ensamble learning with Random Forest.

**Constructor parameters:**

1. `data`: Original data to use to construct the tree, this data must have passed through the appropriate pre-processing.
2. `descriptor`: Must contain at least `target`, `columns`, `categorical` and `numerical`.
3. `T`: Number of decision trees to use as part of this ensamble.
4. `subsample_attributes`: Number of attributes/feaures to subsample on each node split of the tree construction.
5. `m`: Number of data samples to sample (with replacement) from the dataset for each of the trees in the ensamble.
6. `seed`: (Optional, default: None) Seed value for the randomness used to sample from the dataset, allows reproducibility of experiments.
7. `treeCls`: (Optional, default: ID3) Class of the trees to use as part of the ensamble.
8 `**kwargs`: (Optional) Additional parameters to send to the trees of the ensamble when constructing them.

**Call:**

The ensamble is called with a matrix-like `data` parameter of samples to be classified. The function returns an array with a label for each of the samples provided in their respective order.

## Linear Regression

### LMS

Allows performing linear regression on data.

**Constructor parameters:**

1. `data`: Original data to use to construct the regression, this data must have passed through the appropriate pre-processing.
2. `descriptor`: Must contain at least `target`, `columns` and `numerical`. If `numerical` is missing, all the columns will be assumed as numerical. The descriptor must have passed through the appropriate pre-processing.
3. `lr`: Learning rate to utilize during the regression.
4. `max_iters`: (Optional, default: 0) Maximum numbers of iterations to perform for weight adjustment. If 0 then it does not has a maximum number of iterations.
5. `threshold`: (Optional, default: 1e-6) Threshold for the minimum change of the weights under which the algorithm stops as it is considered to have converged.
6. `strategy`: (Optional, default: batch) Specifies whether to use ``batch'' or ``stochastic'' gradient descent.
7. `seed`: (Optional, default: None) Allows the specification of a randomness seed for reproducibility when using stochastic gradient descent.

**Call:**

The LMS is called with a matrix-like `data` parameter of samples to be regressed. The function returns an array with the calculated value for each of the samples provided in their respective order.

## Perceptron

### Perceptron

Allows performing the standard perceptron algorithm on data.

**Constructor parameters:**

1. `data`: Original data to use to construct the perceptron, this data must have passed through the appropriate pre-processing. Target data is expected to be either 1 or -1.
2. `descriptor`: Must contain at least `target`, `columns` and `numerical`. If `numerical` is missing, all the columns will be assumed as numerical.
3. `lr`: (Optional, default: 1) Learning rate to utilize during the regression. 
4. `max_iters`: (Optional, default: 10) Maximum numbers of iterations to perform for weight adjustment.
5. `seed`: (Optional, default: None) Allows the specification of a randomness seed for reproducibility.

**Call:**

The Perceptron is called with a matrix-like `data` parameter of samples to be classified. The function returns an array with the calculated value for each of the samples provided in their respective order.

### Voted Perceptron

Allows performing the voted perceptron algorithm on data.

**Constructor parameters:**

1. `data`: Original data to use to construct the perceptron, this data must have passed through the appropriate pre-processing. Target data is expected to be either 1 or -1.
2. `descriptor`: Must contain at least `target`, `columns` and `numerical`. If `numerical` is missing, all the columns will be assumed as numerical.
3. `lr`: (Optional, default: 1) Learning rate to utilize during the regression. 
4. `max_iters`: (Optional, default: 10) Maximum numbers of iterations to perform for weight adjustment.

**Call:**

The Voted Perceptron is called with a matrix-like `data` parameter of samples to be classified. The function returns an array with the calculated value for each of the samples provided in their respective order.

### Average Perceptron

Allows performing the average perceptron algorithm on data.

**Constructor parameters:**

1. `data`: Original data to use to construct the perceptron, this data must have passed through the appropriate pre-processing. Target data is expected to be either 1 or -1.
2. `descriptor`: Must contain at least `target`, `columns` and `numerical`. If `numerical` is missing, all the columns will be assumed as numerical.
3. `lr`: (Optional, default: 1) Learning rate to utilize during the regression. 
4. `max_iters`: (Optional, default: 10) Maximum numbers of iterations to perform for weight adjustment.

**Call:**

The Average Perceptron is called with a matrix-like `data` parameter of samples to be classified. The function returns an array with the calculated value for each of the samples provided in their respective order.

## SVM

### SVMSGD

Allows performing a support vector machine through stochastic gradient descent on data.

** Constructor parameters:**

1. `data`: Original data to use to construct the perceptron, this data must have passed through the appropriate pre-processing. Target data is expected to be either 1 or -1.
2. `descriptor`: Must contain at least `target`, `columns` and `numerical`. If `numerical` is missing, all the columns will be assumed as numerical.
3. `C`: balance constant between slack variables and the weights of the vector.
4. `lr`: (Optional, default: 1) Learning rate to utilize during the regression.
5. `lr_function`: (Optional, default: identity) Learning rate schedule to utilize, can use identity o decay functions.
6. `max_iters`: (Optional, default: 10) Maximum numbers of iterations to perform for weight adjustment.
7. `seed`: (Optional, default: None) Allows the specification of a randomness seed for reproducibility.
8. `kwargs`: (Optional) Any additional parameters to send to the learning rate schedule function, if decay could include `gamma` and `a`.

**Call:**

The SVMSGD is called with a matrix-like `data` parameter of samples to be classified. The function returns an array with the calculated value for each of the samples provided in their respective order.

### SVM

Allows performing a support vector machine through dual form on data.

** Constructor parameters:**

1. `data`: Original data to use to construct the perceptron, this data must have passed through the appropriate pre-processing. Target data is expected to be either 1 or -1.
2. `descriptor`: Must contain at least `target`, `columns` and `numerical`. If `numerical` is missing, all the columns will be assumed as numerical.
3. `C`: balance constant between slack variables and the weights of the vector.
4. `kernel`: (Optional, default: `none`) Kernel function to utilize as part of the SVM, can be `none` or `gaussian`.
5. `kwargs`: (Optional) Any additional parameters to send to the kernel function, if gaussian then should include `gamma`.

**Call:**

The SVM is called with a matrix-like `data` parameter of samples to be classified. The function returns an array with the calculated value for each of the samples provided in their respective order.

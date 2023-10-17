This is a machine learning library developed by Pablo Sauma for CS5350/6350 in University of Utah.

# Usage

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

### Replace Missing With Majority

This preprocessor takes a dataset and replaces its missing values with those of the majority attribute for data samples that share its label (for labeled data) or just the majority attribute for that column (for unlabeled data).

**Constructor parameters:**

1. `data`: Original data to use to construct the preprocessor.
2. `descriptor`: Must contain at least `target`, `columns` and `categorical`. Optional: `weight`. 
3. `missing`: Value representing the missing value in the data.
4. `columns`: (Optional) List which specifies only certain columns to which to apply the preprocessing. If left empty then the processor will apply to all categorical columns.

**Call:**

The preprocessor is called with a matrix-like `data` parameter which follows the original `descriptor` specifications and a secondary `labeled` boolean parameter to specify if the data is labeled or not.

## Decision Trees

### ID3

Allows the classification of data through the construction of a tree. 

**Constructor parameters:**

1. `data`: Original data to use to construct the tree.
2. `descriptor`: Must contain at least `target`, `columns`, `categorical` and `numerical`. Optioanl: `weight`.
3. `criterion`: (Optional) May be `information_gain`/`entropy` (default), `gini_index`/`gini` or `majority_error`/`majority`.
4. `max_depth`: (Optional) Specified the maximum depth of the tree, if zero then it does not has a maximum depth (0 by default).
5. `preprocess`: (Optional) List of preprocessor objects to be applied to the data before construction and prediction using this tree.

**Call:**

The decision tree is called with a matrix-like `data` parameter of samples to be classified. The function returns an array with a label for each of the samples provided in their respective order.





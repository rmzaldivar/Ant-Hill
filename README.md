# Ant Hill

Ant Hill is a novel online algorithm that combines the custom-convolutional layers from the Hyena Hierarchy with the spatial skew Gaussian process to create a powerful and flexible online learning model. By using the custom-convolutional layers as the covariance function for the spatial skew Gaussian process, Ant Hill benefits from the optimization and learning capabilities of both the Hyena Hierarchy and the skew-Gaussian processes. This integration allows Ant Hill to adapt to various spatial data with non-stationary and non-Gaussian distributions while being computationally efficient for large-scale applications.

## Ant Hill model

### Input

Spatial data `X = {x_1, x_2, ..., x_n}` and corresponding output `Y = {y_1, y_2, ..., y_n}`.

### Preprocessing

Preprocess the input data using techniques such as normalization or standardization, as needed.

### Covariance function

Utilize the custom-convolutional layers from the Hyena Hierarchy as the covariance function in the spatial skew Gaussian process:

K(x_p, x_q) = ∑_{i=1}^M λ_i H_i(x_p)⋅H_i(x_q)

where `M` is the number of custom-convolutional layers in the Hyena Hierarchy, `x_p` and `x_q` are two spatial data points, and `λ_i` are positive coefficients.

### Model training

Train the Ant Hill model using online Bayesian optimization, which updates the model sequentially with new data points, leveraging the spatial skew Gaussian process for effective adaptation to non-stationary and non-Gaussian spatial distributions.

1. Initialize the model with a random subset of the data.
2. For each incoming data point `(x_t, y_t)`:
    1. Update the covariance matrix using the new data point and the covariance function `K(·, ·)`.
    2. Compute the posterior distribution of the model with respect to the new data point.
    3. Update the model's hyperparameters by maximizing the marginal likelihood of the data.
    4. Continue to the next data point.

### Inference

For new spatial data points, use the Ant Hill model's posterior distribution to make predictions and provide uncertainty estimates.

### Evaluation

Evaluate the performance of the Ant Hill model using appropriate metrics such as the mean squared error (MSE), mean absolute error (MAE), or the coefficient of determination (R^2).

Ant Hill's online learning capability makes it well-suited for large-scale applications where spatial data arrives sequentially, and the need for model adaptation is crucial. By combining the advantages of the custom-convolutional layers from the Hyena Hierarchy and the spatial skew Gaussian process, Ant Hill offers a powerful and flexible solution for a wide range of spatial modeling tasks.

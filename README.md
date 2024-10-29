# Optimization Package (EGL, CMA, and IGL)

This package provides implementations for Enhanced Gradient Learning (EGL), Covariance Matrix Adaptation (CMA), and Iterative Gradient Learning (IGL) algorithms. It includes modules for datasets, distributions, trust regions, normalizers, stopping conditions, and customizable parameters. The `minimize` function initiates optimization with flexible configurations.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Running EGL, CMA, and IGL](#running-egl-cma-igl)
  - [Parameter Configurations](#parameter-configurations)
- [Components](#components)
  - [Datasets](#datasets)
  - [Distributions](#distributions)
  - [Trust Regions](#trust-regions)
  - [Normalizers](#normalizers)
  - [Callbacks](#callbacks)
  - [Stop Conditions](#stop-conditions)
- [Advanced Usage](#advanced-usage)
  - [Using the Train Function](#using-the-train-function)
  - [Using the Minimize Function](#using-the-minimize-function)
- [Examples](#examples)

---

## Installation

```bash
pip install egl
```

## Usage

### Importing

The package supports the execution of EGL, CMA, and IGL algorithms, each configured with unique datasets, distributions, trust regions, and loss functions.

```python
from egl import EGL, CMA, IGL, minimize
```

### Using the Minimize Function

The `minimize` function provides an api (that matches scipy's) , with optimized configurations.

```python
result = minimize(
    function=objective_function,
    x0=starting_point,
    args=[use second order optimization, use weighted mean gradient],
    bounds=[(lower_bound, upper_bound)],
    callback=callback_function,
)
```

## Components

### Datasets

EGL manages a dataset of tuples based on dataset of sampled points.
There are many ways to decide how to couple the points, we provided multiple datasets options and of course you can create your own.
- **`TuplesDataset`**: Utilizes paired data for optimization with enhanced generalization.
- **`PairsInRangeDataset`**: Pair each point with every single point that was sampled in the same epoch. 
- **`PairsInEpsRangeDataset`**: Pair every point with every other point within a given epsilon euclidean distance.
- **`PairFromDistributionDataset`**: A wrapper to sample points from distribution based on the points' values.

### Distributions

We showed that egl improves results when calculating a weighted mean gradient based on the functions values.
The `WeightsDistributionBase` defines a base class to create weights distributions.

- **SigmoidWeights**: Applies a sigmoid function with custom gamma and quantile values to distribute weights.
- **QuantileWeights**: Take certain quantile of minimum values, all other values are weight 0.

### Trust Regions
We implemented two trust region with different normalizations. 
- **`TanhTrustRegion`**: Maps values within a hyperbolic tangent boundary.
- **`LinearTrustRegion`**: Utilizes linear scaling for boundary constraints.

We found that genetic algorithms suffer when using the hyperbolic function to normalize the values.

### Normalizers
We created `AdaptedOutputUnconstrainedMapping` to normalize the function value to ease the convergence of the algorithm.
This is only for convergence base algorithm (i.e. EGL, IGL).
The normalization is adaptable based on the the observed sampled values.
You can modify the adaptation rate and the outliers' value threshold in the class's constructor.

### Callbacks

Define callbacks to execute custom actions at various stages by create a callback `AlgorithmCallbackHandler.
This class implements 4 methods:
- **on_algorithm_start**: Called once when the algorithm initializes.
- **on_epoch_end**: Execute actions at the end of each epoch (after each function step of the algorithm).
- **on_algorithm_update**: Execute for each trust region update.
- **on_algorithm_end**: Execute actions at the end of the algorithm.

You can create custom callback handlers and pass them to the `train` in the `callback_handlers` parameter.

```
egl.train(
    ...
    callback_handlers=[<callback_handlers>]
)
```
### Stop Conditions

Define custom stopping conditions to end algorithm iterations under specific criteria:
- **AlgorithmStopCondition**: Implement a condition by overriding the `should_stop` method to define custom end conditions.
- Create instance of the class and pass it to the `train` in the `stopping_conditions` parameter.

## Advanced Usage

### Using the Train Function

The `train` method is essential to start training the EGL, CMA, and IGL models. Its parameters allow for fine-grained control over the exploration process, shrinking strategy, and improvement checks.

```python
egl.train(
    epochs=100,
    exploration_size=50,
    num_loop_without_improvement=5,
    min_iteration_before_shrink=10,
    surrogate_model_training_epochs=60,
    warmup_minibatch=5,
    warmup_loops=6,
    stopping_conditions=[<stop_conditions>],
    callback_handlers=[<callback_handlers>]
)
```

#### Key Parameters

- **epochs**: The total number of epochs for the training.
- **exploration_size**: Number of new points explored each epoch.
- **num_loop_without_improvement**: The allowed epochs without improvement before shrinking the trust region.
- **min_iteration_before_shrink**: Minimum epochs before the first shrink action.
- **surrogate_model_training_epochs** (default `60`): Epochs to train the surrogate model.
- **warmup_minibatch** and **warmup_loops** (defaults `5` and `6`): Initialize the model with batch samples and warmup loops.
- **stopping_conditions**: List of stop conditions for early termination.
- **callback_handlers**: List of callback handlers for custom actions post-epoch and during updates.

### Examples

#### Running CMA with Sigmoid Weights and Linear Trust Region

```python
from egl import CMA, SigmoidWeights, LinearTrustRegion

# Set up CMA with Sigmoid distribution and Linear trust region
cma = CMA(
    trust_region=LinearTrustRegion(...),
    distribution=SigmoidWeights(gamma=-10, quantile=80),
    ...
)
result = cma.train(epochs=100, ...)
```

#### Running EGL with Pairs Dataset and Tanh Trust Region

```python
from egl import EGL, PairsInRangeDataset, TanhTrustRegion

# Set up EGL with PairsInRangeDataset and Tanh trust region
egl = EGL(
    dataset_type=PairsInRangeDataset(...),
    trust_region=TanhTrustRegion(...),
    ...
)
result = egl.train(epochs=100, ...)
```


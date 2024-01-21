### BreadcrumbsDeep-vs-Shallow-Neural-Network-discussion
 
 ***As we may know Deep networks can be adapted to such prior information while the Shallow one can not. Let's actually see the results in different scenarios in this post.***\
***Inspired by the work https://ojs.aaai.org/index.php/AAAI/article/view/10913 and the opportunity to expose this problem from my Professor in Predictive Analytics class.***

*Let's start*



##### Set UP
```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
```

##### Generate simulated data
```python
n_samples = 120000
X = np.random.uniform(-2 * pi, 2 * pi, n_samples)
Y = 2 * ((2 * np.cos(X)**2 - 1)**2) - 1
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)
```


##### Function to create a neural network model - Model architecture 

```python
def create_model(layers, units):
    model = Sequential()
    model.add(Dense(units, input_shape=(1,), activation='relu'))
    for _ in range(layers - 1):
        model.add(Dense(units, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9), loss='mean_squared_error')
    return model



# Define configurations
configs = {
    '1_layer': [24, 48, 72, 128, 256],
    '2_layers': [12, 24, 36],
    '3_layers': [8, 16, 24]
}
```

##### Results and discussion 

```python
# Store MSE and number of parameters for each configuration
performance = {}

# Train and evaluate models for each configuration
for name, units_list in configs.items():
    mse_list = []
    param_list = []
    for units in units_list:
        layers = int(name.split('_')[0])
        model = create_model(layers, units)
        model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=0)
        mse = model.evaluate(X_test, Y_test, verbose=0)
        mse_list.append(mse)
        params = model.count_params()  # Count the total number of parameters
        param_list.append(params)
    performance[name] = {'mse': mse_list, 'params': param_list}


# Plot Number of Units vs MSE
plt.figure(figsize=(12, 5))
for name, data in performance.items():
    layers = int(name.split('_')[0])
    plt.plot(configs[name], data['mse'], label=f'{layers} hidden layer{"s" if layers > 1 else ""}')
plt.xlabel('Number of Units')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs. Number of Neurons in Hidden Layers')
plt.legend()
plt.show()

# Plot Number of Parameters vs MSE
plt.figure(figsize=(12, 5))
for name, data in performance.items():
    layers = int(name.split('_')[0])
    plt.plot(data['params'], data['mse'], label=f'{layers} hidden layer{"s" if layers > 1 else ""}')
plt.xlabel('Number of Parameters')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs. Number of Parameters in Neural Network')
plt.legend()
plt.show()
```
<img width="560" alt="Screenshot 2024-01-21 at 15 07 41" src="https://github.com/trungle14/trungle14.github.io/assets/143222481/04d6ac15-b210-4cb6-a19d-9e79f1b517b8">


We can alwaays see the deep NN always performance better than the shallow one although we have the same total number of neuron and other parameter.

**Parameter Efficiency:** Even with the same total number of neurons, a deep architecture might use its parameters more efficiently. By distributing neurons across multiple layers, a DNN can facilitate a more intricate partitioning of the input space, leading to better generalization from the data. Shallow networks might not partition the input space as effectively, potentially requiring more neurons to achieve similar performance.\
**Generalization to New Data:** Deep networks tend to generalize better to new, unseen data. Their layered structure allows them to extract and abstract features in a way that is more robust to variations in input data.
However, it's important to note that deeper networks are not universally better. They require more computational resources and data to train effectively, and they can be more prone to overfitting, especially in scenarios with limited training data. In some simpler tasks, or when interpretability is key, a shallower network might be preferable.

# Predictive Maintenance of Turbofan Engines with PySyft

<p align="center">
<img src="images/turbofan.png" alt="Graphic of a turbofan engine. Source: The Guardian." width="700">
  <br /><em>Image source: <a href="https://www.theguardian.com/preparing-for-9-billion/ng-interactive/2017/sep/27/reinventing-jet-engine-airplanes-technology-turbofan-engine">The Guardian</a>.</em>
</p>

This repository contains code for using Turbofan dataset and PySyft to generate predictions of engine failure based on models trained using Federated Learning (FL).

Based on good work of the [Turbofan Federated Learning POC by matthiaslau](https://github.com/matthiaslau/Turbofan-Federated-Learning-POC), I bring up to date Federated Learning on the Turbofan simulated engine dataset for prediction of engines' Remaining Useful Life. Methods like this help turbofan engine manufacturers better plan maintenance schedules without sacrificing competitiveness.

PySyft has gone through substantial refactoring in the 5 months since its last commit. This repository attempts to bring the Turbofan POC up-to-date, and introduce experimental results on the combinatorial effects of FL and: 

* Differential privacy, added at different levels
* Non-independently and identically distributed datasets (as is often the case for federated datasets)

## Turbofan Dataset

[NASA's Turbofan Dataset](https://data.nasa.gov/dataset/Turbofan-engine-degradation-simulation-data-set/vrks-gjie) consists of engine degradation data, where C-MAPSS engine simulations were run to failure, and various operational and sensor values simulated.

The matthiaslau POC uses the FD001 files in the dataset, divides the dataset into four sets. In our case, we use these splits:
<table>
  <tr>
    <th>Split</th>
    <th>No. of engines</th>
  <tr>
    <td>Initial training</td>
    <td>10</td>
  </tr>
  <tr>
    <td>Train</td>
    <td>90, split evenly across workers</td>
  </tr>
  <tr>
    <td>Validation</td>
    <td>50</td>
  <tr>
    <td>Test</td>
    <td>50</td>
  </tr>
</table>

### Data preprocessing

In addition to the preprocessing done by matthiaslau, I also:

* Minibatch the dataset, using minibatches of size 4 by default. This was to improve the quality of our gradient descent updates.
* Add support for differential privacy to data using a Laplacian mechanism, which the probability of adding noise set at 0.2 by default.

## Network set-up

### Pre-requisites

### Running the instances

An idealised configuration for FL might look like this:

<img src="/images/network.png" alt="Idealised schematic of FL network components." width="500">

To run our network components, I use AWS EC2 instances with the following specifications (otherwise left to default):
* Instance type: t3.medium
* OS: Ubuntu
* AMI: Deep Learning Conda AMI 

For our POC, I simplify the configuration by using 2 instances:

1. Instance Alice (can be understood as a "central server")
    1. Runs [PyGridNetwork](https://github.com/OpenMined/PyGridNetwork)
    1. Runs Jupyter Notebooks contained in this repository: to issue FL instructions to workers
1. Instance Bob (can be understood as comprising multiple data owners)
    1. Runs several [PyGridNodes](https://github.com/OpenMined/PyGridNode)
  
## Initialise the repository

Clone this repository and run: `bash init.sh` to initialise the repository, and download and preprocess the dataset.

## Run PyGrid components in the background

See Part 1.1 of `distribute_dataset.ipynb` for detailed instructions on how to initialise the network components. NOTE: A deprecation notice has been issued for PyGridNode, so this method is subject to change in the near future.

## Distribute data to workers

In our case, the workers do not already host the datasets on their servers. I use `distribute_dataset.ipynb` to distribute 2 lots of data from the "central server" Alice to Bob.

Included in this notebook is an option to add differential privacy to the data, according to a certain probability to add noise (arbitratily chosen to be 0.2 by default).

## Run training

Training is run using`train.ipynb`. Feel free to play with the various hyperparameters.

## Testing

Saved models can be tested using `test.ipynb`, which outputs the mean error on the test dataset.

## With thanks to
1. Prof Li Zeng Xiang at the Institute of High Performance Computing, Singapore
1. [Turbofan Federated Learning POC by matthiaslau](https://github.com/matthiaslau/Turbofan-Federated-Learning-POC)
1. [Federated Learning for Credit Scoring at OpenMined](https://blog.openmined.org/federated-credit-scoring/)

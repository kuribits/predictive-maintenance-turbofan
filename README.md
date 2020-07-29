# Turbofan Proof of Concept
Using the Turbofan dataset and PySyft to generate predictions of engine failure based on models trained using Federated Learning (FL).

This repository is based on the [Turbofan Federated Learning POC by matthiaslau](https://github.com/matthiaslau/Turbofan-Federated-Learning-POC), which demonstrated how to run Federated Learning on the Turbofan simulated engine dataset for prediction of engines' Remaining Useful Life. This would help jet engine manufacturers better plan maintenance schedules to prevent unexpected engine downtime.

Many components of PySyft have been updated in the 5 months since its last commit. This repository attempts to bring the Turbofan POC up-to-date, and introduce experimental results on the combinatorial effects of FL and: 

* Differential privacy, added at different levels
* Non-independently and identically distributed datasets (as is often the case for federated datasets)

## Turbofan Dataset
[NASA's Turbofan Dataset](https://data.nasa.gov/dataset/Turbofan-engine-degradation-simulation-data-set/vrks-gjie) consists of engine degradation data, where C-MAPSS engine simulations were run to failure, and various operational and sensor values simulated.

The matthiaslau POC uses the FD001 files in the dataset, divides the dataset into four sets. In our case, the split is as follows:

1. Initial training: 10 engines
1. Train: 90 engines, split evenly across workers
1. Validation: 50 engines
1. Test: 50 engines

### Data preprocessing

In addition to the preprocessing done by matthiaslau, we also:

* Minibatch the dataset, using minibatches of size 4 by default. This was to improve the quality of our gradient descent updates.
* Add support for differential privacy to data using a Laplacian mechanism, which the probability of adding noise set at 0.2 by default.

## Network set-up
We use AWS EC2 t3.medium instances to run our network components.


## Initialise the repository


Clone this repository and run:
```
bash init.sh
```

## Run PyGridNetwork components

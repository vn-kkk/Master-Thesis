# ResNet\_PP(EOS+CP)\_MRTD\_Emulator



A Residual Network trained on Piecewise Polytropic input parameters along with an auxiliary injection of Central pressure to predict Mass, Radius and Tidal Deformability.





## Contents:



0\. ResNet\_PP(EOS+CP)\_MRTD\_Emulator.ipynb

This is the main all in one jupyter notebook. Contains

* Dataset creation
* Training and
* Testing of the model

&nbsp;

---



1. Cluster



Has files that will be needed when running the job on a cluster

* Create\_dataset: Used to create a dataset with the required number of files. **ONLY NEED TO BE DONE ONCE**
* Train: Used to train a ResNet Model
* tov\_tide: The FORTRAN tov solver that returns the Tidal Deformability value



---



2\. Cluster\_Output



Contains

* The Best trained Model
* loss curves
* normalization statistics
* .err and .out files from training



---



3\. Datasets



Contains .npy files created by a cluster run within a folder



---



4\. Outputs



When the training jupyter notebook is run locally the Outputs from the code are in here. They contain:

* The Best trained Model
* loss curves
* normalization statistics

But mostly empty as the training is run on the cluster



---



5\. Testing



It contains the Testing.ipynb file and the tov\_tide requirements to Test the model accuracy after training locally.

The output from when Testing a model (locally) on unseen data land up here. It contains:

* The M-R plot
* TD-Compactness plot
* Model Statistics



---



# ResNet\_Tabular(EOS+CP)\_MRTD\_Emulator



A Residual Network trained on Tabulated Hybrid (Hadronic + Quark) input parameters along with an auxiliary injection of Central pressure to predict Mass, Radius and Tidal Deformability.





## Contents:



1. Cluster



Has files that will be needed when running the job on a cluster

* Create\_dataset: Used to create a dataset with the required number of files. 
* Train: Used to train a ResNet Model



---



2\. Cluster\_Output



Contains

* The Best trained Model
* loss curves
* normalization statistics
* .err and .out files from training



---



3\. Datasets



Contains .npy files created by either the notebook within the folder locally or on the cluster

* files\_used\_for\_training
* train\_data\_files
* val\_data\_files



---



4\. Outputs



When the training jupyter notebook is run locally the Outputs from the code are in here. They contain:

* The Best trained Model
* loss curves
* normalization statistics

But mostly empty as the training is run on the cluster



---



5\. Testing



The output from when Testing a model (locally) on unseen data land up here. It contains:

* The M-R plot
* TD-Compactness plot

Model Statistics






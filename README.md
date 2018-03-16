# Iris Classifier Demos

* Install python >= 3.4.x, numpy, scipy
* Run main.py and provide classifier option
* It will print out confusion matrix


## Avaiable Classifier Options

* GaussianML
* MixtureOfGaussian
* GaussianKDE (optional(kernelSdv) (default = 0.002))
* KNearest (optional(K) (default = 1))


## Exmpales

`python main.py GaussianML`

`python main.py GaussianKDE 0.025`

`python main.py KNearest 20`

The data in data/iris.data are from UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/iris)
Data Creator : R.A. Fisher
Donor : Michael Marchall (MARCHALL%PLU '@' io.arc.nasa.gov)


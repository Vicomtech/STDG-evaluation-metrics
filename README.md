# Standardised Metrics and Methods for Synthetic Tabular Data Evaluation

In this repository you can find a pipeline to evaluate Synthetic Tabular Data (STD) generated with different approaches. The main objective of the repository is to propose standardised metrics and methods for STD evaluation in three different dimensions: resemblance, utility and privacy. 

## Repository Structure

- EVALUATION FUNCTIONS: the folder contains 3 folders with the python scripts that contains the evaluation functions for the three different dimensions.
  - RESEMBLANCE: Folder containing the .py files with the evaluation functions of resemblance dimension.
  - UTILITY: Folder containing the .py files with the evaluation functions of utility dimension.
  - PRIVACY: Folder containing the .py files with the evaluation functions of privacy dimension.

## Used datasets

In total 6 open-source dataset have been used to generate STD and use the proposed evaluation metrics and methods. Some notebooks are provided as examples to use the proposed metrics and methods with the next datasets:
- Dataset A - Diabetes hospitals for years 1999-2008: https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008
- Dataset B - Carviovascular Disease Dataset: https://www.kaggle.com/sulianova/cardiovascular-disease-dataset
- Dataset C - Estimation of obesity levels based on eating habits and physical condition dataset: https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+
- Dataset D - Contraceptive Method Choice dataset: https://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice
- Dataset E - Pima Indians Diabetes Database: https://www.kaggle.com/uciml/pima-indians-diabetes-database
- Dataset F - Indian Liver Patient dataset: https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset)

## Used STDG approaches

To generate STD from the previously presented dataset the following four STDG approaches have been used:
- Gaussian Multivariate (GM): https://github.com/sdv-dev/Copulas
- Synthetic Data Vault (SDV): https://github.com/sdv-dev/SDV
- Conditional Tabular Generative Adversarial Network (CTGAN): https://github.com/sdv-dev/CTGAN
- Wasserstein Generative Adversarial Network with Gradient Penalty (WGANGP): https://github.com/ydataai/ydata-synthetic

## Contact

If you have any question or suggestion, do not hesitate to contact us at the following addresses:

- Mikel Hernandez: mhernandez@vicomtech.org
- Gorka Epelde: gepelde@vicomtech.org
- Ane Alberdi: aalberdiar@mondragon.edu
- Rodrigo Cilla: rodri.cilla@gmail.com
- Debbie Rankin: d.rankin1@ulster.ac.uk

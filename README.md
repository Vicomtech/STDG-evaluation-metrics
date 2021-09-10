# Standardised Metrics and Methods for Synthetic Tabular Data Evaluation

In this repository you can find a pipeline to evaluate Synthetic Tabular Data (STD) generated with different approaches. The main objective of the repository is to propose standardised metrics and methods for STD evaluation in three different dimensions: resemblance, utility and privacy. The next image show the taxonomy of the proposed metrics and methods for STD evaluation.

![Proposed pipeline for STD evaluation](https://github.com/Vicomtech/STDG-evaluation-metrics/blob/main/evaluation_pipeline.png)

## Repository Structure

- EVALUATION FUNCTIONS: the folder contains 3 folders with the python scripts that contains the evaluation functions for the three different dimensions.
  - RESEMBLANCE: Folder containing the .py files with the evaluation functions of resemblance dimension.
    - *univariate_resamblance.py*: python file containing the evaluation functions for the univariate resemblance analysis (URA).
    - *multivariate_resemblance.py*: python file containing the evaluation functions for the multivariate resemblance analysis (MRA).
    - *dimensional_resemblance.py*: python file containing the evaluation functions for the dimensional resemblance analysis (DRA).
    - *data_labelling.py*: python file containing the evaluation functions for the data labelling analysis (DLA).
  - UTILITY: Folder containing the .py files with the evaluation functions of utility dimension.
    - *utility_evaluation.py*: python file containing the evaluation functions for the train on real, test on real (TRTR) and train on synthetic, test on real (TSTR) analyses.
  - PRIVACY: Folder containing the .py files with the evaluation functions of privacy dimension.
    - *similarity_evaluation.py*: python file containing the evaluation functions for the similarity evaluation analysis (SEA).
    - *membership_resemblance.py*: python file containing the evaluation functions for the membership inference attack (MIA) simulation.
    - *attribute_inference.py*: python file containing the evaluation functions for the attribute inference attack (AIA) simulation.
- PREPROCESSING: the folder contains 2 python files with some functions for data preprocessing for synthetic tabular data generation (STDG)
  - *gaussian_multivariate.py*: python file containing the preprocessing functions for Gaussian Multivariate (GM) STDG approach.
  - *preprocessing.py*: python file containing the preprocessing functions for some STDG approaches.
- notebooks: the folder contains 6 subfolders, one for each used dataset, that contains notebooks examples for STDG and STD evaluation. For each subfolder the next files and folders can be found:
  - *EDA and Data Split Dataset X.ipynb*: notebook in which a brief exploratory data analysis (EDA) and a data split (80% for STDG and 20% for utility evaluation) is performed.
  - Synthetic data generation: folder that contains the STDG approaches for each dataset.
    - *CTGAN Dataset X.ipynb*: notebook to generate STD using CTGAN approach.
    - *GM Dataset X.ipynb*: notebook to generate STD using GM approach.
    - *SDV Dataset X.ipynb*: notebook to generate STD using SDV approach.
    - *WGANGP Dataset X.ipynb*: notebook to generate STD using WGANGP approach.
  - Synthetic data evaluation: folder that contains fodlers with the notebooks for STD evaluation for each dataset. The next folders can be found:
    - Privacy: contains the notebooks for the privacy evaluation of STD.
      - *1_Similarity_Evaluation_DatasetX.ipynb*: notebook for the similarity evaluation analysis (SEA).
      - *2_Membership_Inference_DatasetX.ipynb*: notebook for the membership inference attack (MIA).
      - *3_Attribute_Inference_Test_DatasetX.ipynb*: notebook for the attribute inference attack (AIA).
    - Resemblance: contains the notebooks for the resemblance evaluation of STD.
      - *1_Univariate_Resemblance_DatasetX.ipynb*: notebook for the univariate resemblance analysis (URA).
      - *2_Multivariate_Resemblance_DatasetX.ipynb*: notebook for the multivariate resemblance analysis (MRA).
      - *3_Dimensional_Resemblance_DatasetX.ipynb*: notebook for the dimensional resemblance analysis (DRA).
      - *4_Data_Labelling_Resemblance_DatasetX.ipynb*: notebook for the data labelling analysis (DLA).
    - Utility: contains the notebooks for the utility evaluation of STD.
      - *TRTR Dataset X.ipynb*: notebook for Train on Real Test on Real (TRTR) analysis.
      - *TRTR and TSTR Results Comparison.ipynb*: notebook for the comparison of the results when TRTR and Train on Synthetic Test on Real (TSTR).
      - *TSTR CTGAN Dataset X.ipynb*: notebook for TSTR analysis with STD generated with CTGAN approach.
      - *TSTR GM Dataset X.ipynb*: notebook for TSTR analysis with STD generated with GM approach.
      - *TSTR SDV Dataset X.ipynb*: notebook for TSTR analysis with STD generated with SDV approach.
      - *TSTR WGANGP Dataset X.ipynb*: notebook for TSTR analysis with STD generated with WGANGP approach.

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

## Acknowledgments

To acknowledge code and examples of this repository (*[Standardised Metrics and Methods for Synthetic Tabular Data Evaluation](https://github.com/Vicomtech/STDG-evaluation-metrics)*) in an academic publication, please cite

> Standardised Metrics and Methods for Synthetic Tabular Data Evaluation
>
> Mikel Hernandez, Gorka Epelde, Ane Alberdi, Rodrigo Cilla and Debbie Rankin
>
> Pre-print at Zenodo DOI:
> [10.5281/zenodo.5356157](https://doi.org/10.5281/zenodo.5356157)

**Note:** Currently manuscript is under revision, this section will be updated when we have updates. If you use or find this repository helpful, please take the time to star this repository on Github. This is an easy way for us to assess adoption and it can help us obtain future funding for the project.

## Contact

If you have any question or suggestion, do not hesitate to contact us at the following addresses:

- Mikel Hernandez: mhernandez@vicomtech.org
- Gorka Epelde: gepelde@vicomtech.org

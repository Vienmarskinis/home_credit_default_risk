# Module 3: Machine Learning

## Capstone Project - Home Credit Default Risk Kaggle Competition

### Author: Tomas Balsevičius

The Project has been split into 4 main notebooks: step1 to step4.
There's also supplemental material: refused_pycaret.ipynb and Refused_sweetviz.html, where I explored the power of autoML and "autoEDA"

The project was created using python-3.11.1

To run the step1-step4 notebooks, first install dependencies using pip:

```Python
pip install -r requirements.txt
```

To run the refused_pycaret.ipynb notebook, I have used another virtual environment:

```Python
pip install -r requirements_pycaret.txt
```

## Introduction
For the full project introduction, please refer to step1 notebook.

I was tasked with the hypothetical situation described in 341.ipynb - it's an unmodified file provided by Turing College. The main takeaways from the problem is that I am given a loan company's dataset consisting of multiple linked tables. I am free to create my own project structure and several proof-of-concept models that I could offer to any loan companies. 

### Project Structure
This is the actual project structure that serves as the actual documentation of the project.

**Step 1. Test / Train data split** 
- I solved two business problems in this project, but this split happens only once at the very beginning of the project.

**Step 2 and 3. EDA, Aggregations and Modelling TARGET** 
- These steps were hard to split. I iteratively explore and create aggregations in step 2, while adding them and seeing what works in step 3 modelling. In step 3, I created a reduced complexity 20 feature model, selected threshold according to assumptions about business needs, tested.

**Step 4. EDA and Modelling of the Application Refused Feature**
- Another business problem solution. This time only one table is involved, more intricate tools for EDA, statistical analysis, modelling are used. Estimated impact on customer segments. The model is tested and deployed.

### Potential Business Problems
- Predict the "TARGET" feature in the application_train table. This feature indicates whether a client has payment problems.
- Predict the "NAME_CONTRACT_STATUS" in the previous_application table. This feature indicates whether the application was approved, cancelled, refused.
- Predict the "CODE_REJECT_REASON" in the previous_application table. This feature indicates why the application was rejected.

## Technologies

In the project, I will be using these technologies and libraries:
- Python,
- Pandas (dataframes),
- Seaborn and Matplotlib (visualisation),
- SweetViz ("autoEDA"),
- SciPY (statistical inference)
- Sklearn and XGBoost (modelling),
- PyCaret (autoML),
- Sklearn pipelines (data processing),
- Shap (feature importance),
- Boruta (feature selection),
- Docker (containerization),
- Google Cloud (deployment).

## Conclusions

In this project, I have achieved the following:
- Created the project workflow and brain-stormed three potential business problems, for two of which I have prepared ML solutions, and one is yet to be investigated.
- Performed EDA on the whole dataset.
- Attempted modelling the application table's TARGET feature in step 3. This has required to prepare a lot of aggregations from other tables in step 2.
- Created a fairly good model to predict whether the application should be refused or not.

## What Can Be Improved

- In TARGET modelling, cycle back to feature engineering, explore aggregations that use only the latest few statuses.
- Deploy the TARGET model. This would most likely require setting up a dynamic database with all the tables.
- Since the "Application refused" problem is time-dependant, it could be good to train only on the latest applications provided there's enough data for that. We should also test only on the freshest data, not a random sample.
- Aggregate the cash, credit, installments, also the previous applications and use them to enrich the dataset for the "Application refused" problem. This would require more intricate work for training the model than we have done in steps 2-3.
- There's one potential business problem left to explore and more problems can most likely be brain-stormed.
# 1. Project Discription
## 1.1 What is the goal of this project?

This project is based on the author's internship in one of the five major banks in Mainland China,  
**which aims to**: 

* enhance the overall performance of **black-box models**, as well as to 
* showcase the models using **Explanable Artificial Intelligence (XAI)**

![goal](https://github.com/DannyyDing/Bank-project/blob/main/imgs/goals.png)


## 1.2 What is the background of this project?
Compared with traditional statistical models, different kinds of machine learning algorithms 
have better predictive performance. Black box models, like `XGBoost` and `Deep Neural Networks (DNN)`, 
empower homo sapiens with even more powerful tools, which might ***further improve accuracy***.
As it is required for these black-box models, algorithm enginners are to make some tuning to these models
to reach this goal.

Nevertheless, latent risk might also be brought about pursuing this target. The internal structure of
these black-box models might be more and more complicated, leading to the fact that ***they are not explanable***.
In some high-risk decision-making, such as anti-money laundering, anti-fraud and other areas applied to, the "black-box" attribute of the model
may endow the model with unpredictable risks or make biased decisions, and the model is not easily detectable when it is attacked,
***which poses great threat to financial stability***.

**Explanable Artificial Intelligence (XAI)**, however, might be the route out. By using statistical theories, 
`SHAPley Values (SHAP)`, `Partial Dependence Plot (PDP)`, `Individual Connditional Expectation (ICE)` and 
`Accumulated Local Effect (ALE)`, we could evaluate the black-box models applied, and 
know the importance of each features, which we could use to have a deeper understanding of ***what is really important when black-box models make
the decision***. 

## 1.3 What is the route of accomplishing two goals?
**Goal 1**: To set a `deep neural network (DNN)` using `PyTorch`.  
**Goal 2**: By using `General Additive Models (GAM)`, we structing unlinear data that are close
to real-world situations, and substantiated the feasibility of `SHAP`, 
after which we implement this algorithm in real world situations.

# 2. Data Description
## 2.1 Kaggle Bike Sharing Dataset

**Resource**: [visit here](https://www.kaggle.com/lakshmi25npathi/bike-sharing-dataset)  
**Main Role**: Substantiating the feasibility of SHAP in real world situations  
**Description**:  
This dataset contains the hourly and daily count of rental bikes between the years 2011 and 2012
in the Capital bike share system with the corresponding weather and seasonal information. 
We took daily data (`731 x 13`) for analysis (with 12 features and one label named cnt).  
After `Feature Engineering`, the final shape of the data applied would be `731 x 35`. 

## 2.2 Kaggle Lending Club Dataset

**Resource**: [visit here](https://www.kaggle.com/wordsforthewise/lending-club)  
**Main Role**: Enhancing the accuracy of DNN   
**Description**:  
LendingClub is a company that helps members pay down high-interest debt, 
save money, and take control of their financial future. 
The original dataset (`57094 x 95`) for analysis (with 94 features and one label named target).  
After `Feature Engineering`, the final shape of the data applied would be `57094 x 161`. 


© Author Information  
By **Danny Ding**  
Junior at Shanghai Jiao Tong University  
Email: [dannyquad@sjtu.edu.cn](dannyquad@sjtu.edu.cn)

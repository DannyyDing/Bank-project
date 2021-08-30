# 1. Project Discription
## 1.1 What is the goal of this project?
![goal](https://github.com/DannyyDing/Bank-project/blob/main/imgs/goals.png)

This project is based on the author's internship in one of the five major banks in Mainland China,  
**which aims to**: 

* enhance the overall performance of **black-box models**, as well as to 
* showcase the models using **Explanable Artificial Intelligence (XAI)**

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

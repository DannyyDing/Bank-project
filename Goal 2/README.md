# Goal 2  
We showcase the models using **Explanable Artificial Intelligence (XAI)**.
## 2.1 Feature_Engineering#1
### **Target**:  
to implement feature engineering(full-dummy variables)  
### **Notes**:  
We **do not** transform the `NOMINAL variables` into traditionnal dummy variables.  
Instead, we transform them into `FULL-dummy Variables`.  
Remember the time when we implement
```javascript
  pandas.get_dummies
```
it creates k-1 new variables for each `NOMINAL variables` (where k equals the number of different classes within a single `NOMINAL variable`).  
Admittedly, it does make the features the simplist. If one single dummy variable is allocated with 1(with others set to 0), then it shows the class that the dummy variable stands for.
If all the dummy variables are allocated with 0, then it stands for the class that these k-1 situations missed.  
However, ***please remember our target is to explain the model***. So if we refer to the traditional way, we are not able to
account for the missed situation. Is it important? Or, does it have nothing to do with the final label? We do not know.   
By transforming all `NOMINAL variables` into `FULL-dummy Variables`, we surely make the features more complicated, but thanks to this effort, ***it is easier for us
to explain the model in the final round***, i.e., the time when we implement SHAP.

## 2.2 Feature_Engineering#2
### **Target**:  
to implement feature engineering(Standard + PCA)
### **Notes**:  
#### **2.2.1 Delete some irrelevant features**  
After 2.1 R_Feature_Engineering#1, we have 34 variables. Here, we deleted some of them.   
Our princlples whould be: since GAM requires variable independence, it is needed for us to dismiss the correlation between features.  
The deleted varibles and the corresponding reasons would be:

* ***'temp', 'atemp'***: Too much correlation with other features; 
* ***'season_SPRING', 'season_SUMMER', 'season_FALL', 'season_WINTER',***
***'yr_2011', 'mnth_JAN', 'mnth_FEB', 'mnth_MAR', 'mnth_APR', 'mnth_MAY', 'mnth_JUN',***
***'mnth_JUL', 'mnth_AUG', 'mnth_SEP', 'mnth_OKT', 'mnth_NOV', 'mnth_DEZ'***:   Could be replaced by days_since_2011
* ***'holiday_NO_WORKING_DAY'***: Not so strong correlation with cnt

#### **2.2.2 Standard and PCA**  
Then, we stantardized the data and implemented PCA.  
**PCA Variables:**  
We take ***hum and windspeed*** (which are two `RATIO variables` for PCA).  
**NONE-PCA Variablers: **  
We leave days_since_2011 alone with dummy variables because it is a kind of `INTERVAL variables`, which we are not able to PCA together.
Also, we do not do PCA for `FULL-dummy variables`, because the 0 or 1 in dummy variables is not meaningful at all.
If we conduct PCA, the result is not explanable.  
We could use correlation map to evaluate the effect of PCA.   
**Before PCA**:  
<img src="https://github.com/DannyyDing/Bank-project/blob/main/imgs/PCA_before.png" width="700" alt="before PCA">  
**After PCA**:  
<img src="https://github.com/DannyyDing/Bank-project/blob/main/imgs/PCA_after.png" width="700" alt="after PCA">
We could see that PCA has eliminated the correlation of the first two `RATIO variables`.  
At the same time, we saved the `pca_components file` for subsequent processes.

## 2.3 GAM
### **Target**:  
to do GAM (thin-plate regression spline) on processed data, struct the label that is close to the real-world situation, 
and to calculate the weight gathered by GAM
### **Notes**:  
#### **2.3.1 Ideas for structing data close to real situations using GAM (thin-plate regression spline)**  
Generalized Additive Models (GAM) is a natural way to extend the multiple linear regression model:  
<img src="https://github.com/DannyyDing/Bank-project/blob/main/imgs/GAM_1.png" width="500" alt="MLR">  
In order to allow for non-linear relationships between each feature,
we replace each linear component with a (smooth) non-linear function:  
<img src="https://github.com/DannyyDing/Bank-project/blob/main/imgs/GAM_2.png" width="500" alt="GAM">  
There are many methods for fitting functions to a single variable. For example, we could struct data 
manually, like linear, higher-order or non-linear functions. In these methods, however, no matter how intentive we are
when we allocate the functions, it is highly possible that they do not reflect the real world.  
The aim for us is to get what kind of features is important. Admittedly, **it might be easier for us to check the result given by structed data we create, and the result given by SHAP**
(it is easy because WE KNOW what we used to struct the data, the only thing we need to do is to 
check whether these features would be given by SHAP, then everything is done), 
***but the fatal issue is that it deviates from reality, where linear, high-order or non-linear functions might be present at the same time***. Then, however easy it might be to use this method, it is not meaningful at all.  
We use `GAM (thin-plate regression spline)` to tackle this issue. In `GAM`, A `natural spline` is a kind of `regression spline`, i.e., we require:  

* continuity at the knot
* continuity in the first derivative at the knot, and
* Require continuity in the second derivative at the knot


with an additional boundary constraint that solves **the high variance issue**, i.e., it is 
the function is linear at the boundary,the region where X is:

* smaller than the smallest knot, or
* larger than the largest knot

In this way, we could use this algorithm to work out a complicated function that is the closest to the real-world situation.
At the same time, we could also see what features contributed the most to the structed labels.  
Take the following feature for an example:
```javascript
  gam(cnt ~ s(days_since_2011), data = bike)
```
In the instantiation of GAM, we do not include a value for parameter bs,
therefore the default basis: **bs = 'tp'** will be used. It is `thin-plate regression spline`, which is similar to 
`natural spline` with knots at some unique values of X.  
**But there is some kind of difference: **  

* **It is controlled by penality λ**, and
* **The knots are not predictable**. Thin plate regression spline is a low rank approximation to `full thin-plate spline`,
and is based on truncated eigen decomposition, which lowers the computation complexity.
This is a similar idea to principal components analysis (PCA).

We also do not set freedom K (which takes 10 as default). Then, the fitting curves are shown below.

<img src="https://github.com/DannyyDing/Bank-project/blob/main/imgs/GAM_3.png" width="500" alt="GAM Result (separate)">
<img src="https://github.com/DannyyDing/Bank-project/blob/main/imgs/GAM_4.png" width="500" alt="GAM Result (separate)">  


With K = 10, we plotted 9 figures (order number from 1-9), and the one not plotted is for intercept that is horizontal.  
We could allocate these 9 figures with different order numbers by counting the places that have `Zero First Derivative`. 
Thus, the `thin-plate regression spline` could give us the curve with a certain degree, and its weight (Note: the intercept value is the mean of cnt).  
Then, we could make an integrated curve, which is shown below.  
<img src="https://github.com/DannyyDing/Bank-project/blob/main/imgs/GAM_6.png" width="500" alt="GAM Integrated">  
So, we have completed the process of evaluating the contribution. Given the curve, we could **calculate the contribution based on the X value**
(in this case, X is days_since_2011). And ***that contribution equals f(X)*** which is one part of the function we refer to previously.  
<img src="https://github.com/DannyyDing/Bank-project/blob/main/imgs/GAM_2.png" width="500" alt="GAM">  

And next, if we devide f(X) with intercept (the mean of cnt), we could know ***the importance of this feature***.  
<img src="https://github.com/DannyyDing/Bank-project/blob/main/imgs/GAM_5.png" width="300" alt="GAM Importance">  

In essence, we refer to `GAM (thin-plate regression spline)` for structing the data, the reason why we resort to this method would be:  

* The structed data are near to ***the real-world situation***;
* We know each function of each degree clearly, and after combination with weight, we could just 
check the function value corresponding to x regardless of discrete figures, which ensures that
***we structed the data with a certain function as well as convenience***;
* We could compare the importance calculated by f(x) with the one given by SHAP afterwards to ***substantiate the feasibility of SHAP***.

#### **2.3.2 GAM for PCA Variables(X0 and X1)**  
Following the method staterd in 2.3.1, we plotted the graph for X0 and X1 (which originally are ***hum*** and ***windspeed***):  
<img src="https://github.com/DannyyDing/Bank-project/blob/main/imgs/GAM_X0.png" width="500" alt="GAM_X0">  
<img src="https://github.com/DannyyDing/Bank-project/blob/main/imgs/GAM_X1.png" width="500" alt="GAM_X1">  

#### **2.3.3 GAM for NONE-PCA Variables**  
The method is the same, and the curves are shown below.  
<img src="https://github.com/DannyyDing/Bank-project/blob/main/imgs/GAM_days_since_2011.png" width="500" alt="GAM_days_since_2011">  
<img src="https://github.com/DannyyDing/Bank-project/blob/main/imgs/GAM_holiday.png" width="500" alt="GAM_holiday">  
<img src="https://github.com/DannyyDing/Bank-project/blob/main/imgs/GAM_weekday.png" width="500" alt="GAM_weekday">  
<img src="https://github.com/DannyyDing/Bank-project/blob/main/imgs/GAM_weathersit.png" width="500" alt="GAM_weathersit">  

#### **2.3.4 Work out the final label** 
We add up all the f(X) as well as the intercept to work out the final label.

#### **2.3.5 Importance Calculation**
For ***Non-PCA variables***, we could use this function stated above to calculate the importance given by GAM:  
<img src="https://github.com/DannyyDing/Bank-project/blob/main/imgs/GAM_5.png" width="300" alt="GAM Importance">  
For ***PCA variables***, we would like to see the importance of ***hum*** and ***windspeed***
rather than ***X0*** and ***X1***. So, we need to do one step of transformation which is based on
`L2 normal form distance`:  
<img src="https://github.com/DannyyDing/Bank-project/blob/main/imgs/GAM_7.png" width="500" alt="L2 normal form distance">  
In the function above,  

* ***x*** is the original variable, and ***X*** is the variable after PCA;
* ***PCA*** is one component from the matrix 
```javascript
  pca.components_
```
* ***I(x)*** is the importance of the original variable, and ***I(X)*** is the importance of the variable after PCA.  

In particular, for the case in this project, we have:  
<img src="https://github.com/DannyyDing/Bank-project/blob/main/imgs/GAM_hum%26windspeed.png" width="1000" alt="GAM_hum&windspeed">  
We randomly select 10 sample points. For each sample point, we calculate the importance of all the features 
and arrange them in descending order, take the top 5 importances and plot them.  
Take #552 for an example:  
<img src="https://github.com/DannyyDing/Bank-project/blob/main/imgs/GAM_%23552.png" width="500" alt="GAM_#551">  
In next session, we are to get the importance from SHAP and compare the results.

## 2.4 XGBoost
### **Target**:  
to implement XGBoost on the processed data, and use SHAP to explain the model
### **Notes**:  
#### **2.4.1 XGBoost(without tuning)**  
We did not do any tuning to XGBoost, because **the raw model has a relatively high accuracy**, and
our main target in `Goal 2` is **to substantiate the feasibility of SHAP** rather than to find a 
even powerful regression tree.

#### **2.4.2 SHAP for local explanation**  
The SHAP algorithm, an explanatory algorithm implemented in 2017 by Scott M. Lundberg. ([visit here](https://proceedings.neurips.cc//paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf)), is based on
the idea of `Joint Game Theory`, and it uses the `Shapley Value Principle` to perform feature contribution scores on the prediction.  
`Shapley Value Principle` is a rigorous value distribution principle, and is based on the average value of each member's marginal contribution, whose aim is to decompose the total value of the project.  
<img src="https://github.com/DannyyDing/Bank-project/blob/main/imgs/SHAP.png" width="800" alt="SHAP">  
The function above is for **local explanation**, i.e., for illustrating what is important for a single sample, where:  

* ***j***: the feature to be explained; 
* ***S***: Feature Space; 
* ***p***: the number of features; 

By using ***Combination***, we could rewrite the function above as:  
<img src="https://github.com/DannyyDing/Bank-project/blob/main/imgs/SHAP_simple.png" width="800" alt="SHAP simple">  
Therefore, Shapley Value is the result of extracting S features from the feature space
from which the feature to be observed is removed, and taking the average again
on this extracted whole feature space.

We compared the TOP 5 variables that are important given by GAM and SHAP on 10 randomly-selected samples,
and the matching ratio is `0.74`(see Importance comparison between GAM and SHAP.pdf, [see here](https://github.com/DannyyDing/Bank-project/blob/main/Goal%202/Importance%20comparison%20between%20GAM%20and%20SHAP.pdf)), which substantiated the feasibility of using SHAP to explain the 
machine learning models, and account for the importance of features.


© Author Information  
By **Danny Ding**  
Junior at Shanghai Jiao Tong University  
Email: [dannyquad@sjtu.edu.cn](mailto:dannyquad@sjtu.edu.cn)

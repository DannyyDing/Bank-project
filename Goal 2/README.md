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
to do GAM (Natural Splines) on processed data, struct the label that is close to the real-world situation, 
and to calculate the weight gathered by GAM
### **Notes**:  
#### **2.3.1 Ideas for structing data close to real situations using GAM (Natural Splines)**  
Generalized Additive Models (GAM) is a natural way to extend the multiple linear regression model:  
<img src="https://github.com/DannyyDing/Bank-project/blob/main/imgs/GAM_1.png" width="500" alt="MLR">  
In order to allow for non-linear relationships between each feature,
we replace each linear component with a (smooth) non-linear function:  
<img src="https://github.com/DannyyDing/Bank-project/blob/main/imgs/GAM_2.png" width="500" alt="GAM">  
There are many methods for fitting functions to a single variable. For example, we could struct data 
manually, like linear, higher-order or non-linear functions. In these methods, however, no matter how intentive we are
when we allocate the functions, it is highly possible that they do not reflect the real world.  
Admittedly, **it might be easier for us to check the result given by structed data we create, and the result given by SHAP**
(it is easy because WE KNOW what we used to struct the data, the only thing we need to do is to 
check whether these features would be given by SHAP, then everything is done), 
***but the fatal issue is that it deviates from reality***. Then, however easy it might be to use this method, it is not meaningful at all.  
In `GAM`, A `natural spline` is a kind of **regression spline**, i.e., we require:  

* continuity at the knot
* continuity in the first derivative at the knot, and
* Require continuity in the second derivative at the knot


with an additional boundary constraint that solves **the high variance issue**, i.e., it is 
the function is linear at the boundary,the region where X is:

* smaller than the smallest knot, or
* larger than the largest knot

In this way, we could use this algorithm to work out a function that is the closest to the real-world situation.
At the same time, we could also see what features contributed to the structed labels.  
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
based on truncated eigen decomposition, which lowers the computation complexity.
This is a similar idea to principal components analysis (PCA).








© Author Information  
By **Danny Ding**  
Junior at Shanghai Jiao Tong University  
Email: [dannyquad@sjtu.edu.cn](mailto:dannyquad@sjtu.edu.cn)

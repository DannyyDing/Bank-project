# Goal 2  
We showcase the models using **Explanable Artificial Intelligence (XAI)**.
## 2.1 R_Feature_Engineering#1.ipynb
###**Target**:  
to implement feature engineering(full-dummy variables)  
###**Notes**:  
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

## 2.2 Python_Feature_Engineering#2
### **Target**:  
to implement feature engineering(Standard + PCA)
### **Notes**:  
####**2.2.1 Delete some irrelevant features**  
After 2.1 R_Feature_Engineering#1, we have 34 variables. Here, we deleted some of them.   
Our princlples whould be: since GAM requires variable independence, it is needed for us to dismiss the correlation between features.  
The deleted varibles and the corresponding reasons would be:

* ***'temp', 'atemp'***: Too much correlation with other features; 
* ***'season_SPRING', 'season_SUMMER', 'season_FALL', 'season_WINTER',***
***'yr_2011', 'mnth_JAN', 'mnth_FEB', 'mnth_MAR', 'mnth_APR', 'mnth_MAY', 'mnth_JUN',***
***'mnth_JUL', 'mnth_AUG', 'mnth_SEP', 'mnth_OKT', 'mnth_NOV', 'mnth_DEZ'***:   Could be replaced by days_since_2011
* ***'holiday_NO_WORKING_DAY'***: Not so strong correlation with cnt

####**2.2.2 Standard and PCA**  
Then, we stantardized the data and implemented PCA.  
**PCA Variables:**  
We take ***hum and windspeed*** (which are two `RATIO variables` for PCA).  
**NONE-PCA Variablers: **  
We leave days_since_2011 alone with dummy variables because it is a kind of `INTERVAL variables`, which we are not able to PCA together.
Also, we do not do PCA for `FULL-dummy variables`, because the 0 or 1 in dummy variables is not meaningful at all.
If we conduct PCA, the result is not explanable.  
We could use correlation map to evaluate the effect of PCA.   
**Before PCA**:  
![before PCA](https://github.com/DannyyDing/Bank-project/blob/main/imgs/PCA_before.png)  
**After PCA**:  
![after PCA](https://github.com/DannyyDing/Bank-project/blob/main/imgs/PCA_after.png)
We could see that PCA has eliminated the correlation of the first two `RATIO variables`.


© Author Information  
By **Danny Ding**  
Junior at Shanghai Jiao Tong University  
Email: [dannyquad@sjtu.edu.cn](mailto:dannyquad@sjtu.edu.cn)

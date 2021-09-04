# Goal 1  
We enhance the overall performance of black-box models, i.e., **Deep Neural Networks (DNN)** with **PyTorch**.
## 1.1 Feature_Engineering
### **Target**:  
to implement feature engineering (full-dummy variables)
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
By transforming all `NOMINAL variables` into `FULL-dummy Variables` (i.e., we create k new variables, and if the data sample
belong to the certain class, then it would be 1, and all other variables would be allocated with 0), we surely make the features more complicated. But, thanks to this effort, ***it is easier for us
to explain the model in the final round***, i.e., the time when we implement SHAP.

For the variables listed below, we transformed them into full-dummy variables (for the whole description of fields,
[see here](https://github.com/DannyyDing/Bank-project/blob/main/Goal%201/DataSet/LCDataDictionary.xlsx)):  

* ***'home_ownership'(4)***: "The home ownership status provided by the borrower during registration. Our values are:
RENT, OWN, MORTGAGE, OTHER."
* ***'verification_statu'(3)***: "Indicates if the borrowers income was verified by
LC, not verified, or if the income source was verified."
* ***'purpose'(12)***: "A category provided by the borrower for the loan request."
* ***'addr_state'(49)***: "The state provided by the borrower in the loan application."
* ***'initial_list_status'(2)***: "The initial listing status of the loan. Possible values are – W, F."
* ***'application_type'(2)***: "Indicates whether the loan is an individual application or a joint application with two coborrowers."
* ***'hardship_flag'(2)***: "Flags whether or not the borrower is on a hardship plan."
* ***'debt_settlement_flag'(2)***: "Flags whether or not the borrower, who has charged-off, is working with a debt-settlement company."

Also, we deleted two variables:  

* ***'funded_amnt'***: too much correlation with other features;
* ***'term'***: all data samples share the same value '60', which is not helpful in predicton.

## 1.2 DNN
### **Target**:  
to build a DNN model based on PyTorch
### **Notes**:  
The whole training route for DNN model built is shown below.  
<img src="https://github.com/DannyyDing/Bank-project/blob/main/imgs/DNN_route.png" width="1000" alt="DNN Training Route">  






© Author Information  
By **Danny Ding**  
Junior at Shanghai Jiao Tong University  
Email: [dannyquad@sjtu.edu.cn](mailto:dannyquad@sjtu.edu.cn)

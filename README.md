# 信用卡违约率分析
***开发环境：jupyter notebook, python3.6***

## 数据说明
数据集来自 UCI Machine Learning Repository, 是台湾某银行2005年4月到9月的信用卡数据，包含25个字段，字段含义如下：

1. ID: ID of each client
2. LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
3. SEX: Gender (1=male, 2=female)
4. EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
5. MARRIAGE: Marital status (1=married, 2=single, 3=others)
6. AGE: Age in years
7. PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above)
8. PAY_2: Repayment status in August, 2005 (scale same as above)
9. PAY_3: Repayment status in July, 2005 (scale same as above)
10. PAY_4: Repayment status in June, 2005 (scale same as above)
11. PAY_5: Repayment status in May, 2005 (scale same as above)
12. PAY_6: Repayment status in April, 2005 (scale same as above)
13. BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
14. BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
15. BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
16. BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
17. BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
18. BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
19. PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
20. PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
21. PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
22. PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
23. PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
24. PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
25. default.payment.next.month: Default payment (1=yes, 0=no)

## 模型选择
初步选择 SVM, RF随机森林, KNN 分类器对模型进行分析，使用 GridSearch 获取最佳参数，所有库来自 sklearn，详见 credit_default_analysis。最终模型的得分如下：

```python
GridSearch最优参数： {'svc__C': 1, 'svc__gamma': 0.01}
GridSearch最优分数： 0.8173
准确率 0.8195

GridSearch最优参数： {'randomforestclassifier__n_estimators': 6}
GridSearch最优分数： 0.7984
准确率 0.8047

GridSearch最优参数： {'kneighborsclassifier__n_neighbors': 8}
GridSearch最优分数： 0.8040
准确率 0.8038
```
可见 SVM 的准确率最高。

## SVM
下面使用 SVM 作进一步分析，详见 credit_default_svm.
### 数据观察
通过观察变量的关系，发现在最后的12个特征中，分别代表过去六个月的还款和账单数额，0代表未还款，发现最后六个特征的未还款期数和与违约存在正对应关系，所以在模型中需要增加一列表示这个值。
另外性别、教育和结婚情况是分类特征，需要作一个离散化处理，可以使用 One-Hot-Encode。
### 特征工程
构建新的特征列，分别表示账单为0和还款为0的情况：

```python
data['bill_0count'] = data[['BILL_AMT1','BILL_AMT2',
                    'BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']].apply(lambda x : x.value_counts().get(0,0),axis=1)
data['pay_0count'] = data[['PAY_AMT1','PAY_AMT2',
                    'PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']].apply(lambda x : x.value_counts().get(0,0),axis=1)
```

然后两项相减，获得有账单但未还款统计的特征列：

```python
data.eval('pay_sub_bill = pay_0count - bill_0count', inplace=True)
```

最终的特征列中，只保留还款和账单为0计数的差值列：

```python
data.drop(['ID','pay_0count','bill_0count'], inplace=True, axis=1)
target = data['default.payment.next.month']
columns = data.columns.tolist()
columns.remove('default.payment.next.month')
features = data[columns]
```

使用 OneHotEncoder 将分类特征离散化：

```python
from sklearn.preprocessing import OneHotEncoder
OHE = OneHotEncoder(sparse=False)
sex = OHE.fit_transform(features[['SEX']])
marriage = OHE.fit_transform(features[['MARRIAGE']])
education = OHE.fit_transform(features[['EDUCATION']])
cat_variables = np.hstack((sex, marriage, education))
cat_var_names = ['SEX','MARRIAGE', 'EDUCATION']
num_variables = features.drop(cat_var_names, axis=1)
final_features = np.hstack((cat_variables,num_variables))
```

### 模型评估
最后使用 SVM 进行分类的结果如下：

```python
GridSearch最优参数： {'svc__C': 1, 'svc__gamma': 0.01}
GridSearch最优分数： 0.8178
准确率 0.8220
```

分类准确率提高到0.8220.

## 业务解析
可以比较明显地看到，如果客户如果在近期存在违约的情况，很大可能会继续违约，风险管理中，应该对这类客户重点防御。

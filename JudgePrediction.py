import os
import json
import pandas as pd
import numpy as np
# from scipy.stats import randint as sp_randint
# from sklearn.metrics import accuracy_score, r2_score
# from sklearn.model_selection import cross_val_score, RandomizedSearchCV
# from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
# from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
# from sklearn.svm import SVR
# from sklearn.neural_network import MLPRegressor
# from mlxtend.regressor import StackingRegressor
# from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.tree import DecisionTreeRegressor
# import matplotlib.pyplot as plt
# import xgboost as xgb
import pyodbc

def calculate_missing_values(df):
    missing_values_count = df.isnull().sum()
    missing_values_pers = 100 * missing_values_count / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([missing_values_count, missing_values_pers], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(columns={0: 'Missing Values', 1: '% of Total Values'})

    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    return mis_val_table_ren_columns

cwd = os.getcwd()
# print(cwd + '\db_properties\db_connection.json')
with open(cwd + '\db_properties\db_connection.json') as f:
    data = json.load(f)
conn = pyodbc.connect("DRIVER={{SQL Server}};SERVER={0}; database={1}; \
       trusted_connection=no;UID={2};PWD={3}".format(data["server"]
                                                     , data["db"]
                                                     , data["login"]
                                                     , data["pass"]))
cursor = conn.cursor()
script_path = cwd + '\scripts'
file_list = ["CourtCase.sql", "CourtCaseSchedule.sql"]
# file_list = ["CourtCase.sql", "CourtCase_test.sql", "CourtCaseParty.sql", "CourtCaseSchedule.sql"
#     , "CourtCaseCrimes.sql", "CourtCasePartyLegalRepresentative.sql", "CourtCaseDocument.sql"]
datas = []
for script in os.listdir(script_path):
    if script.endswith(".sql") and script in file_list:
        file = script_path + "\\" + script
        with open(file) as s:
            sql = s.read()
            cursor.execute(sql)
            rows = np.array(cursor.fetchall())
            columns = np.array(cursor.description)[:, 0]
            datas.append(pd.DataFrame(data=rows, columns=columns))
            print('file', script,  'has been processed')

conn.close()

CourtCase, CourtCaseSchedule = datas

# CourtCase, CourtCaseCrimes, CourtCaseDocument, CourtCaseParty, CourtCasePartyLegalRepresentative \
#     , CourtCaseSchedule, CourtCase_test = datas

ccsh_aggregations = {"HearingDate" : {"min_hearingdate": "min"}}
CourtCaseSchedule = CourtCaseSchedule.groupby("CourtCaseID").agg({**ccsh_aggregations})
CourtCaseSchedule.columns = CourtCaseSchedule.columns.droplevel(level=0)
CourtCase = pd.merge(CourtCase, CourtCaseSchedule, how='left', left_on="CourtCaseID", right_index=True)
# CourtCase_test = pd.merge(CourtCase_test, CourtCaseSchedule, how='left', left_on="CourtCaseID", right_index=True)

CourtCase = CourtCase.sample(frac=1)

CourtCaseParty["IsNPPA"] = CourtCaseParty.PartyID.apply(lambda x: 1 if x == -2 else 0)

ccp_aggregations = {}
ccp_aggregations["CourtCasePartyID"] = {"ccp_count": "count"}
ccp_aggregations["IsNPPA"] = {"is_nppa_present": "max"}

CourtCaseParty = CourtCaseParty.groupby("CourtCaseID").agg({**ccp_aggregations})
CourtCaseParty.columns = CourtCaseParty.columns.droplevel(level=0)
CourtCase = pd.merge(CourtCase, CourtCaseParty, how='left', left_on="CourtCaseID", right_index=True)
CourtCase_test = pd.merge(CourtCase_test, CourtCaseParty, how='left', left_on="CourtCaseID", right_index=True)

ccsh_aggregations = {"HearingDate" : {"min_hearingdate": "min"}}
CourtCaseSchedule = CourtCaseSchedule.groupby("CourtCaseID").agg({**ccsh_aggregations})
CourtCaseSchedule.columns = CourtCaseSchedule.columns.droplevel(level=0)
CourtCase = pd.merge(CourtCase, CourtCaseSchedule, how='left', left_on="CourtCaseID", right_index=True)
CourtCase_test = pd.merge(CourtCase_test, CourtCaseSchedule, how='left', left_on="CourtCaseID", right_index=True)

CourtCaseCrimes = CourtCaseCrimes.groupby("CourtCaseID").count()
CourtCase = pd.merge(CourtCase, CourtCaseCrimes, how='left', left_on="CourtCaseID", right_index=True)
CourtCase_test = pd.merge(CourtCase_test, CourtCaseCrimes, how='left', left_on="CourtCaseID", right_index=True)

CourtCase = pd.merge(CourtCase, CourtCasePartyLegalRepresentative, how='left', left_on="CourtCaseID", right_on="CourtCaseID")
CourtCase_test = pd.merge(CourtCase_test, CourtCasePartyLegalRepresentative, how='left', left_on="CourtCaseID", right_on="CourtCaseID")

# CourtCaseAddmisibility = pd.read_csv("CourtCaseAddmisibility.csv")
# ccai_aggregations = {"AdmissibilityItemID" : {"cnt_admitem" : "count"}}
# CourtCaseAddmisibility = CourtCaseAddmisibility.groupby("CourtCaseID").agg({**ccai_aggregations})
# CourtCaseAddmisibility.columns = CourtCaseAddmisibility.columns.droplevel(level=0)
# CourtCase = pd.merge(CourtCase, CourtCaseAddmisibility, how='left', left_on="CourtCaseID", right_index=True)
# CourtCase_test = pd.merge(CourtCase_test, CourtCaseAddmisibility, how='left', left_on="CourtCaseID", right_index=True)

# CourtCaseIssues = pd.read_csv("CourtCaseIssues.csv")
# ccissues_aggregations = {"CourtCaseIssuesToBeAnalysedID" : {"cnt_issues" : "count"}}
# CourtCaseIssues = CourtCaseIssues.groupby("CourtCaseID").agg({**ccissues_aggregations})
# CourtCaseIssues.columns = CourtCaseIssues.columns.droplevel(level=0)
# CourtCase = pd.merge(CourtCase, CourtCaseIssues, how='left', left_on="CourtCaseID", right_index=True)
# CourtCase_test = pd.merge(CourtCase_test, CourtCaseIssues, how='left', left_on="CourtCaseID", right_index=True)

CourtCaseDocument = pd.get_dummies(CourtCaseDocument, columns=["DocumentTypeID"])
cc_doc_dum_agg = {}
dum_columns = [x for x in CourtCaseDocument.columns if x.startswith("DocumentTypeID")]
for col in dum_columns:
    cc_doc_dum_agg[col] = {col:"sum"}
ccdoc_aggregations = {"Size" : {"total_size" : "sum", "avg_size" : "mean"}}
CourtCaseDocument = CourtCaseDocument.groupby("CourtCaseID").agg({**ccdoc_aggregations, **cc_doc_dum_agg})
CourtCaseDocument.columns = CourtCaseDocument.columns.droplevel(level=0)
CourtCase = pd.merge(CourtCase, CourtCaseDocument, how='left', left_on="CourtCaseID", right_index=True)
CourtCase_test = pd.merge(CourtCase_test, CourtCaseDocument, how='left', left_on="CourtCaseID", right_index=True)



[CourtCase[x].fillna(0, inplace=True) for x in CourtCase.columns if x.startswith("ArticleID")];
CourtCase.CountOfLegalRepresentative.fillna(0, inplace=True)
CourtCase.ccp_count.fillna(0, inplace=True)
# CourtCase.cnt_admitem.fillna(0, inplace=True)
# CourtCase.cnt_issues.fillna(0, inplace=True)
CourtCase.is_nppa_present.fillna(0, inplace=True)

[CourtCase_test[x].fillna(0, inplace=True) for x in CourtCase_test.columns if x.startswith("ArticleID")];
CourtCase_test.CountOfLegalRepresentative.fillna(0, inplace=True)
CourtCase_test.ccp_count.fillna(0, inplace=True)
# CourtCase_test.cnt_admitem.fillna(0, inplace=True)
# CourtCase_test.cnt_issues.fillna(0, inplace=True)
CourtCase_test.is_nppa_present.fillna(0, inplace=True)

print(CourtCase_test.shape)
print(CourtCase.shape)

CourtCase["HasRecieptDocument"] = CourtCase.ReceiptDocumentID.fillna(0).apply(lambda x: 1 if x > 0 else 0)
CourtCase["HasProsecutionCase"] = CourtCase.ProsecutionCaseID.fillna(0).apply(lambda x: 1 if x > 0 else 0)
# CourtCase["IsAppealedcase"] = CourtCase.AppealedCourtCaseID.fillna(0).apply(lambda x: 1 if x > 0 else 0)
CourtCase["ColorID"] = CourtCase.ColorID.fillna(-1)
CourtCase["InstanceLevelID"] = CourtCase.InstanceLevelID.fillna(-1)
CourtCase["SubCategoryID"] = CourtCase.SubCategoryID.fillna(-1)
CourtCase["CasePriorityID"] = CourtCase.CasePriorityID.fillna(-1)
CourtCase["IsDetentionCase"] = CourtCase.IsDetentionCase.fillna(0)
CourtCase["IsPublicCase"] = CourtCase.IsPublicCase.fillna(0)
CourtCase["CommittedByMinor"] = CourtCase.CommittedByMinor.fillna(0)
CourtCase["GenderBasedViolence"] = CourtCase.GenderBasedViolence.fillna(0)
CourtCase["InitiatedFromAbunzi"] = CourtCase.InitiatedFromAbunzi.fillna(0)
CourtCase["SolvedFromAbunzi"] = CourtCase.SolvedFromAbunzi.fillna(0)
CourtCase["HasDetails"] = CourtCase.HasDetails.fillna(0)
CourtCase["IsExempted"] = CourtCase.IsExempted.fillna(0)
CourtCase["AttachedDate"] = CourtCase.AttachedDate.fillna(0)
CourtCase.drop(columns=["HasPassedCaseNumberAllocated", "CaseCode", "MinorVersion", "MajorVersion", "CourtID", "CourtCaseID"
                   , "ReceiptDocumentID",  "ProsecutionCaseID", "WFActionID"
                   , "NotRegisteredCaseCode", "WFStateID", "UpdatedUserID", "OwnerUserID", "PublicOwnerUserId", 'CreatedUserID'
                   ,'AppealedCourtCaseID', "CountOfJudgmentPages"
                  ], inplace=True)

CourtCase_test["HasRecieptDocument"] = CourtCase_test.ReceiptDocumentID.fillna(0).apply(lambda x: 1 if x > 0 else 0)
CourtCase_test["HasProsecutionCase"] = CourtCase_test.ProsecutionCaseID.fillna(0).apply(lambda x: 1 if x > 0 else 0)
# CourtCase_test["IsAppealedcase"] = CourtCase_test.AppealedCourtCaseID.fillna(0).apply(lambda x: 1 if x > 0 else 0)
CourtCase_test["ColorID"] = CourtCase_test.ColorID.fillna(-1)
CourtCase_test["InstanceLevelID"] = CourtCase_test.InstanceLevelID.fillna(-1)
CourtCase_test["SubCategoryID"] = CourtCase_test.SubCategoryID.fillna(-1)
CourtCase_test["CasePriorityID"] = CourtCase_test.CasePriorityID.fillna(-1)
CourtCase_test["IsDetentionCase"] = CourtCase_test.IsDetentionCase.fillna(0)
CourtCase_test["IsPublicCase"] = CourtCase_test.IsPublicCase.fillna(0)
CourtCase_test["CommittedByMinor"] = CourtCase_test.CommittedByMinor.fillna(0)
CourtCase_test["GenderBasedViolence"] = CourtCase_test.GenderBasedViolence.fillna(0)
CourtCase_test["InitiatedFromAbunzi"] = CourtCase_test.InitiatedFromAbunzi.fillna(0)
CourtCase_test["SolvedFromAbunzi"] = CourtCase_test.SolvedFromAbunzi.fillna(0)
CourtCase_test["HasDetails"] = CourtCase_test.HasDetails.fillna(0)
CourtCase_test["IsExempted"] = CourtCase_test.IsExempted.fillna(0)
CourtCase_test["AttachedDate"] = CourtCase_test.AttachedDate.fillna(0)
CourtCase_test.drop(columns=["HasPassedCaseNumberAllocated", "CaseCode", "MinorVersion", "MajorVersion", "CourtID", "CourtCaseID"
                   , "ReceiptDocumentID",  "ProsecutionCaseID", "WFActionID"
                   , "NotRegisteredCaseCode", "WFStateID", "UpdatedUserID", "OwnerUserID", "PublicOwnerUserId", 'CreatedUserID'
                   ,'AppealedCourtCaseID', "CountOfJudgmentPages"
                  ], inplace=True)

print(CourtCase_test.shape)
print(CourtCase.shape)

A = calculate_missing_values(CourtCase)
drop_col = A[A["% of Total Values"] > 90].index
CourtCase = CourtCase.drop(columns=drop_col)
CourtCase.dropna(inplace=True)
CourtCase_test = CourtCase_test.drop(columns=drop_col)
CourtCase_test.dropna(inplace=True)
print(CourtCase_test.shape)
print(CourtCase.shape)

calculate_missing_values(CourtCase)
calculate_missing_values(CourtCase_test)

X = CourtCase.drop(columns=["DecisionDuration"])
Y = CourtCase["DecisionDuration"]

# X_test = X[:100]
# Y_test = Y[:100]
# X = X[100:]
# Y = Y[100:]
X_test = CourtCase_test.drop(columns=["DecisionDuration"])
Y_test = CourtCase_test["DecisionDuration"]
print(X.shape, Y.shape)
print(X_test.shape, Y_test.shape)

# rf_params = {'n_estimators': 100,
#  'min_samples_split': 2,
#  'min_samples_leaf': 2,
#  'max_features': 'auto',
#  'max_depth': None,
#  'bootstrap': True}
#
# rf = RandomForestRegressor(**rf_params)
# np.mean(cross_val_score(rf, X, Y, scoring='r2', cv=5, verbose=5))
#
# rf.fit(X, Y)
# rf.feature_importances_
#
# feat_imp = pd.DataFrame({'importance':rf.feature_importances_})
# feat_imp['feature'] = X.columns
# feat_imp.sort_values(by='importance', ascending=False, inplace=True)
# feat_imp = feat_imp.iloc[:30]
# feat_imp.sort_values(by='importance', inplace=True)
# feat_imp = feat_imp.set_index('feature', drop=True)
# feat_imp.plot.barh(title="feature_importance", figsize=(8,8))
# plt.xlabel('Feature Importance Score')
# plt.show()
#
# rf.fit(X, Y)
# y_pred = rf.predict(X_test)
# r2_score(Y_test, y_pred)

# boost_params = {'n_estimators': 200,
#  'min_samples_split': 40,
#  'min_samples_leaf': 4,
#  'max_features': 'sqrt',
#  'max_depth': 20,
#  'learning_rate': 0.05}
# boost = GradientBoostingRegressor(**boost_params)

# boost.fit(X, Y)
# y_pred = boost.predict(X_test)
# r2_score(Y_test, y_pred)

# np.mean(cross_val_score(boost, X, Y, scoring='r2', cv=5, verbose=5))

# boost.fit(X, Y)
# boost.feature_importances_
#
# feat_imp = pd.DataFrame({'importance':boost.feature_importances_})
# feat_imp['feature'] = X.columns
# feat_imp.sort_values(by='importance', ascending=False, inplace=True)
# feat_imp = feat_imp.iloc[:30]
# feat_imp.sort_values(by='importance', inplace=True)
# feat_imp = feat_imp.set_index('feature', drop=True)
# feat_imp.plot.barh(title="feature_importance", figsize=(8,8))
# plt.xlabel('Feature Importance Score')
# plt.show()

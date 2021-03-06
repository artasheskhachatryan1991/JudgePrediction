import os
import json
import pandas as pd
import numpy as np
import argparse
from scipy.stats import randint as sp_randint
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from mlxtend.regressor import StackingRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import xgboost as xgb
import pyodbc

def load_data(conn, table, is_train = True, train_decision_date = None, test_id = None):    
    cursor = conn.cursor()       
    sql = 'select * from dbo.load_' + table    
    if table == 'CourtCase':
        sql += '(?, ?)'
        cursor.execute(sql, (test_id, train_decision_date))
        rows = np.array(cursor.fetchall())
        columns = np.array(cursor.description)[:, 0]            
        assert len(rows) > 0, 'No data found for specified filters'
    else:
        sql += '(?)'
        cursor.execute(sql, (test_id))
        rows = np.array(cursor.fetchall())
        columns = np.array(cursor.description)[:, 0]
    return pd.DataFrame(data=list(rows), columns=columns)

def load_all_data(conn, tables, is_train = True, train_decision_date = None, test_id = None):
    try:
        assert is_train or test_id != None
    except AssertionError():
        raise AssertionError("test_id must be passed in training mode") 
    results = {}    
    for table in tables:
        results[table] = load_data(conn, table, is_train, train_decision_date, test_id)
        print(f'{table} data has been processed')            
    return results

def data_preprocessing(data):
    CourtCase = data["CourtCase"]
    CourtCaseParty = data["CourtCaseParty"]
    CourtCaseSchedule = data["CourtCaseSchedule"]
    CourtCaseCrimes = data["CourtCaseCrimes"]
    CourtCaseDocument = data["CourtCaseDocument"]
    CourtCasePartyLegalRepresentative = data["CourtCasePartyLegalRepresentative"]
    
    CourtCase = CourtCase.sample(frac=1)

    CourtCaseParty["IsNPPA"] = CourtCaseParty.PartyID.apply(lambda x: 1 if x == -2 else 0)

    ccp_aggregations = {}
    ccp_aggregations["CourtCasePartyID"] = {"ccp_count": "count"}
    ccp_aggregations["IsNPPA"] = {"is_nppa_present": "max"}

    CourtCaseParty = CourtCaseParty.groupby("CourtCaseID").agg({**ccp_aggregations})
    CourtCaseParty.columns = CourtCaseParty.columns.droplevel(level=0)
    CourtCase = pd.merge(CourtCase, CourtCaseParty, how='left', left_on="CourtCaseID", right_index=True)
    

    ccsh_aggregations = {"HearingDate" : {"min_hearingdate": "min"}}
    CourtCaseSchedule = CourtCaseSchedule.groupby("CourtCaseID").agg({**ccsh_aggregations})
    CourtCaseSchedule.columns = CourtCaseSchedule.columns.droplevel(level=0)
    CourtCase = pd.merge(CourtCase, CourtCaseSchedule, how='left', left_on="CourtCaseID", right_index=True)


    CourtCaseCrimes = CourtCaseCrimes.groupby("CourtCaseID").count()
    CourtCase = pd.merge(CourtCase, CourtCaseCrimes, how='left', left_on="CourtCaseID", right_index=True)

    CourtCase = pd.merge(CourtCase, CourtCasePartyLegalRepresentative, how='left', left_on="CourtCaseID", right_on="CourtCaseID")


    # CourtCaseAddmisibility = pd.read_csv("CourtCaseAddmisibility.csv")
    # ccai_aggregations = {"AdmissibilityItemID" : {"cnt_admitem" : "count"}}
    # CourtCaseAddmisibility = CourtCaseAddmisibility.groupby("CourtCaseID").agg({**ccai_aggregations})
    # CourtCaseAddmisibility.columns = CourtCaseAddmisibility.columns.droplevel(level=0)
    # CourtCase = pd.merge(CourtCase, CourtCaseAddmisibility, how='left', left_on="CourtCaseID", right_index=True)    

    # CourtCaseIssues = pd.read_csv("CourtCaseIssues.csv")
    # ccissues_aggregations = {"CourtCaseIssuesToBeAnalysedID" : {"cnt_issues" : "count"}}
    # CourtCaseIssues = CourtCaseIssues.groupby("CourtCaseID").agg({**ccissues_aggregations})
    # CourtCaseIssues.columns = CourtCaseIssues.columns.droplevel(level=0)
    # CourtCase = pd.merge(CourtCase, CourtCaseIssues, how='left', left_on="CourtCaseID", right_index=True)    

    CourtCaseDocument = pd.get_dummies(CourtCaseDocument, columns=["DocumentTypeID"])

    cc_doc_dum_agg = {}
    dum_columns = [x for x in CourtCaseDocument.columns if x.startswith("DocumentTypeID")]
    for col in dum_columns:
        cc_doc_dum_agg[col] = {col:"sum"}
    ccdoc_aggregations = {"Size" : {"total_size" : "sum", "avg_size" : "mean"}}
    CourtCaseDocument.Size = pd.to_numeric(CourtCaseDocument.Size)

    CourtCaseDocument = CourtCaseDocument.groupby("CourtCaseID").agg({**ccdoc_aggregations, **cc_doc_dum_agg})
    CourtCaseDocument.columns = CourtCaseDocument.columns.droplevel(level=0)
    CourtCase = pd.merge(CourtCase, CourtCaseDocument, how='left', left_on="CourtCaseID", right_index=True)

    [CourtCase[x].fillna(0, inplace=True) for x in CourtCase.columns if x.startswith("ArticleID")];
    CourtCase.CountOfLegalRepresentative.fillna(0, inplace=True)
    CourtCase.ccp_count.fillna(0, inplace=True)
    # CourtCase.cnt_admitem.fillna(0, inplace=True)
    # CourtCase.cnt_issues.fillna(0, inplace=True)
    CourtCase.is_nppa_present.fillna(0, inplace=True)        

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
  
    drop_col = ['DecisionPronouncementDate', 'DecisionPronouncementDateYearID',
       'ExecutionCaseApprovedUserID', 'PaymentBankID', 'LitigationCaseID',
       'CaseRejectionID', 'ExtraOrdinaryProcedureID', 'PreviousCourtCaseID',
       'SpecialCaseID']
    CourtCase = CourtCase.drop(columns=drop_col)
    CourtCase.dropna(inplace=True)
    
    assert len(CourtCase) > 0, 'No data left after preprocessing'    

    X = CourtCase.drop(columns=["DecisionDuration"])
    Y = CourtCase["DecisionDuration"]

    return X, Y

def get_connection(path):
    cwd = os.getcwd()
    with open(cwd + path) as f:
        data = json.load(f)
    conn = pyodbc.connect("DRIVER={{SQL Server}};SERVER={0}; database={1}; \
           trusted_connection=no;UID={2};PWD={3}".format(data["server"]
                                                         , data["db"]
                                                         , data["login"]
                                                         , data["pass"]))
    return conn

def parse_args_train(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument('--properties_path',
                        default = '\db_properties\db_connection.json',
                        help='relative path to database connection properties')
    parser.add_argument('--train_decision_date',                        
                        help='date until where to train the data')
    args = parser.parse_args(*argument_array)
    return args

def parse_args_test(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_id',                        
                        help='case id to test the duration', )    
    args = parser.parse_args(*argument_array)
    return args


def main(args):
    TABLE_LIST = ["CourtCase", "CourtCaseParty", "CourtCaseSchedule"
    , "CourtCaseCrimes", "CourtCasePartyLegalRepresentative", "CourtCaseDocument"]
    PROPERTIES_PATH = args.properties_path
    TRAIN_DECISION_DATE = args.train_decision_date

    boost_params = {'n_estimators': 200,
 'min_samples_split': 40,
 'min_samples_leaf': 4,
 'max_features': 'sqrt',
 'max_depth': 20,
 'learning_rate': 0.05}
    boost = GradientBoostingRegressor(**boost_params)

    train_data = load_all_data(get_connection(PROPERTIES_PATH), TABLE_LIST, is_train=True, train_decision_date=TRAIN_DECISION_DATE)
    train_data = data_preprocessing(train_data)
    train_X, train_Y = train_data
    boost.fit(train_X, train_Y)

    print("training has been completed succesfully !!!!")
    print("--------------------------------------------")
    
    while True:
        test_id = input("enter CaseID to predict the duration ")
        TEST_ID = int(test_id)
		
        judge_id = input("enter JudgeID to predict the duration ")
        judge_id = int(judge_id)
        
        test_data = load_all_data(get_connection(PROPERTIES_PATH), TABLE_LIST, is_train=False, train_decision_date=None, test_id=TEST_ID)
        test_data = data_preprocessing(test_data)
        test_X, test_Y = test_data
		
        test_X["AssignedJudgeUserID"] = judge_id

        missing_cols = set(train_X.columns ) - set(test_X.columns )
        # Add a missing column in test set with default value equal to 0
        for c in missing_cols:
            test_X[c] = 0
        # Ensure the order of column in the test set is in the same order than in train set
        test_X = test_X[train_X.columns]
        y_pred = boost.predict(test_X)[0]

        print("predicted duration is %f days" % y_pred)
        print("actual duration is %f days" % test_Y)

if __name__ == '__main__':
    args = parse_args_train()
    main(args)
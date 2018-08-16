import os
import json
import pandas as pd
import numpy as np
import argparse
from scipy.stats import randint as sp_randint
from sklearn.metrics import accuracy_score, r2_score, mean_squared_log_error
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from mlxtend.regressor import StackingRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBRegressor
import xgboost as xgb
from xgboost import plot_importance
import pyodbc
from lightgbm.sklearn import LGBMRegressor
import lightgbm as lgb
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
TABLE_LIST = ["CourtCase", "CourtCaseParty", "CourtCaseSchedule", "CourtCaseCrimes", "CourtCasePartyLegalRepresentative", "CourtCaseDocument"]

def load_data(conn, table, is_train = True, test_id = None, load_from_disk = False):
	if is_train:
		if load_from_disk:
			return pd.read_pickle(table + '.pkl')	
	cursor = conn.cursor()       
	sql = 'exec dbo.load_' + table + ' @CourtCaseID = ?'
	print(sql, '----', test_id)
	cursor.execute(sql, (test_id))
	columns = np.array(cursor.description)[:, 0]
	rows = np.array(cursor.fetchall())	
	if table == 'CourtCase':
		assert len(rows) > 0, 'No data found for specified filters'
	data = pd.DataFrame(data=list(rows), columns=columns)
	if is_train:
		data.to_pickle(table + ".pkl")
	return data
	
def load_all_data(conn, tables, is_train = True, test_id = None, load_from_disk = False):
    try:
        assert is_train or test_id != None
    except AssertionError():
        raise AssertionError("test_id must be passed in training mode") 
    results = {}    
    for table in tables:
        results[table] = load_data(conn, table, is_train, test_id, load_from_disk = load_from_disk)
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

    CourtCaseParty["IsNPPA"] = CourtCaseParty.PartyInstanceID.apply(lambda x: 1 if x == -2 else 0)
    
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
    [CourtCase[x].fillna(0, inplace=True) for x in CourtCase.columns if x.startswith("DocumentTypeID")];
    CourtCase.CountOfLegalRepresentative.fillna(0, inplace=True)
    CourtCase.ccp_count.fillna(0, inplace=True)
    # CourtCase.cnt_admitem.fillna(0, inplace=True)
    # CourtCase.cnt_issues.fillna(0, inplace=True)
    CourtCase.is_nppa_present.fillna(0, inplace=True)        

    CourtCase["HasRecieptDocument"] = CourtCase.ReceiptDocumentID.fillna(0).apply(lambda x: 1 if x > 0 else 0)
    CourtCase["HasProsecutionCase"] = CourtCase.ProsecutionCaseInstanceID.fillna(0).apply(lambda x: 1 if x > 0 else 0)
    # CourtCase["IsAppealedcase"] = CourtCase.AppealedCourtCaseID.fillna(0).apply(lambda x: 1 if x > 0 else 0)
    CourtCase["ColorID"] = CourtCase.ColorID.fillna(-1)
    CourtCase["InstanceLevelID"] = CourtCase.InstanceLevelID.fillna(-1)
    CourtCase["ExtraOrdinaryProcedureID"] = CourtCase.ExtraOrdinaryProcedureID.fillna(-1)
    CourtCase["SpecialCaseID"] = CourtCase.SpecialCaseID.fillna(-1)
    CourtCase["SubCategoryID"] = CourtCase.SubCategoryID.fillna(-1)
    CourtCase["CasePriorityID"] = CourtCase.CasePriorityID.fillna(-1)
    CourtCase["IsDetentionCase"] = CourtCase.IsDetentionCase.fillna(0)
    CourtCase["IsPublicCase"] = CourtCase.IsPublicCase.fillna(0)
    CourtCase["CommittedByMinor"] = CourtCase.CommittedByMinor.fillna(0)
    CourtCase["GenderBasedViolence"] = CourtCase.GenderBasedViolence.fillna(0)
    CourtCase["InitiatedFromAbunzi"] = CourtCase.InitiatedFromAbunzi.fillna(0)
    CourtCase["SolvedFromAbunzi"] = CourtCase.SolvedFromAbunzi.fillna(0)
    CourtCase["AppealedCourtCaseInstanceID"] = CourtCase.AppealedCourtCaseInstanceID.fillna(-1)
    CourtCase["HasAppealCase"] = CourtCase.AppealedCourtCaseInstanceID.apply(lambda x: 0 if x == -1 else 1)
    
    
    #ProsecutionCaseInstanceID
    CourtCase["ProsecutionCaseInstanceID"] = CourtCase.ProsecutionCaseInstanceID.fillna(-1)
    CourtCase["HasProsecutionCase"] = CourtCase.ProsecutionCaseInstanceID.apply(lambda x: 0 if x == -1 else 1)
        
    CourtCase["ReceiptDocumentID"] = CourtCase.ReceiptDocumentID.fillna(-1)
    CourtCase["HasReceiptDocument"] = CourtCase.ReceiptDocumentID.apply(lambda x: 0 if x == -1 else 1)
    
    CourtCase["PreviousCourtCaseInstanceID"] = CourtCase.PreviousCourtCaseInstanceID.fillna(-1)
    CourtCase["HasPreviousCase"] = CourtCase.PreviousCourtCaseInstanceID.apply(lambda x: 0 if x == -1 else 1)
    #CourtCase["HasDetails"] = CourtCase.HasDetails.fillna(0)
    CourtCase["IsExempted"] = CourtCase.IsExempted.fillna(0)
#     CourtCase["Distance"] = CourtCase["DateCreated"] - CourtCase["min_hearingdate"]
    CourtCase["AttachedDate"] = CourtCase.AttachedDate.fillna(0)
    
    CourtCase["CountryID"] = CourtCase.CountryID.fillna(-1)
    CourtCase["ProvinceID"] = CourtCase.ProvinceID.fillna(-1)
    CourtCase["DistrictID"] = CourtCase.DistrictID.fillna(-1)
    CourtCase["ParentGroupID"] = CourtCase.DistrictID.fillna(-1)
    CourtCase["avg_size"] = CourtCase.avg_size.fillna(0)
    CourtCase["total_size"] = CourtCase.total_size.fillna(0)
    CourtCase.drop(columns=["CourtCaseID", "PublicOwnerUserId","CaseRejectionID", "CourtCaseInstanceID", "PaymentBankID", "ReceiptNumber", "RejectionDetails"], inplace=True)    

    
    CourtCase['FillingFee'] = pd.to_numeric(CourtCase['FillingFee'])    
    
    assert len(CourtCase) > 0, 'No data left after preprocessing'
    
#     CourtCase = CourtCase[CourtCase["DecisionDuration"] < 30]
    
#     train_data = CourtCase[CourtCase['DecisionEndDate'] <= 43294]
#     test_data = CourtCase[(CourtCase['DecisionEndDate'] > 43294)]
    
#     train_data["DecisionDuration"] = train_data["DecisionDuration"].apply(lambda x: 60 if x>= 60 else x)
    
#     train_X = train_data.drop(columns=["DecisionDuration", "DecisionEndDate"])
#     train_Y = train_data["DecisionDuration"]
    
#     test_X = test_data.drop(columns=["DecisionDuration", "DecisionEndDate"])
#     test_Y = test_data["DecisionDuration"]

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
	parser.add_argument('--properties_path'
						,default = '\db_properties\db_connection.json'
						,help='relative path to database connection properties')
	parser.add_argument('--load_model', '-l'
						,help='load model from disk')
	parser.add_argument('--load_from_disk', '-ld'
						, help='load data from disk')
	args = parser.parse_args(*argument_array)
	return args

def init(PROPERTIES_PATH, LOAD_FROM_DISK):
	
	
#	boost_params = {'n_estimators': 200,
# 'min_samples_split': 40,
# 'min_samples_leaf': 4,
# 'max_features': 'sqrt',
# 'max_depth': 20,
# 'learning_rate': 0.05}
# 
#	boost = GradientBoostingRegressor(**boost_params)

	boost = LGBMRegressor(
 learning_rate =0.05,
 n_estimators=1127,
 max_depth=-1,
 min_child_weight=0,
 num_leaves = 68,
 min_child_samples = 5,
 objective= 'regression',
 subsample_for_bin = 1000,
 min_split_gain = 0,
 feature_fraction = 0.5, 
 nthread=-1
)
	train_data = load_all_data(get_connection(PROPERTIES_PATH), TABLE_LIST, is_train=True, load_from_disk = LOAD_FROM_DISK)
	train_data = data_preprocessing(train_data)
	train_X, train_Y = train_data
	boost.fit(train_X, train_Y)
	np.save('col.npy', train_X.columns)

	print("training has been completed succesfully !!!!")
	print("--------------------------------------------")
	
	return boost;
	
def predict(boost, caseId, judgeId, properties_path):
    train_col = np.load('col.npy')


    #test_id = input("enter CaseID to predict the duration ")
    TEST_ID = caseId #int(test_id)
    
    #judge_id = input("enter JudgeID to predict the duration ")
    judge_id = judgeId # int(judge_id)
    
    test_data = load_all_data(get_connection(properties_path), TABLE_LIST, is_train=False, test_id=TEST_ID)
    test_data = data_preprocessing(test_data)
    test_X, test_Y = test_data    
    
    test_X["AssignedJudgeUserID"] = judge_id

    missing_cols = set(train_col) - set(test_X.columns )
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        test_X[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    test_X = test_X[train_col]
    y_pred = boost.predict(test_X)[0]
    
    if test_Y.loc[0] == None:
        print("actual duration is unknown")
    else:
        print("actual duration is %f days" % test_Y)
    print("predicted duration is %f days" % y_pred)
    
    return y_pred
	
def main(args):
	properties_path = args.properties_path
	laod_from_disk = args.load_from_disk
	
	if args.load_model:
		boost = joblib.load('model.pkl')
	else:		
		boost = init(properties_path, laod_from_disk)
		joblib.dump(boost, 'model.pkl')
	
	while True:
		test_id = int(input("enter CaseID to predict the duration "))
		judge_id = int(input("enter JudgeID to predict the duration "))
		predict(boost, test_id, judge_id, properties_path)
if __name__ == '__main__':
	args = parse_args_train()
	main(args)
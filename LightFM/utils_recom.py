import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import MinMaxScaler


def load_data(train_filename, test_filename, user_filename, ques_filename):
    train_data = pd.read_csv(train_filename)
    test_data = pd.read_csv(test_filename)
    user_data = pd.read_csv(user_filename)
    ques_data = pd.read_csv(ques_filename)

    del train_data['Unnamed: 0']
    del test_data['Unnamed: 0']
    del user_data['Unnamed: 0']
    del ques_data['Unnamed: 0']

    num_users = len(user_data)
    num_ques = len(ques_data)
    train = sp.lil_matrix((num_users,num_ques), dtype=np.int32)

    for index, row in train_data.iterrows():
        ans = row['answered']
        train[row['expert_no'],row['ques_no']] = 1 if ans == 1 else 0
    train = train.tocoo()

    user_feat_mat = sp.lil_matrix((num_users,143), dtype=np.float32)
    ques_feat_mat = sp.lil_matrix((num_ques,23), dtype=np.float32)

    for i,row in enumerate(user_data.as_matrix()):
        user_info = row[4:]
        for index, val in zip(range(143),user_info):
            if val == 1:
                user_feat_mat[i,index] = 1

    for j,row in enumerate(ques_data.as_matrix()):
        ques_info = list(row[2:22])+list(row[27:])
        for index, val in zip(range(23),ques_info):
            if val > 0:
                ques_feat_mat[j,index] = val

    test_user_ids = np.asarray(list(test_data['expert_no']))
    test_ques_ids = np.asarray(list(test_data['ques_no']))

    return train, user_feat_mat, ques_feat_mat,test_user_ids,test_ques_ids


def generate_result_file(output_filename, test_filename,predictions):
    min_max_scaler = MinMaxScaler()
    predictions_scaled = min_max_scaler.fit_transform(predictions)
    actual_test_data  = pd.read_table('validate_nolabel.txt',names=['ques_id', 'expert_id'],sep = ',',skiprows = 1)
    test_data = pd.read_csv(test_filename)
    del test_data['Unnamed: 0']
    result_df = test_data[['ques_id', 'expert_id']].copy()
    prob_list = list(predictions_scaled)
    result_df['label'] = prob_list
    final_result = actual_test_data.merge(result_df, on=['ques_id','expert_id'])
    final_result.to_csv(path_or_buf='result.csv')





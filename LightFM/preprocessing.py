import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def read_data():
    ques_data = pd.read_table('question_info.txt',
                            names=['ques_id', 'ques_tag', 'word_id_seq', 'char_id_seq', 'no_of_ans', 'upvotes',
                                   'top_qual_ans'], sep = '\t')
    user_data = pd.read_table('user_info.txt', names=['expert_id', 'tags', 'word_id_seq', 'char_id_seq'], sep = '\t')
    invited_info_train = pd.read_table('invited_info_train.txt', names=['ques_id', 'expert_id', 'answered'], sep = '\t')
    test_data = pd.read_table('validate_nolabel.txt',
                            names=['ques_id', 'expert_id'],sep = ',',skiprows = 1)
    return ques_data, user_data, invited_info_train, test_data

def create_train_test_files():
    ques_data, user_data, invited_info_train, test_data = read_data()
    ques_data['ques_no'] = range(len(ques_data))
    exp_ques_data = pd.get_dummies(ques_data.ques_tag)
    exp_ques_data.insert(0, 'ques_no', ques_data.ques_no)
    exp_ques_data.insert(1, 'ques_id', ques_data.ques_id)
    exp_ques_data['no_of_ans'] = ques_data.no_of_ans
    exp_ques_data['upvotes'] = ques_data.upvotes
    exp_ques_data['top_qual_ans'] = ques_data.top_qual_ans
    exp_ques_data = exp_ques_data.rename(columns={})
    q_tags = [{k: 'qT' + str(k)} for k in range(20)]
    q_tags_col_names = {}
    for map_name in q_tags:
        q_tags_col_names[map_name.keys()[0]] = map_name.values()[0]
    exp_ques_data = exp_ques_data.rename(columns=q_tags_col_names)
    exp_ques_data['word_id_seq'] = ques_data.word_id_seq
    exp_ques_data['char_id_seq'] = ques_data.char_id_seq

    user_tags = map(lambda x: map(lambda y: int(y), x.split('/')), user_tags)
    tag_set = set()
    map(lambda x: map(lambda y: tag_set.add(y), x), user_tags)
    num_tags = len(tag_set)
    user_tags = user_data.tags
    tag_feature_vec = []
    tags = range(num_tags)
    for tag_list in user_tags:
        tag_feature_vec.append([1 if tag in tag_list else 0 for tag in tags])
    for i, col in enumerate(zip(*tag_feature_vec)):
        col_name = 'T' + str(i)
        user_data[col_name] = col
    user_data = user_data.drop('tags', 1)
    user_data.insert(0, 'expert_no', range(len(user_data)))
    exp_user_data = user_data
    expert_word_features = map(lambda x: x.split('/'), exp_user_data.word_id_seq)
    ques_word_features = map(lambda x: x.split('/'), ques_data.word_id_seq)

    # words = set()
    # for word_seq in expert_word_features:
    #     for word in word_seq:
    #         words.add(word)
    # for word_seq in ques_word_features:
    #     for word in word_seq:
    #         words.add(word)
    # expert_word_features = map(lambda x: map(lambda y: int(y) if y else 0, x), expert_word_features)
    # ques_word_features = map(lambda x: map(lambda y: int(y) if y else 0, x), ques_word_features)
    # expert_word_features_vec = []
    # for word_seq_list in expert_word_features:
    #     expert_word_features_vec.append([1 if word in word_seq_list else 0 for word in words])
    # ques_word_features_vec = []
    # for word_seq_list in ques_word_features:
    #     ques_word_features_vec.append([1 if tag in tag_list else 0 for word in words])

    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(exp_ques_data.no_of_ans)
    exp_ques_data['no_of_ans_norm'] = x_scaled
    x_scaled = min_max_scaler.fit_transform(exp_ques_data.top_qual_ans)
    exp_ques_data['top_qual_ans_norm'] = x_scaled
    exp_user_data = exp_user_data.rename(
        columns={'word_id_seq': 'expert_word_id_seq', 'char_id_seq': 'expert_char_id_seq'})
    exp_ques_data = exp_ques_data.rename(columns={'word_id_seq': 'ques_word_id_seq', 'char_id_seq': 'ques_char_id_seq'})
    exp_user_data.to_csv(path_or_buf='exp_user_data.csv')
    exp_ques_data.to_csv(path_or_buf='exp_ques_data.csv')
    train_data = invited_info_train.merge(exp_user_data, on='expert_id', how='inner')
    train_data = train_data.merge(exp_ques_data, on='ques_id', how='inner')
    train_data.to_csv(path_or_buf='train_data.csv')
    exp_test_data = test_data.merge(exp_user_data, on='expert_id', how='inner')
    exp_test_data = exp_test_data.merge(exp_ques_data, on='ques_id', how='inner')
    exp_test_data.to_csv(path_or_buf='test_data.csv')

def clean_train_test_files():
    train_data = pd.read_csv('train_data.csv')
    test_data = pd.read_csv('test_data.csv')
    del train_data['Unnamed: 0']
    del train_data['expert_word_id_seq']
    del train_data['expert_char_id_seq']
    del train_data['ques_word_id_seq']
    del train_data['ques_char_id_seq']
    del train_data['no_of_ans']
    del train_data['upvotes']
    del train_data['top_qual_ans']
    del test_data['Unnamed: 0']
    del test_data['expert_word_id_seq']
    del test_data['expert_char_id_seq']
    del test_data['ques_word_id_seq']
    del test_data['ques_char_id_seq']
    del test_data['no_of_ans']
    del test_data['upvotes']
    del test_data['top_qual_ans']
    train_data.to_csv(path_or_buf='train_data_clean.csv')
    test_data.to_csv(path_or_buf='test_data_clean.csv')


if __name__ == '__main__':
    create_train_test_files()
    clean_train_test_files()





















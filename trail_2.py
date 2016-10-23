import graphlab as gl
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Method to convert tags into tag list associated with an expert id
def get_tags(tags):
    month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    tags = tags.split('/')
    user_tags = []
    for tag in tags:
        if len(tag.split('-')) > 1:
            a, b = tag.split('-')
            if a in month:
                user_tags.append(int(b))
            else:
                user_tags.append(int(a))
        else:
            user_tags.append(int(tag))
    return user_tags


def read_data():
    ques_data = pd.read_csv('question_info.csv',
                            names=['ques_id', 'ques_tag', 'word_id_seq', 'char_id_seq', 'no_of_answers', 'upvotes',
                                   'top_qual_ans'])
    user_data = pd.read_csv('user_info.csv', names=['expert_id', 'tags', 'word_id_seq', 'char_id_seq'])
    invited_info_train = pd.read_csv('invited_info_train.csv', names=['ques_id', 'user_id', 'answered'])
    test_data = pd.read_csv('validate_nolabel.csv',
                            names=['qid', 'uid', 'lable'])
    return ques_data, user_data, invited_info_train, test_data


def get_user_tags_list_dict(user_data):
    # Dict containing expert_id : list of tags
    user_tags_list_dict = {}
    for index, row in user_data.iterrows():
        tags = row['tags']
        expert_id = row['expert_id']
        user_tags_list_dict[expert_id] = get_tags(tags)
    return user_tags_list_dict


def get_tag_ques_dict(ques_data):
    # Dict containig tag_id : list of associated questions
    tag_ques_dict = {}
    for index, row in ques_data.iterrows():
        if row['ques_tag'] in tag_ques_dict:
            tag_ques_dict[row['ques_tag']].append(row['ques_id'])
        else:
            tag_ques_dict[row['ques_tag']] = []
            tag_ques_dict[row['ques_tag']].append(row['ques_id'])
    return tag_ques_dict


def get_tag_set(user_tags_list_dict):
    # Set of all different tags
    tag_set = set()
    for tag_list in user_tags_list_dict.values():
        tag_set = tag_set.union(set(tag_list))
    return tag_set


def get_user_tag(invited_info_train):
    # Dict containing expert_id : {dict containing tag_id : number of questions answered by the expert_id}
    user_tag = {}
    for index, row in invited_info_train.iterrows():
        ques_tag = ques_data[ques_data['ques_id'] == row['ques_id']]['ques_tag'].iloc[0]
        if row['user_id'] in user_tag:
            if ques_tag in user_tag[row['user_id']]:
                user_tag[row['user_id']][ques_tag] += 1
            else:
                user_tag[row['user_id']][ques_tag] = 1
        else:
            user_tag[row['user_id']] = {}
            user_tag[row['user_id']][ques_tag] = 1
    return user_tag


def get_user_tag_prob(user_tag):
    # Dict containing expert_id : {dict containing tag_id : probability of expert_id answering the question }
    # Formula - # of questions of that tag answered/ # of total questions of that tag appeared
    user_tag_prob = {}
    for expert_id, tag_dict in user_tag.items():
        user_tag_prob[expert_id] = {}
        for tag, answered in tag_dict.items():
            total_tag_questions = len(set(tag_ques_dict[tag]))
            user_tag_prob[expert_id][tag] = user_tag[expert_id][tag] / float(total_tag_questions)
    return user_tag_prob


if __name__ == '__main__':

    ques_data, user_data, invited_info_train, test_data = read_data()
    print test_data.head()
    user_tags_list_dict = get_user_tags_list_dict(user_data)
    tag_ques_dict = get_tag_ques_dict(ques_data)
    tag_set = get_tag_set(user_tags_list_dict)
    user_tag = get_user_tag(invited_info_train)
    user_tag_prob = get_user_tag_prob(user_tag)

    # Doubling the score/prob for tags that appeared in expert's list of associated tags
    for expert_id, tag_dict in user_tag_prob.items():
        for tag, prob in tag_dict.items():
            if tag in user_tags_list_dict[expert_id]:
                user_tag_prob[expert_id][tag] *= 2

    # Handling the missing expert_ids, tags
    for expert_id, tag_list in user_tags_list_dict.items():
        if expert_id not in user_tag_prob:
            user_tag_prob[expert_id] = {}
            for tag in tag_list:
                user_tag_prob[expert_id][tag] = 0.00001
        else:
            for tag in tag_list:
                if tag not in user_tag_prob[expert_id] and tag in tag_ques_dict:
                    user_tag_prob[expert_id][tag] = 0.000001
                elif tag not in user_tag_prob[expert_id]:
                    user_tag_prob[expert_id][tag] = 0.0001

    df_user_tag_prob = pd.DataFrame(columns=['expert_id', 'prob', 'tag_id'])
    for expert_id, prob_dict in user_tag_prob.items():
        if bool(prob_dict):
            for tag_id, prob in prob_dict.items():
                df_user_tag_prob = df_user_tag_prob.append({'expert_id': expert_id, 'tag_id': tag_id, 'prob': prob},
                                                           ignore_index=True)
    df_user_tag_prob['tag_id'] = df_user_tag_prob['tag_id'].fillna(0.0).astype(int)

    train_data = gl.SFrame(df_user_tag_prob)
    m = gl.ranking_factorization_recommender.create(train_data,
                                                    user_id='expert_id',
                                                    item_id='tag_id',
                                                    target='prob',
                                                    solver='ials')

    # recoms = m.recommend(users=list(user_data['expert_id']), k=143)

    recoms = m.recommend(users=list(test_data['uid']), k=143)

    result_dict = []
    for index, row in test_data.iterrows():
        temp = recoms[recoms['expert_id'] == row['uid']]
        tag_id = ques_data[ques_data['ques_id'] == row['qid']]['ques_tag'].iat[0]
        try:
            recs = temp[temp['tag_id'] == tag_id]['score']
            if len(recs):
                score = temp[temp['tag_id'] == tag_id]['score'][0]
                result_dict.append((row['qid'], row['uid'], score))
            else:
                recs = df_user_tag_prob[df_user_tag_prob['expert_id'] == row['uid']]
                prob = recs[recs['tag_id'] == tag_id]['prob'].iat[0]
                result_dict.append((row['qid'], row['uid'], prob))
        except Exception as e:
            result_dict.append((row['qid'], row['uid'], 0))

    result_df = pd.DataFrame.from_records(data=result_dict, columns=['qid', 'uid', 'prob'])
    min_max_scaler = MinMaxScaler()
    result_df[['prob']] = min_max_scaler.fit_transform(result_df[['prob']])
    result_df.to_csv(path_or_buf='result.csv')

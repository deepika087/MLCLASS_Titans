from lightfm import LightFM
from utils_recom import *
from lightfm.evaluation import auc_score


train = 'train_data_clean.csv'
test = 'test_data_clean.csv'
output = 'result.csv'
user = 'exp_user_data.csv'
ques = 'exp_ques_data.csv'

if __name__ == '__main__':
    NUM_THREADS = 4
    NUM_COMPONENTS = 300
    NUM_EPOCHS = 50
    ITEM_ALPHA = 1e-6
    USER_ALPHA = 1e-6

    train, user_feat_mat, ques_feat_mat,test_user_ids,test_ques_ids = load_data(train_filename=train, test_filename=test, user_filename=user, ques_filename=ques)
    model = LightFM(no_components=NUM_COMPONENTS, loss='bpr', item_alpha=ITEM_ALPHA)


    #Traing with training data
    # model.fit(train, epochs=NUM_EPOCHS, num_threads=NUM_THREADS)
    # train_auc=auc_score(model, train, num_threads=NUM_THREADS).mean()
    # print('Collaborative filtering train AUC: %s' % train_auc)

    #Training with training data and item, user features
    model.fit(train, epochs=NUM_EPOCHS,user_features=user_feat_mat,item_features=ques_feat_mat, num_threads=NUM_THREADS)
    # train_auc = auc_score(model,
    #                       train,
    #                       item_features=ques_feat_mat,
    #                       user_features=user_feat_mat,
    #                       num_threads=NUM_THREADS).mean()
    #print('Hybrid training set AUC: %s' % train_auc)

    predictions = model.predict(test_user_ids, test_ques_ids,user_features=user_feat_mat, item_features=ques_feat_mat, num_threads=4)
    generate_result_file(output, test, predictions)









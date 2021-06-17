import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC

def eva_Kmeans(x, y, k=3, time=10):
    x = np.array(x)
    x = np.squeeze(x)
    y = np.array(y)

    if len(y.shape) > 1:
        y = np.argmax(y, axis=1)

    estimator = KMeans(n_clusters=k)
    ari_list = []
    nmi_list = []
    for i in range(time):
        estimator.fit(x, y)
        y_pred = estimator.predict(x)
        nmi_score = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        ari_score = adjusted_rand_score(y, y_pred)
        ari_list.append(ari_score)
        nmi_list.append(nmi_score)
    print('NMI (10 avg): {:.4f} , ARI (10avg): {:.4f}'.format(np.mean(nmi_list), np.mean(ari_list)))


def eva_SVM(x, y, split_list=[0.2, 0.4, 0.6, 0.8], time=10):
    x = np.array(x)
    x = np.squeeze(x)
    y = np.array(y)

    if len(y.shape) > 1:
        y = np.argmax(y, axis=1)
    for split in split_list:
        micro_list = []
        macro_list = []
        for i in range(time):
            random_states = 182318 + i
            train_x, test_x, train_y, test_y = train_test_split(
                x, y, test_size=1-split, shuffle=True, random_state=random_states)

            svm = LinearSVC(dual=False)
            svm.fit(train_x, train_y)
            y_pred = svm.predict(test_x)
            macro_f1 = f1_score(test_y, y_pred, average='macro')
            micro_f1 = f1_score(test_y, y_pred, average='micro')
            macro_list.append(macro_f1)
            micro_list.append(micro_f1)

        print('SVM({}avg, split:{}) f1_macro: {:.4f}, f1_micro: {:.4f}'.format(
            time, split, np.mean(macro_list), np.mean(micro_list)))
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering as Agg
from scipy.cluster.hierarchy import dendrogram
from sklearn import preprocessing
import re

st.title("Wine Data Analysis")

uploaded_samples = st.file_uploader("Choose a file", key=1)
flag_samples = False
if uploaded_samples is not None:
    flag_samples = True
    # df1=pd.read_excel('https://cs.westminstercollege.edu/~jingsai/project/Wine%20data%20from%202020%20Chem%20307,%202-6-22.xlsx')
    df1=pd.read_excel(uploaded_samples)
    df1.set_index(df1.columns[0], inplace=True) # move sample ID as row name
    df1.columns = df1.columns.str.extract("\[(\w+)\]", expand=False)
    # df1.head()

    st.subheader('Samples')
    st.write(df1)

uploaded_tests= st.file_uploader("Choose a file", key=2)
flag_tests = False
if uploaded_tests is not None:
    flag_tests = True
    # df2=pd.read_excel('Wine data from 2022 Chem 307, 2-27-22.xlsx')
    df2=pd.read_excel(uploaded_tests)
    df2.set_index(df2.columns[0], inplace=True) # move sample ID as row name
    df2.columns = df2.columns.str.extract("\[(\w+)\]", expand=False)
    df2.index = "Test" + df2.index

    st.subheader('Tests')
    st.write(df2)

def find_group(label,plot=False):
    df = pd.concat([df1, df2.loc[[label],:]]).dropna(axis=1)
    # print(df2.loc[[label],:])
    # print(df)
    sample_name = df.index

    # print(df.shape[0], "wine samples and", df.shape[1], "elements.")
    data = df.values
    data = preprocessing.StandardScaler().fit_transform(data)

    df['ID']=list(range(df.shape[0]))

    model = Run_Agg_Clustering(n_clusters = None, data=data, distance_threshold=0)

    labels = find_labels(label, model, df)
    
    if plot:
        Plot_Dendrogram(model, 
                    truncate_mode=None,
                    fig_size = (20,10), 
                    leaf_font_size = 12, 
                    label_font_size = 15, 
                    labels = sample_name)

        # Show_Clusters(model, sample_name)

    return labels

def find_labels(label, model, df):
    data = model.children_
    n_samples = len(data)+1
    # print(df)
    index = df.loc[label,'ID']
    row = np.argwhere(data == index)[0]
    # print(index)
    pair = data[row[0], 1-row[1]]
    groups = [pair]
    current, total = 1, 1
    while current <= total:
        row = groups[current-1]
        if row >= n_samples:
            groups.extend(data[row-n_samples])
            total += 2         
        current += 1
        
    groups = np.array(groups)
    groups_leaf = groups[groups<n_samples]
    leaves = [df.index[df['ID'] == leaf] for leaf in groups_leaf]
    return [list(leaves[i])[0] for i in range(len(leaves))]

def Run_Agg_Clustering(n_clusters,data,  **kwargs):
    model = Agg(linkage="average", n_clusters = n_clusters, **kwargs)
    model = model.fit(data)
    return model

def Plot_Dendrogram(model, fig_size, label_font_size, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    fig = plt.figure(figsize=fig_size)
    
    # dendrogram(linkage_matrix, color_threshold = 6, **kwargs)
    dendrogram(linkage_matrix, **kwargs)
    
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Number of wines in node (or name of wine if no parenthesis).", fontsize = label_font_size)
    # plt.show()
    st.pyplot(fig)

def Show_Clusters(model, sample_name):
    labels = model.labels_
    result = pd.DataFrame(labels, index = sample_name, columns=["Group"])
    result.reset_index(inplace=True)
    result.sort_values(by=['Group', 'Sample ID'], inplace=True)
    result.set_index('Sample ID', inplace=True)
    # result['index'] = result.index
    # result.sort_values(by=['Group', 'index'], inplace=True)
    # result.drop('index', axis=1, inplace=True)
    # result.index.name = 'index'
    return result

if flag_samples and flag_tests:

    menu = list(df2.index)
    menu.insert(0, '<Select>')
    option = st.selectbox(
     'Please select a wine to test:',
     menu, index=0)

    if option != '<Select>':
        st.write('You selected:', option)
        group = find_group(option, True)
        st.write(group)

    # groups = []
    # for index in df2.index:
    #     # print(index)
    #     groups.append(find_group(index))
        
    # st.write(pd.Series(groups, index=df2.index))
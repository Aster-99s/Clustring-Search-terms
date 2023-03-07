import pandas as pd
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

input_file = input("Please enter the name of the input file: ")

### Settings
cluster_accuracy = 83
min_cluster_size = 2
word_in_cluster_name = 3
output_file ='clustred_' + input_file 

### Reading input file and storing data
df = pd.read_excel(input_file)
corpus_set_all = set(df['Search term'])
cluster = True


### choosing a sentence transformer
transformer = 'gtr-t5-large'


### Filtering out stop words
stop_words = set(stopwords.words('english'))
corpus_set = set()
for sentence in corpus_set_all:
    sentence_list = sentence.split()
    filtered_sentence = []
    for word in sentence_list:
        if word not in stop_words:
            filtered_sentence.append(word)
    filtered_sentence = TreebankWordDetokenizer().detokenize(filtered_sentence)
    corpus_set.add(filtered_sentence)


### Clustering keywords
model = SentenceTransformer(transformer)
corpus_sentences_list = []
cluster_name_list = []
df_all = []

while cluster:
    corpus_sentences = list(corpus_set)     # Copying the list of sentences
    check_len = len(corpus_sentences)
    corpus_embeddings = model.encode(corpus_sentences, batch_size=256, show_progress_bar=True, convert_to_tensor=True)
    clusters = util.community_detection(corpus_embeddings, min_community_size=min_cluster_size, threshold=cluster_accuracy/100) #<-- Threshold
    
    for keyword, cluster in enumerate(clusters):                 # Loop through clusters
        for sentence_id in cluster[0:]:                           # Loop through sentences
            if len(word_tokenize(corpus_sentences[sentence_id])) >= word_in_cluster_name:                   # Filter sentences
                corpus_sentences_list.append(corpus_sentences[sentence_id])             # Build corpus 
                cluster_name_list.append("Cluster {}, #{} Elements ".format(keyword + 1, len(cluster)))      # Build cluster names
    df_new = pd.DataFrame(None)
    df_new['Cluster Name'] = cluster_name_list
    df_new["Search term"] = corpus_sentences_list
    df_all.append(df_new)
    have = set(df_new['Search term'])                               # Which clusters were already used?
    corpus_set = corpus_set_all - have                              # What's left?
    remaining = len(corpus_set)
    if check_len == remaining:
        break


### Creating final dataframe and writing to output file
df_new = pd.concat(df_all)
df = df.merge(df_new.drop_duplicates('Search term'), how='left', on='Search term')
df['Length'] = df['Search term'].astype(str).map(len)
df = df.sort_values(by="Length", ascending=True)
df['Cluster Name'] = df.groupby('Cluster Name')['Search term'].transform('first')
df.sort_values(['Cluster Name', 'Search term'], ascending=[True, True], inplace=True)
df['Cluster Name'] = df['Cluster Name'].fillna("No Cluster Assigned")
df = df[['Cluster Name', 'Search term']]
df.to_excel(output_file, index=False)
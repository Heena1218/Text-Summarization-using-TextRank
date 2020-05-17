import os
import io
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

from nltk.cluster.util import cosine_distance

UPLOAD_FOLDER = "/Files"

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def student():
   return render_template('index.html')

@app.route('/',methods = ['POST'])
def result():
   if request.method == 'POST':
       f = request.files['file']
       number = request.form['number']
       number = int(number)
       f.save(secure_filename(f.filename))
       name = f.filename
       stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
 
       raw_text = convert_pdf_to_txt(name)
       sentences = data_preprocessing(raw_text)
       #after text cleaning
       clean_Sentences = text_cleaning(sentences)
       sentence_similarity_matrix = build_similarity_matrix(clean_Sentences,stopwords)
       sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
       scores = nx.pagerank(sentence_similarity_graph)
       ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
 

       #summary = summary_generation(clean_Sentences,sentences)


       os.remove(secure_filename(f.filename))
       return render_template("done.html",result = ranked_sentences,number = number)

# ------ function to convert pdf to text ------ #
def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    laparams = LAParams()
    with io.StringIO() as retstr:
        with TextConverter(rsrcmgr, retstr, codec = 'utf-8',
                           laparams=laparams) as device:
            with open(path, 'rb') as fp:
                interpreter = PDFPageInterpreter(rsrcmgr, device)
                password = ""
                maxpages = 0
                caching = True
                pagenos = set()
                for page in PDFPage.get_pages(fp,
                                              pagenos,
                                              maxpages=maxpages,
                                              password=password,
                                              caching=caching,
                                              check_extractable=True):
                    interpreter.process_page(page)
                return retstr.getvalue()

# -----function to clean the data -------- #
def data_preprocessing(input):
    text = "".join(input)
    text = text.replace('\n','')
    text = sent_tokenize(text)
    #--- removing sentences having length greater than 40 words ---- #
    Cleaned_Sentences = []
    for row in text:
        length = 0
        text_split = row.split()
        for word in text_split:
            length +=1
        if length < 40:
            Cleaned_Sentences.append(row)
    #---- removing sentences having word length greater than 25 ----#
    sentence_after_cleaning = []
    for row in Cleaned_Sentences:
        row_split = row.split()
        for word in row_split:
            length = len(word)
            if length > 25:
                sentence_after_cleaning.append(row)
                break
    sentences = list(set(Cleaned_Sentences)-set(sentence_after_cleaning))
    #---- removing sentences having length less than 10 words ----#
    Cleaned_Sentences_After = []
    for row in sentences:
        length=0
        text_split= row.split()
        for word in text_split:
            length +=1
        if length < 10:
            Cleaned_Sentences_After.append(row)
    sentences = list(set(sentences)-set(Cleaned_Sentences_After))
    return sentences

def remove_stopwords(sen):
    stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

def text_cleaning(sentences):
    #sentences = [y for x in sentences for y in x] # flatten list
    sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
    sentences = [s.lower() for s in sentences]
    
    return sentences

def sentence_similarity(sent1, sent2,stopwords):
    if stopwords is None:
        stopwords = []
    
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
            
    return similarity_matrix

def summary_generation(clean_sentences,sentences):
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
    file_id = '1OEiJv9OzIvAXy8BCqHDX__iuNFJKOzOU'
    downloaded = drive.CreateFile({'id': file_id})
    downloaded.GetContentFile('glove.6B.100d.txt')
    word_embeddings = {}
    f = open('glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    sentences_vectors = []
    for i in clean_sentences:
        if len(i)!=0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentences_vectors.append(v)
    # similarity matrix
    sim_mat = np.zeros([len(clean_sentences),len(clean_sentences)])
    
    for i in range(len(clean_sentences)):
        for j in range(len(clean_sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentences_vectors[i].reshape(1,100), sentences_vectors[j].reshape(1,100))[0,0]
    
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    return ranked_sentences
    
if __name__ == '__main__':
   app.run(debug = True)
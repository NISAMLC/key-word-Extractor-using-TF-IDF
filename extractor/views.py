from django.shortcuts import render
import pickle
from .sustain import countVectorizer

# Create your views here.
def handler(request):
    result  ={'No data sent':None}

    if request.method =='POST':
        sequence = request.POST['Name']
        result = get_keywords_text(sequence)
    else:
        pass
    return render(request,'index.html',{'response':result.keys()})


def get_keywords_text(docs):
    # generate tf-idf for the given document
    tfidf_transformer =  pickle.load(open('extractor/keywords-tfidf-model.pkl','rb'))
    tf_idf_vector = tfidf_transformer.transform(countVectorizer.transform([docs]))

    # sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(tf_idf_vector.tocoo())

    # extract only the top n; n here is 10
    feature_names  =pickle.load(open('extractor/keywords-feature-names.pkl','rb'))
    keywords = extract_topn_from_vector(feature_names, sorted_items, 10)

    return keywords

def sort_coo(coo_matrix):
  tuples = zip(coo_matrix.col, coo_matrix.data)
  return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def sort_coo(coo_matrix):
  tuples = zip(coo_matrix.col, coo_matrix.data)
  return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results
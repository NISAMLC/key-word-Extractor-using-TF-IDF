from django.core.cache import cache
import pickle

model_cache_key = 'model_cache'
countVectorizer = cache.get(model_cache_key)

if countVectorizer is None:
    countVectorizer  = pickle.load(open('extractor/keywords-count-vectorizer.pkl','rb'))
    cache.set(model_cache_key,countVectorizer,None)
import faiss
import numpy as np
import pandas as pd


df = pd.read_pickle("df.pkl")
index_map = pd.Series(df.index, index=df['track_name'])

def recommendations(song, faiss_idx, vectors):
  try:
    idx = index_map[song]
    
    query = vectors[idx:idx+1].astype('float32')

    sim_scores, top_indices = faiss_idx.search(query, 11)

    top_sim_indices = top_indices[0][1:]
    sim_scores = sim_scores[0][1:]
    
    return top_sim_indices, sim_scores
  except KeyError:
    print("song not in dataset")
    return [], []

euc_vectors = np.load('euc_vectors.npy')
index_euc = faiss.IndexFlatL2(euc_vectors.shape[1])
index_euc.add(euc_vectors)

def rec2(song):
  try:
    idx = index_map[song]
    query = euc_vectors[idx:idx+1].astype('float32')

    dist_scores, top_indices = index_euc.search(query, 11)

    top_indices = top_indices[0][1:]
    top_dist_scores = dist_scores[0][1:]
    
    res = df[['track_name', 'artist_name', 'genre']].iloc[top_indices].copy()
    res['euc_score'] = top_dist_scores

    return res.index, top_dist_scores

  except KeyError:
    print("song not in dataset")
    return [], []
  
def normal_scores(scores, invert=False):
  mx, mn = max(scores), min(scores)
  if mx == mn:
    return np.ones_like(scores)
  normalized = (scores - mn) / (mx - mn)
  if invert:
    return 1 - normalized
  else:
    return normalized
  
tfidf_idx = faiss.read_index("tfidf_idx.index")
cnt_idx = faiss.read_index("cnt_idx.index")
tfidf_vectors = np.load("tfidf_vectors.npy")
cnt_vectors = np.load("cnt_vectors.npy")
  
def getRecommendation(song, tf_weight = 0.25, cnt_weight=0.25, euc_weight=0.5):
  ind1, tf_scores = recommendations(song, faiss_idx=tfidf_idx, vectors=tfidf_vectors)
  ind2, cnt_scores = recommendations(song, faiss_idx=cnt_idx, vectors=cnt_vectors)
  ind3, euc_scores = rec2(song)
  
  tf_scores = normal_scores(tf_scores)
  cnt_scores = normal_scores(cnt_scores)
  euc_scores = normal_scores(euc_scores, invert=True)
  
  d = {}
  for i in range(len(ind1)):
    if ind1[i] not in d:
      d[ind1[i]] = tf_scores[i] * tf_weight
    else:
      d[ind1[i]] += tf_scores[i] * tf_weight
      
  for i in range(len(ind2)):
    if ind2[i] not in d:
      d[ind2[i]] = cnt_scores[i] * cnt_weight
    else:
      d[ind2[i]] += cnt_scores[i] * cnt_weight
  
  for i in range(len(ind3)):
    if ind3[i] not in d:
      d[ind3[i]] = euc_scores[i] * euc_weight
    else:
      d[ind3[i]] += euc_scores[i] * euc_weight
  d = dict(sorted(d.items(), key=lambda item: item[1], reverse=True))
  top_idx = list(d.keys())[:10]
  res = df[['artist_name', 'track_name', 'genre', 'duration_time']].iloc[top_idx].copy()
  res['score'] = [d[x] for x in top_idx]
  print("EUC", euc_scores)
  print("tfidf", tf_scores)
  print("CNT", cnt_scores)
  return res
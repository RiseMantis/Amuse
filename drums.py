from fastapi import FastAPI
from pydantic import BaseModel
from rec import getRecommendation

app = FastAPI()

class Song(BaseModel):
  song: str
  tf_weight: float = 0.25
  cnt_weight: float = 0.25
  euc_weight: float = 0.5

@app.post('/backend')
def getRecs(data: Song):
  recs = getRecommendation(data.song, tf_weight=data.tf_weight, cnt_weight=data.cnt_weight, euc_weight=data.euc_weight)
  return recs.to_dict(orient='records')
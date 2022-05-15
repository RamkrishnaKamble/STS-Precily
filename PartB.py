import uvicorn
from partA import get_similarity
from gensim.models.doc2vec import Doc2Vec
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
class Item(BaseModel):
    text1: str
    text2: str

@app.post("/test")
def get_dict(item:Item):
    test_dict = {"text1": item.text1, "text2": item.text2 }
    score =  float(get_similarity(test_dict))
    score = round(score,2)
    return {"simillarity score":score}

if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port = 8000)

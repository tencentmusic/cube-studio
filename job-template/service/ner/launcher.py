import uvicorn
from fastapi import FastAPI
import pickle
import os

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"

app = FastAPI()


def load_model(filename='./ckpts/model_BiLSTM_CRF.pkl'):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model
MODEL_PATH=os.getenv("MODEL_PATH",'/mnt/admin/model.pkl')
model = load_model(MODEL_PATH)

def serve(model, sentence):
    # sentence = '1962年1月出生，南京工学院毕业。'
    ans = model.predict_sentence(sentence)[0]
    # for i in range(len(ans[0])):
    #     print('{}:{}'.format(sentence[i], ans[0][i]))
    n = len(ans)
    pos = 0
    output = []
    tags = []
    count = 0
    while pos<n:
        count += 1
        if count >=100*n:
            return ans
        tmp = ''
        if pos < n and ans[pos][0] == 'B':
            tags.append(ans[pos][2:])
            tmp += sentence[pos]
            pos += 1
            while pos < n and ans[pos][0] != 'B' and ans[pos][0] != 'O':
                count += 1
                if count >=100*n:
                    return ans
                tmp += sentence[pos]
                pos += 1
            output.append(tmp)
        tmp = ''
        while pos < n and ans[pos][0] == 'O':
            count +=1
            if count >=100*n:
                return ans
            tmp += sentence[pos]
            pos += 1
        if tmp:
            tags.append('O')
            output.append(tmp)
    outputs = ','.join(output)
    tagsStr = ','.join(tags)
    return outputs + '\n' + tagsStr


@app.get("/")
async def serve_api(s: str):
    res = serve(model, s)
    return {"result": res}


if __name__ == "__main__":

    uvicorn.run(app=app, host='0.0.0.0', port=8123)


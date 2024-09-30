from crf_model import resp_from_model as resp_model
import json
if __name__ == "__main__":
    txt = "หมึกผัดไขา่ของโต๊ะ 4 กับ 7 เตรียมให้แล้วหรือยัง"
    result = resp_model.predict_resp(txt)
    print(result)
from crf_model import resp_from_model as resp_model
import speech_recognition as sr
import pyaudio
if __name__ == "__main__":
    
    sr.Microphone.list_microphone_names()
    mic = sr.Microphone(1)
    recog = sr.Recognizer()
    
    with mic as source:
        audio = recog.listen(source)
        
    with mic as source:
        while True:
            audio = recog.listen(source)
            try:
                print("Listen..")
                txt = recog.recognize_google(audio,language='th')
                print(txt)
                # txt = "หมึกผัดไขา่ของโต๊ะ 4 กับ 7 เตรียมให้แล้วหรือยัง"
                result = resp_model.predict_resp(txt)
                print(result)
                
            except:
                continue
    

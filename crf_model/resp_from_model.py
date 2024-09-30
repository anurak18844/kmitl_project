from sklearn.model_selection import train_test_split
import sklearn_crfsuite
from pythainlp.tokenize import word_tokenize
from pythainlp.tag import pos_tag


model = r'C:\Users\user\Desktop\MODEL_NER\crf_model\crf_model_ner'
# model = "crf_model_ner"
crf_model = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=500,
    all_possible_transitions=True,
    model_filename=model
)

def doc2features(doc, i):
    word = doc[i][0]
    postag = doc[i][1]
    features = {
        'word.word': word,
        'word.isspace':word.isspace(),
        'postag':postag,
        'word.isdigit()': word.isdigit()
    }
    if i > 0:
        prevword = doc[i-1][0]
        postag1 = doc[i-1][1]
        features['word.prevword'] = prevword
        features['word.previsspace'] = prevword.isspace()
        features['word.prepostag'] = postag1
        features['word.prevwordisdigit'] = prevword.isdigit()
    else:
        features['BOS'] = True
    if i < len(doc)-1:
        nextword = doc[i+1][0]
        postag1 = doc[i+1][1]
        features['word.nextword'] = nextword
        features['word.nextisspace'] = nextword.isspace()
        features['word.nextpostag'] = postag1
        features['word.nextwordisdigit'] = nextword.isdigit()
    else:
        features['EOS'] = True
    return features

def extract_features(doc):
    return [doc2features(doc, i) for i in range(len(doc))]

def postag(text):
    listtxt = [i for i in text.split('\n') if i!='']
    list_word = []
    for data in listtxt:
        list_word.append(data.split('\t')[0])
    list_word=pos_tag(list_word,engine="perceptron")
    text=""
    i=0
    for data in listtxt:
        text+=data.split('\t')[0]+'\t'+list_word[i][1]+'\t'+data.split('\t')[1]+'\n'
        i+=1
    return text

def get_ner(text):
    word_cut=word_tokenize(text,keep_whitespace=False)
    # print(word_cut)
    list_word=pos_tag(word_cut,engine='perceptron')
    # print(list_word)
    X_test = extract_features([(data,list_word[i][1]) for i,data in enumerate(word_cut)])
    # print(X_test)
    y_=crf_model.predict_single(X_test)
    return [(word_cut[i],list_word[i][1],data) for i,data in enumerate(y_)]

def process_data(data):
    result = {"TABLE": [], "COMMAND": "", "FOOD": [], "QUESTION": False}
    current_table = None
    current_food = []
    for word, tag, label in data:
        if label.startswith("B-TABLE"):
            current_table = int(word)
        elif label.startswith("B-FOOD"):
            current_food = [word]
        elif label.startswith("I-FOOD"):
            current_food.append(word)
        elif label.startswith("B-COMMAND_"):
            result["COMMAND"] = "COMMAND_" + label.split("_")[1]
        elif label.startswith("B-QUESTION"):
            result["QUESTION"] = True
        elif label == "O":
            if current_table is not None:
                result["TABLE"].append(current_table)
                current_table = None
            if current_food:
                result["FOOD"].append("".join(current_food))
                current_food = []
    if current_table is not None:
        result["TABLE"].append(current_table)
    if current_food:
        result["FOOD"].append("".join(current_food))
    return result

def predict_resp(txt):
    p_data = get_ner(txt)
    # print(p_data)
    return process_data(p_data)



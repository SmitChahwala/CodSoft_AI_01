# CHATBOT WITH RULE-BASED RESPONSES
import nltk
import random
import string
import warnings
warnings.filterwarnings("ignore")

f = open('E:\Internship\CodSoft\Task1\chatbot_text.txt', 'r', errors='ignore')
raw = f.read()
raw = raw.lower()

sent_tokens = nltk.sent_tokenize(raw)    # converts to list of sentences
word_tokens = nltk.word_tokenize(raw)    # converts to list of words
sentToken = sent_tokens[:4]
wordToken = word_tokens[:4]
lammer = nltk.stem.WordNetLemmatizer()   # preprocessing

def LemTokens(tokens):
    return [lammer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

#Greetings
greeting_inputs = ("hello", "hi", "greetings", "sup", "what's up", "hey")
greeting_responses = ["hi", "hey", "nods", "hi there", "hello", "I'm glad! you are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in greeting_inputs:
            return random.choice(greeting_responses)
#Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def response(user_response):
    chatbot_response = ' '
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals =  cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()    # convert multiD to 1D array
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf == 0):
        chatbot_response = chatbot_response + "I'm sorry! I don't understand you"
        return chatbot_response
    else:
        chatbot_response = chatbot_response + sent_tokens[idx]
        return chatbot_response
    
if __name__ == '__main__':
    flag = True
    print("Hello, there my name is Spyder. I'll answer your queries. If you want to exit, type Bye!")
    while(flag==True):
        user_response = input()
        user_response = user_response.lower()
        if(user_response!='bye'):
            if(user_response == 'thanks' or user_response == 'thank you'):
                flag = False
                print("Spyder: You're welcome!")
            else:
                if(greeting(user_response)!= None):
                    print("Spyder:" +greeting(user_response))
                else:
                    print("Spyder:", end='')
                    print(response(user_response))
                    sent_tokens.remove(user_response)
        else:
            print("Spyder: Bye! Have a great time!")
                    
                    
from warnings import filterwarnings
import drops
import join
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df = pd.read_csv("KOZMOS/amazon.csv")
df.head()
df.info()
#########################################################3
#Görev-1:METİN ÖNİŞLEME
#############################################################

#########################################################
#Normalizing Case Folding---- yazıları büyütme küçültme....
###########################################################
df['Review'] = df['Review'].str.lower()


################################################
#Punctuations------Noktalama işaretlerini silme
##################################################
df['Review'] = df['Review'].str.replace(r'[^\w\s]', '', regex=True)


#######################################
#Numbers---------sayılardan kurtulma
#######################################
df['Review'] = df['Review'].str.replace(r'\d', '', regex=True)

#######################################
#StopWords------bağlaçları vs(anlamsız kelimeleri) silme
###########################################

#nltk.download('stopwords')
sw = stopwords.words('english')
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

#######################################
#RareWords---nadiren geçen kelimeleri siler--- 1 ve daha az geçen kelimeleri sildik
#####################################
temp_df = pd.Series(' '.join(df['Review']).split()).value_counts()[-1000:]
drops = temp_df[temp_df <= 1000]
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

########################################
#Tokenization---Topluca kelime kelime ayırma
#######################################
#df["reviewText"].apply(lambda x: TextBlob(x).words).head()

########################################
#Lemmazation---aynı  kelimelerin köklerini birleştirme
#######################################
df['Review'] = df['Review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

##########################################
#Görev2-METİN GÖRSELLEŞTİRME
#####################################

########################################
#Text Visuallization---kelimelerin frekanslarını çıkartmak(kelimelerle görsel oluşturmak için yaptık)
#######################################
tf = df['Review'].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

tf.columns = ["words", "tf"]
tf.sort_values("tf", ascending=False)
##########################################
#BarPlot(Sütun Grafiği)
#######################################

plt.figure(figsize=(12, 8))
tf[tf["tf"] > 500].plot.bar(x="words", y="tf", legend=False)
plt.xlabel('Words')
plt.ylabel('tf')
plt.title('Word Frequency Bar Plot')
plt.show()


##########################################
#KelimeBulutu(WordCloud)
############################################

text = " ".join(i for i in df.Review)

wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)

plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

##########################################
#Görev3-Duygu Analizi
#########################################
##################################
#Sentiment Analysis
#################################
df["Review"].head()
import nltk
#nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
df["Review"][0:10].apply(lambda x: sia.polarity_scores(x))
df["Review"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

######################################################
#Feature Engineering
######################################################

df["Review"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["sentiment_label"] = df["Review"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["sentiment_label"].value_counts()

df["sentiment_label"]= LabelEncoder().fit_transform(df["sentiment_label"])
y=df["sentiment_label"]
x= df["Review"]

######################################################
#Görev 4: Makine Öğrenmesine Hazırlık
######################################################

########################################
#Count Vectors----Frekans temsiller(kelimelerin frekanslarını/kaçar tane geçtiğini bulmak)
########################################
#words----kelimelerin numerik temsilleri
#characters---karakterlerin numerik tems.
#ngram(örneği aşağıda)----kelime öbeklenmesi ile feature oluşturmak(kelime kombinasyonları oluşturur)
#a = """Bu örneğin anlaşılabilmesi içişn daha uzun bir metin üzerinden göstereceğim.
#N-graml'lar birlikte kullanılan kelimelerin kombinasyonlarını gösterir ve feature üretmek için kullanılır"""
#   TextBlob(a).ngrams(3)

from sklearn.feature_extraction.text import CountVectorizer
#word frekans
vectorizer = CountVectorizer()
X_c = vectorizer.fit_transform(x)
vectorizer.get_feature_names_out()
X_c.toarray()

#n-gram frekans
vectorizer2= CountVectorizer(analyzer='word', ngram_range=(2, 2))
X_n = vectorizer2.fit_transform(x)
vectorizer2.get_feature_names_out()
X_n.toarray()

########################################
#TF-IDF Vectors----Normalize edilmiş frekans temsiller---kelimelerin geçme frekansları
##############################
#TF-Term Frequency---t teriminin ilgili dokümandaki frekansı/ dökümandaki toplam terim sayısı)
#IDF-Inverse Document Frequency'i Hesapla----- 1+loge((toplam doküman sayısı+1)/(içinde t terimi olan doküman sayısı+1))
#TF*IDF 'i hesapla
#L2 Normalizasyonu yap----Satırların kareleri toplamının karekökünü bul, ilgili satıdaki tüm hücreleri bulduğun değerlere böl
########################################
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf_word_vectorizer=TfidfVectorizer()

x_tf_idf_word = tf_idf_word_vectorizer.fit_transform(x)

tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range=(2, 3))
x_tf_idf_ngram = tf_idf_word_vectorizer.fit_transform(x)
########################################
# Word Embeddings(Word2Vec, GloVe, BERT vs)
########################################



#####################################
#Sentiment Modeling
#####################################
#1.Text Processing
#2. Text Visualization
#3.Sentiment Analysis
#4.Feature Engineering
#5.Sentiment Modeling

#####################################
#Logistic Regression
#####################################

log_model = LogisticRegression().fit(x_tf_idf_word, y)
cross_val_score(log_model,
                x_tf_idf_word,
                y,
                scoring="accuracy",
                cv=5).mean()

new_review = pd.Series("this product is great")
new_review = pd.Series("look at that shit very bad")
new_review = TfidfVectorizer().fit(x).transform(new_review)

log_model.predict(new_review)

##-----Aldığımız veri setinden sor

randon_review = pd.Series(df["Review"].sample(1).values)
new_review = TfidfVectorizer().fit(x).transform(randon_review)
log_model.predict(new_review)
########################################
#RandomForest
########################################
#Count Vectors
rf_model = RandomForestClassifier().fit(X_c, y)
cross_val_score(rf_model, X_c, y, cv=5, n_jobs=-1).mean()

#TF-IDF Word-Level
rf_model = RandomForestClassifier().fit(x_tf_idf_word, y)
cross_val_score(rf_model, x_tf_idf_word, y, cv=5, n_jobs=-1).mean()

#TF-IDF N-GRAM
rf_model = RandomForestClassifier().fit(x_tf_idf_ngram, y)
cross_val_score(rf_model, x_tf_idf_ngram, y, cv=5, n_jobs=-1).mean()


####################################################
#HİPERMARAMETRE OPTİMİZASYONU
####################################################
rf_model= RandomForestClassifier(random_state=17)

rf_params = {"max_depth": [8, None],
             "max_features": [7, "auto"],
             "min_samples_split":[2,5,8],
             "n_estimators": [100,200]}

rf_best_grid = GridSearchCV(rf_model,
                            rf_params,
                            cv=5,
                            n_jobs=-1,
                            verbose=True).fit(X_c, y)

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X_c, y)

cross_val_score(rf_final,X_c, y, cv=5, n_jobs=-1).mean()


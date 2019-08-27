#!/usr/bin/env python
# coding: utf-8

# In[44]:


#Importação de ferramentas básicas para análise de dados com Python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import statsmodels.api as sm
from sklearn import svm


# In[4]:


#Lê o arquivo
vinhos_df = pd.read_csv("winequality.csv", sep = ";")


# In[5]:


#informações básicas sobre arquivo
print("Base de dados: ", vinhos_df.shape[0], "numero de elementos", 
      vinhos_df.shape[1], "variaveis")
print vinhos_df.info()


# In[15]:


#para facilitar análises futuras transforma as variaveis não numericas em numericas e força o qualidade ser numerica
vinhos_df['type'] = vinhos_df.type.replace('White', 1.0)
vinhos_df['type'] = vinhos_df.type.replace('Red', 2.0)
vinhos_df['quality'] = pd.to_numeric(vinhos_df['quality'], errors='coerce')
vinhos_df['quality'] = map(float, vinhos_df['quality'])
vinhos_df['alcohol'] = pd.to_numeric(vinhos_df['alcohol'], errors='coerce')


# In[32]:


#Exclui os dados que não possuem a caracteristica alcool
print vinhos_df.shape
vinhos_df = vinhos_df[vinhos_df.alcohol.notnull()]
vinhos_df = vinhos_df[vinhos_df.alcohol.notna()]
print vinhos_df.shape
print consistenci_dados_tabela(vinhos_df)
print consistenci_dados_tabela2(vinhos_df)


# In[33]:


vinhos_df.describe()


# In[ ]:


#histograma da qualidade do vinho - só ilustrativo da distribuição da variavel
def custom_style(title, xlab, ylab, width = 800, height = 600):
    p = figure(plot_width = width, plot_height = height, title = title, x_axis_label = xlab, y_axis_label = ylab)
    p.title.text_font_size = "20pt"
    p.title.align = "center"
    p.xaxis.axis_label_text_font_size = "20pt"
    p.yaxis.axis_label_text_font_size = "20pt"
    return p

def custom_barplot(data, width = 800, height = 600, color = "#3687cc", title = "Barplot", xlab = "Category",ylab = "Quantidade"):
    counts = data.value_counts()
    p = custom_style(width = width, height = height, title = title, xlab = xlab, ylab = ylab)
    p.vbar(x = counts.index, top = counts.values, width = 0.5, bottom = 0,  color = color)
    return show(p)

custom_barplot(vinhos_df.quality, title = "Distriubição da qualidade do vinho", xlab = "Qualidade classificada de 0 a 10", 
               ylab = "Quantidade")


# In[ ]:


#verifica a simetria de todas as variaveis
def subplot_hist(data, row = 4, column = 3, title = "Subplots", height = 20, width = 19):
    fig = plt.figure(figsize = (width, height))
    fig.suptitle(title, fontsize=25, y = 0.93)
    for i in range(data.shape[1]):
        ax = fig.add_subplot(row, column, i + 1)
        fig.subplots_adjust(hspace = .5)
        sns.distplot(vinhos_df.iloc[:, i], ax=ax)
        ax.xaxis.get_label()
    plt.show()
subplot_hist(vinhos_df.iloc[:, :-1], row = 4, column = 3, title = "Histogramas das variaveis independentes")


# In[ ]:


#Verifica as variaves viesadas ou normalmente distribuidas, quanto mais proximo de zero mais menos viesadas (distribuiçao normal)
def skewness_check(data):
    skew_value = list(st.skew(data))
    skew_string = []
    for skew in skew_value:
        if skew >= -.5 and skew <= .5:
            skew_string.append("Fracamente viesada")
        elif skew <= -.5 and skew >= -1 and skew <= .5 and skew >= 1:
            skew_string.append("Moderadamente viesada")
        else:
            skew_string.append("Muito viesada")
    skew_df = pd.DataFrame({'Variavel': data.columns, 'Viesada': skew_value, 'Valor': skew_string})
    return skew_df
skewness_check(vinhos_df.iloc[:,])


# In[34]:


X = vinhos_df.iloc[:, 2:32].values
y = vinhos_df['quality']


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state=0)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[36]:


df = pd.DataFrame(np.random.rand(4,5),
                  columns=["Nome", "Acurácia", "Recall",
                           "Especificidade", "Precisão"])

tuned_parameters = [
    [{'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': ['auto', 'scale']},
     {'kernel': ['poly'], 'degree': [1,2,3,4,5,6], 'gamma': ['auto', 'scale']}],
    [{'C': [0.1, 1, 10, 100], 'solver': ['lbfgs', 'newton-cg'], 'max_iter': [1000]}],
    [{'max_depth': [2,3,4,5,6,7,8], 'splitter': ['best']}],
    [{'max_depth': [2,3,4,5,6,7,8], 'n_estimators': [10,100,200]}]
]

c = 0
for cl in [SVC, LogisticRegression,
           DecisionTreeClassifier, RandomForestClassifier]:
    classifier = GridSearchCV(cl(), tuned_parameters[c], cv=5,
                              scoring='accuracy')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    print("Melhores parâmetros:\n")
    print(classifier.best_params_)

    cm = confusion_matrix(y_test, y_pred)
    
    VN, FP, FN, VP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    total = VN + FP + FN + VP
    df.iloc[c] = [cl.__name__,
                  (VP + VN) / total,
                  VP / (FN + VP),
                  VN / (VN + FP),
                  VP / (FP + VP)]
    c = c + 1
    
df


# In[45]:


def svm_accuracy_cv(X_train, X_test, y_train, y_test, n_fold = 10):
    best_score = 0
    best_C = 10000
    C_list = [1, 10, 100, 1000]
    for C in C_list:
        svc = svm.SVC(C = C, kernel = 'rbf')
        scores = cross_val_score(svc, X_train, y_train, cv = n_fold)
        score = scores.mean()
        if score > best_score:
            best_C = C
            best_score = score
    svc = svm.SVC(C = best_C, kernel = "rbf")
    svc.fit(X_train, y_train)
    accuracy = svc.score(X_test, y_test)
    return accuracy, best_C
acuracia, C_modelo = svm_accuracy_cv(X_train, X_test, y_train, y_test, n_fold = 10)
print("A acuracia do modelo e", 
      round(acuracia * 100, 2), "com o melhor custo", C_modelo)


# In[39]:


X = vinhos_df.iloc[:, -1]
y = vinhos_df.iloc[:, :-1]
#minimo quadrados ordinarios - linar
mod = sm.OLS(X, y) 
res = mod.fit()    
print(res.summary())


# In[ ]:





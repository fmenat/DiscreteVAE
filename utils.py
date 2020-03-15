from keras.layers import *
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gc, sys,os

def compare_hist_train(hist1,hist2, dataset_name="", global_L = True):
    ### binary vs traditional
    plt.figure(figsize=(15,6))
    if global_L:
        history_dict1 = hist1.history
        history_dict2 = hist2.history
        loss_values1 = history_dict1['loss']
        val_loss_values1 = history_dict1['val_loss']
        loss_values2 = history_dict2['loss']
        val_loss_values2 = history_dict2['val_loss']
        epochs_l = range(1, len(loss_values1) + 1)

        plt.figure(figsize=(15,6))
        plt.plot(epochs_l, loss_values1, 'bo-', label = "Train set traditional")
        plt.plot(epochs_l, val_loss_values1, 'bv-', label = "Val set traditional")
        plt.plot(epochs_l, loss_values2, 'go-', label = "Train set binary")
        plt.plot(epochs_l, val_loss_values2, 'gv-', label = "Val set binary")
    else:
        add_hist_plot(hist1, c='b', model_n = "VAE")
        add_hist_plot(hist2, c='g', model_n = "B-VAE")
  
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc="upper right", fancybox= True)
    plt.title("VAE loss "+dataset_name)
    plt.show()
    
def add_hist_plot(hist, c='b', model_n = ""):
    history_dict = hist.history
    rec_loss_values = history_dict['REC_loss']
    kl_loss_values = history_dict['KL']
    rec_val_loss_values = history_dict['val_REC_loss']
    kl_val_loss_values = history_dict['val_KL']
    epochs_l = range(1, len(rec_loss_values) + 1)

    plt.plot(epochs_l, rec_loss_values, c+'o-', label = "Train REC loss (%s)"%model_n)
    plt.plot(epochs_l, kl_loss_values, c+'o-.', label = "Train KL loss (%s)"%model_n)

    plt.plot(epochs_l, rec_val_loss_values, c+'v-', label = "Val REC loss (%s)"%model_n)
    plt.plot(epochs_l, kl_val_loss_values, c+'v-.', label = "Val KL loss (%s)"%model_n)
    
    
def visualize_probas(logits, probas):
    sns.distplot(probas.flatten())
    plt.title("Bits probability distribution p(b|x)")
    plt.show()
    
    from base_networks import samp_gumb
    samp_probas = samp_gumb(logits)
    
    plt.hist(samp_probas.flatten())
    plt.title("Gumbel-Softmax sample \hat{b}")
    plt.show()
    
def visualize_mean(data):
    sns.distplot(data)
    plt.title("Continous Bits distribution (standar VAE)")
    plt.show()
    

def define_fit(multi_label,X,Y, epochs=20):
    #function to define and train model

    #define model
    model_FF = Sequential()
    model_FF.add(Dense(256, input_dim=X.shape[1], activation="relu"))
    #model_FF.add(Dense(128, activation="relu"))
    if multi_label:
        model_FF.add(Dense(Y.shape[1], activation="sigmoid"))
        model_FF.compile(optimizer='adam', loss="binary_crossentropy")
    else:
        model_FF.add(Dense(Y.shape[1], activation="softmax"))
        model_FF.compile(optimizer='adam', loss="categorical_crossentropy",metrics=["accuracy"])
    model_FF.fit(X, Y, epochs=epochs, batch_size=128, verbose=0)
    return model_FF


class MedianHashing(object):
    def __init__(self):
        self.threshold = None
        self.latent_dim = None
    def fit(self, X):
        self.threshold = np.median(X, axis=0)
        self.latent_dim = X.shape[1]
    def transform(self, X):
        assert(X.shape[1] == self.latent_dim)
        binary_code = np.zeros(X.shape, dtype='int32')
        for i in range(self.latent_dim):
            binary_code[np.nonzero(X[:,i] < self.threshold[i]),i] = 0
            binary_code[np.nonzero(X[:,i] >= self.threshold[i]),i] = 1
        return binary_code
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
#if median is used, my binary codes should use it as well.. a probability 0.6 does not mean that
# the bit is always on..
#median= MedianHashing()
#median.fit(encode_train)
#val_train = median.transform(encode_train)
#val_hash = median.transform(encode_val)
def calculate_hash(data, from_probas=True, from_logits=True):    
    if from_probas: #from probas
        if from_logits:
            from scipy.special import expit
            data = expit(data)
        data_hash = (data > 0.5)*1
    else: #continuos
        data_hash = (np.sign(data) + 1)/2
    return data_hash.astype('int32')

def get_hammD(query, corpus):
    """
        Retrieve similar documents to the query document inside the corpus (source)
    """
    #codify binary codes to fastest data type
    query = query.astype('int8') #no voy a ocupar mas de 127 bits
    corpus = corpus.astype('int8')
    
    query_hammD = np.zeros((len(query),len(corpus)),dtype='int16') #distancia no sera mayor a 2^16
    for i,dato_hash in enumerate(query):
        query_hammD[i] = calculate_hamming_D(dato_hash, corpus) # # bits distintos)
    return query_hammD

def get_similar_hammD_based(query_hammD,tipo="topK", K=100, ball=0):
    """
        Retrieve similar documents to the query document inside the corpus (source)
    """
    query_similares = [] #indices
    for i in range(len(query_hammD)):        
        if tipo=="ball" or tipo=="EM":
            K = np.sum(query_hammD[i] <= ball) #find K over ball radius
            
        #get topK
        ordenados = np.argsort(query_hammD[i]) #indices
        query_similares.append(ordenados[:K].tolist()) #get top-K
    return query_similares


def xor(a,b):
    return (a|b) & ~(a&b)
def calculate_hamming_D(a,B):
    #return np.sum(a.astype('bool')^ B.astype('bool') ,axis=1) #distancia de hamming (# bits distintos)
    #return np.sum(np.logical_xor(a,B) ,axis=1) #distancia de hamming (# bits distintos)
    v = np.sum(a != B,axis=1) #distancia de hamming (# bits distintos) -- fastest
    #return np.sum(xor(a,B) ,axis=1) #distancia de hamming (# bits distintos)
    return v.astype(a.dtype)

def get_similar(query, corpus,tipo="topK", K=100, ball=2):
    """
        Retrieve similar documents to the query document inside the corpus (source)
    """
    #codify binary codes to fastest data type
    query = query.astype('int8') #no voy a ocupar mas de 127 bits
    corpus = corpus.astype('int8')
    
    query_similares = [] #indices
    for dato_hash in query:
        hamming_distance = calculate_hamming_D(dato_hash, corpus) # # bits distintos)
        if tipo=="EM": #match exacto
            ball= 0
        
        if tipo=="ball" or tipo=="EM":
            K = np.sum(hamming_distance<=ball) #find K over ball radius
            
        #get topK
        ordenados = np.argsort(hamming_distance) #indices
        query_similares.append(ordenados[:K].tolist()) #get top-K
    return query_similares

def measure_metrics(labels_name, data_retrieved_query, labels_query, labels_source):
    """
        Measure precision at K and recall at K, where K is the len of the retrieval documents
    """
    if type(labels_source) == list:
        labels_source = np.asarray(labels_source)
    
    #relevant document for query data
    count_labels = {label:np.sum([label in aux for aux in labels_source]) for label in labels_name} 
    
    precision = 0.
    recall =0.
    for similars, label in zip(data_retrieved_query, labels_query): #source de donde se extrajo info
        if len(similars) == 0: #no encontro similares:
            continue
        labels_retrieve = labels_source[similars] #labels of retrieved data
        
        if type(labels_retrieve[0]) == list or type(labels_retrieve[0]) == np.ndarray: #multiple classes
            tp = np.sum([len(set(label)& set(aux))>=1 for aux in labels_retrieve]) #al menos 1 clase en comun --quizas variar
            recall += tp/np.sum([count_labels[aux] for aux in label ]) #cuenta todos los label del dato
        else: #only one class
            tp = np.sum(labels_retrieve == label) #true positive
            recall += tp/count_labels[label]
        precision += tp/len(similars)
    
    return precision/len(labels_query), recall/len(labels_query)

def P_atk(labels_retrieved, label_query, K=1):
    """
        Measure precision at K
    """
    if len(labels_retrieved)>K:
        labels_retrieved = labels_retrieved[:K]

        
    if type(labels_retrieved[0]) == list or type(labels_retrieved[0]) == np.ndarray: #multiple classes
        tp = np.sum([len(set(label_query)& set(aux))>=1 for aux in labels_retrieved]) #al menos 1 clase en comun --quizas variar
    else: #only one class
        tp = np.sum(labels_retrieved == label_query) #true positive
    
    return tp/len(labels_retrieved) #or K

def M_P_atk(datas_similars, labels_query, labels_source, K=1):
    """
        Mean (overall the queries) precision at K
    """
    if type(labels_source) == list:
        labels_source = np.asarray(labels_source)
    return np.mean([P_atk(labels_source[datas_similars[i]],labels_query[i],K=K) if len(datas_similars[i]) != 0 else 0.
                    for i,_ in enumerate(datas_similars)])


def AP_atk(data_retrieved_query, label_query, labels_source, K=0):
    """
        Average precision at K, average all the list precision until K.
    """
    multi_label=False
    if type(label_query) == list or type(label_query) == np.ndarray: #multiple classes
        multi_label=True
        
    if type(labels_source) == list:
        labels_source = np.asarray(labels_source)
        
    if K == 0:
        K = len(data_retrieved_query)
    
    if len(data_retrieved_query)>K:
        data_retrieved_query = data_retrieved_query[:K]
    
    labels_retrieve = labels_source[data_retrieved_query] 
    
    score = []
    num_hits = 0
    p = 0 
    for i in range(K):
        relevant=False
        
        if multi_label:
            if len( set(label_query)& set(labels_retrieve[i]) )>=1: #at least one label in comoon at k
                relevant=True
        else:
            if label_query == labels_retrieve[i]: #only if "i"-element is relevant 
                relevant=True
        
        if relevant:
            num_hits +=1 
            score.append(num_hits/(i+1)) #precition at k 
            
    if len(score) ==0:
        return 0
    else:
        return np.mean(score) #average all the precisionts until K

def MAP_atk(datas_similars, labels_query, labels_source, K=0):
    """
        Mean (overall the queries) average precision at K
    """
    return np.mean([AP_atk(datas_similars[i], labels_query[i], labels_source, K=K) if len(datas_similars[i]) != 0 else 0.
                    for i,_ in enumerate(datas_similars)]) 


##valores unicos de hash? distribucion de casillas
def hash_analysis(hash_data):
    hash_string = []
    for valor in hash_data:
        hash_string.append(str(valor)[1:-1].replace(' ',''))
    valores_unicos = set(hash_string)
    count_hash = {valor: hash_string.count(valor) for valor in valores_unicos}
    return valores_unicos, count_hash

def compare_cells_plot(nb,train_hash1,train_hash2,test_hash1=[],test_hash2=[]):
    print("Entrenamiento----")
    print("Cantidad de datos a llenar la tabla hash: ",train_hash1.shape[0])

    valores_unicos, count_hash =  hash_analysis(train_hash1)
    print("Cantidad de memorias ocupadas hash1: ",len(valores_unicos))
    plt.figure(figsize=(14,4))
    plt.plot(sorted(list(count_hash.values()))[::-1],'bo-',label="Binary")
    
    valores_unicos, count_hash =  hash_analysis(train_hash2)
    print("Cantidad de memorias ocupadas hash2: ",len(valores_unicos))
    plt.plot(sorted(list(count_hash.values()))[::-1],'go-',label="Traditional")
    plt.legend()
    plt.show()
    
    if len(test_hash1) != 0:
        print("Pruebas-----")
        print("Cantidad de datos a llenar la tabla hash: ",test_hash1.shape[0])
        
        valores_unicos, count_hash =  hash_analysis(test_hash1)
        print("Cantidad de memorias ocupadas hash1: ",len(valores_unicos))
        plt.figure(figsize=(15,4))
        plt.plot(sorted(list(count_hash.values()))[::-1],'bo-',label="Binary")
        
        valores_unicos, count_hash =  hash_analysis(test_hash2)
        print("Cantidad de memorias ocupadas hash2: ",len(valores_unicos))
        plt.plot(sorted(list(count_hash.values()))[::-1],'go-',label="Traditional")
        plt.legend()
        plt.show()
        

from PIL import Image
def check_availability(folder_imgs, imgs_files, labels_aux):
    imgs_folder = os.listdir(folder_imgs)

    mask_ = np.zeros((len(imgs_files)), dtype=bool) 
    for contador, (img_n, la) in enumerate(zip(imgs_files, labels_aux)):
        if contador%10000==0:
            gc.collect()
        
        if img_n in imgs_folder and len(la)!=0: #si imagen fue descargada y tiene labels.
            imagen = Image.open(folder_imgs+img_n)
            aux = np.asarray(imagen)
            if len(aux.shape) == 3 and aux.shape[2] == 3:#si tiene 3 canals
                mask_[contador] = True
            
            imagen.close()
    return mask_

def load_imgs_mask(imgs_files, mask_used, size, dtype = 'uint8'):
    N_used = np.sum(mask_used)
    X_t = np.zeros((N_used, size,size,3), dtype=dtype)
    real_i = 0
    for contador, foto_path in enumerate(imgs_files):
        if contador%10000==0:
            print("El contador de lectura va en: ",contador)
            gc.collect()

        if mask_used[contador]:
            #abrir imagen
            imagen = Image.open(foto_path)
            aux = imagen.resize((size,size),Image.ANTIALIAS)
            X_t[real_i] = np.asarray(aux, dtype=dtype)

            imagen.close()
            aux.close()
            del aux, imagen
            real_i +=1
    return X_t

def get_topK_labels(labels_set, labels, K=1):
    count_labels = {label:np.sum([label in aux for aux in labels_set]) for label in labels} 
    sorted_x = sorted(count_labels.items(), key=lambda kv: kv[1], reverse=True)
    print("category with most data (%s) has = %d, the top-K category (%s) has = %d"%(sorted_x[0][0],sorted_x[0][1],sorted_x[K-1][0], sorted_x[K-1][1]))
    return [value[0] for value in sorted_x[:K]]

def set_newlabel_list(new_labels, labels_set):
    return [[topic for topic in labels_list if topic in new_labels] for labels_list in labels_set]

def enmask_data(data, mask):
    if type(data) == list:
        return np.asarray(data)[mask].tolist()
    elif type(data) == np.ndarray:
        return data[mask]
    
def sample_test_mask(labels_list, N=100, multi_label=True):
    idx_class = {}
    for value in np.arange(len(labels_list)):
        if multi_label:
            for tag in labels_list[value]:
                if tag in idx_class:
                    idx_class[tag].append(value)
                else:
                    idx_class[tag] = [value]
        else:
            tag = labels_list[value]
            if tag in idx_class:
                idx_class[tag].append(value)
            else:
                idx_class[tag] = [value]

    mask_train = np.ones(len(labels_list), dtype='bool')
    selected = []
    for clase in idx_class.keys():
        selected_clase = []
        for dato in idx_class[clase]:
            if dato not in selected:
                selected_clase.append(dato) # si dato no ha sido seleccionado como rep de otra clase se guarda

        v = np.random.choice(selected_clase, size=N, replace=False)
        selected += list(v)
        mask_train[v] = False #test set
    return mask_train
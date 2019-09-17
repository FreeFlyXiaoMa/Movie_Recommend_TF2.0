# -*- coding: utf-8 -*-
#@Time    :2019/9/17 10:03
#@Author  :XiaoMa
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import os
import pickle
import re
import shutil

from urllib.request import urlretrieve
from tqdm import tqdm
import zipfile
import hashlib
import time

def _unzip(save_path,_,database_name,data_path):
    """Unzip wrapper with the same interface as _ungzip"""
    print('Extracting {}...'.format(database_name))
    with zipfile.ZipFile(save_path) as zf:
        zf.extractall(data_path)

def download_extract(database_name,data_path):
    """
    Download and extract database
    :param database_name:
    :param data_path:
    :return:
    """
    DATABASE_ML1M='ml-1m'
    if database_name==DATABASE_ML1M:
        url='http://files.grouplens.org/datasets/movielens/ml-1m.zip'
        hash_code = 'c4d9eecfca2ab87c1945afe126590906'
        extract_path=os.path.join(data_path,'ml-1m')
        save_path=os.path.join(data_path,'ml-1m.zip')
        extract_fn=_unzip

    if os.path.exists(extract_path):
        print('Found {} Data'.format(database_name))
        return

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if not os.path.exists(save_path):
        with  DLProgress(unit='B',unit_scale=True,miniters=1,desc='Downloading {}'.format(database_name)) as pbar:
            urlretrieve(url,save_path,pbar.hook())
    assert hashlib.md5(open(save_path,'rb').read()).hexdigest() == hash_code,'{} file is corrupted. Remove the file and try again'.format(save_path)
    os.makedirs(extract_path)
    try:
        extract_fn(save_path,extract_path,database_name,data_path)
    except Exception as err:
        shutil.rmtree(extract_path) #Remove extraction folder if there is an error
        raise err
    print('Done')

class DLProgress(tqdm):
    """
    Handle Progress Bar while Downloading
    """
    last_block=0
    def hook(self,block_num=1,block_size=1,total_size=None):
        """
        A hook function that will be called once on establishment of the network connection and
        once after each block read thereafter
        :param block_num:
        :param block_size:
        :param total_size:
        :return:
        """
        self.total=total_size
        self.update((block_num-self.last_block)*block_num)
        self.last_block=block_num

data_dir='./'
# download_extract('ml-1m',data_dir)

def load_data():
    """
    Load DataSet from file
    :return:
    """
    users_title = ['UserID', 'Gender', 'Age', 'JobID', 'Zip-code']
    users = pd.read_csv('./ml-1m/users.dat', sep='::', header=None, names=users_title, engine='python')
    users=users.filter(regex='UserID|Gender|Age|JobID')
    users_org=users.values

    gender_map={'F':0,'M':1}
    users['Gender']=users['Gender'].map(gender_map)

    age_map={val:ii for ii,val in enumerate(set(users['Age']))}
    users['Age']=users['Age'].map(age_map)

    #Movie
    movies_title=['MovieID','Title','Genres']
    movies=pd.read_csv('./ml-1m/movies.dat',sep='::',header=None,names=movies_title,engine='python')
    movies_orig=movies.values
    #去掉Title中的年份
    pattern=re.compile(r'^(.*)\((\d+)\)$')
    title_map={val:re.match(pattern,val).group(1) for ii,val in enumerate(set(movies['Title']))}
    movies['Title']=movies['Title'].map(title_map)

    #电影类型转数字字典
    genres_set=set()
    for val in movies['Genres'].str.split('|'):
        genres_set.update(val)
    genres_set.add('<PAD>')
    genres2int={val:ii for ii,val in enumerate(genres_set)}

    #将电影类型转成等长数字列表，长度是18
    genres_map={val:[genres2int[row] for row in val.split('|')] for ii,val in enumerate(set(movies['Genres']))}

    for key in genres_map:
        for cnt in range(max(genres2int.values())-len(genres_map[key])):
            genres_map[key].insert(len(genres_map[key])+cnt,genres2int['<PAD>'])

    movies['Genres']=movies['Genres'].map(genres_map)

    #电影Title转成数字字典
    title_set=set()
    for val in movies['Title'].str.split():
        title_set.update(val)

    title_set.add('<PAD>')
    title2int={val:ii for ii,val in enumerate(title_set)}

    #将电影Title转成登场数字列表，长度是15
    title_count=15
    title_map={val:[title2int[row] for row in val.split()] for ii,val in enumerate(set(movies['Title']))}

    for key in title_map:
        for cnt in range(title_count - len(title_map[key])):
            title_map[key].insert(len(title_map[key])+cnt,title2int['<PAD>'])
    movies['Title']=movies['Title'].map(title_map)

    #读取评分数据集
    ratings_title = ['UserID', 'MovieID', 'ratings', 'timestamps']
    ratings = pd.read_csv('./ml-1m/ratings.dat', sep='::', names=ratings_title, engine='python')
    ratings=ratings.filter(regex='UserID|MovieID|ratings')

    #合并三个表
    data=pd.merge(pd.merge(ratings,users),movies)

    #将数据分成X和y两张表
    target_fields=['ratings']
    features_pd,targets_pd=data.drop(target_fields,axis=1),data[target_fields]

    features=features_pd.values
    targets_values=targets_pd.values

    return title_count,title_set,genres2int,features,targets_values,ratings,users,movies,data,movies_orig,users_org

title_count,title_set,genres2int,features,targets_values,ratings,users,movies,data,movies_orig,users_org=load_data()
pickle.dump((title_count,title_set,genres2int,features,targets_values,ratings,users,movies,data,movies_orig,users_org),open('process.p','wb'))

def save_params(params):
    """
    Save parameters to file
    :param params:
    :return:
    """
    pickle.dump(params,open('params.p','wb'))

def load_params(params):
    return pickle.load(open('params.p','rb'))

#嵌入矩阵的维度
embed_dim=32
#用户ID个数
uid_max=max(features.take(0,1))+1
#性别
gender_max=max(features.take(2,1))+1
#年龄类别个数
age_max=max(features.take(3,1))+1
#职业个数
job_max=max(features.take(4,1))+1

#电影ID个数
movie_id_max=max(features.take(1,1))+1
#电影类型个数
movie_categories_max=max(genres2int.values())+1
#电影名单词个数
movie_title_max=len(title_set)

#对电影类型嵌入向量做加和操作的标志
combiner='sum'

#电影名长度
sentences_size=title_count
#文本卷积滑动窗口，分别滑动2，3，4，5个单词
window_sizes={2,3,4,5}
#卷积核数量
filter_num=8

#电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第5行的数据电影ID不一定是5
movieid2idx={val[0]:i for i,val in enumerate(movies.values)}

num_epochs=5
batch_size=256
dropout_keep=0.5
learning_rate=0.0001
show_every_n_batches=20

save_dir='./save'

def get_inputs():
    uid=tf.keras.layers.Input(shape=(1,),dtype='int32',name='uid')
    user_gender=tf.keras.layers.Input(shape=(1,),dtype='int32',name='user_gender')
    user_age=tf.keras.layers.Input(shape=(1,),dtype='int32',name='user_age')
    user_job=tf.keras.layers.Input(shape=(1,),dtype='int32',name='user_job')

    movie_id=tf.keras.layers.Input(shape=(1,),dtype='int32',name='movie_id')
    movie_categories=tf.keras.layers.Input(shape=(1,),dtype='int32',name='movie_categories')
    movie_titles=tf.keras.layers.Input(shape=(1,),dtype='int32',name='movie_titles')

    return uid,user_gender,user_age,user_job,movie_id,movie_categories,movie_titles

def get_user_embedding(uid,user_gender,user_age,user_job):
    """
    User Embedding
    :param uid:
    :param user_gender:
    :param user_age:
    :param user_job:
    :return:
    """
    uid_embed_layer=tf.keras.layers.Embedding(uid_max,embed_dim,input_length=1,name='uid_embed_layer')(uid)
    gender_embed_layer=tf.keras.layers.Embedding(gender_max,embed_dim//2,input_length=1,name='gender_embed_layer')(user_gender)
    age_embed_layer=tf.keras.layers.Embedding(age_max,embed_dim//2,input_length=1,name='age_embed_layer')(user_age)
    job_embed_layer=tf.keras.layers.Embedding(job_max,embed_dim//2,input_length=1,name='job_embed_layer')(user_job)
    return uid_embed_layer,gender_embed_layer,age_embed_layer,job_embed_layer

def get_user_feature_layer(uid_embed_layer,gender_embed_layer,age_embed_layer,job_embed_layer):
    """
    User Dense Layer
    :param uid_embed_layer:
    :param gender_embed_layer:
    :param age_embed_layer:
    :param job_embed_layer:
    :return:
    """
    #第一层全连接
    uid_fc_layer=tf.keras.layers.Dense(embed_dim,name='uid_fc_layer',activation='relu')(uid_embed_layer)
    gender_fc_layer=tf.keras.layers.Dense(embed_dim,name='gender_fc_layer',activation='relu')(gender_embed_layer)
    age_fc_layer=tf.keras.layers.Dense(embed_dim,name='age_fc_layer',activation='relu')(age_embed_layer)
    job_fc_layer=tf.keras.layers.Dense(embed_dim,name='job_fc_layer',activation='relu')(job_embed_layer)

    #第二层全连接
    user_combine_layer=tf.keras.layers.concatenate([uid_fc_layer,gender_fc_layer,age_fc_layer,job_fc_layer],2) #(?,1,4*embed_dim)
    user_combine_layer=tf.keras.layers.Dense(200,activation='tanh')(user_combine_layer) #(?,4*embed_dim,200)

    user_combine_layer_flat=tf.keras.layers.Reshape([200],name='user_combine_layer_flat')(user_combine_layer)
    return user_combine_layer,user_combine_layer_flat

def get_movie_id_embed_layer(movie_id):
    movie_id_embed_layer=tf.keras.layers.Embedding(movie_categories_max,embed_dim,input_length=18,name='movie_id_embed_layer')(movie_id)
    return movie_id_embed_layer

def get_movie_categories_layers(movie_categories):
    movie_categories_embed_layer=tf.keras.layers.Embedding(movie_categories_max,embed_dim,input_length=18,name='movie_categories_embed_layer')(movie_categories)
    movie_categories_embed_layer=tf.keras.layers.Lambda(lambda layer:tf.reduce_sum(layer,axis=1,keepdims=True))(movie_categories_embed_layer)
    return movie_categories_embed_layer

def get_movie_cnn_layer(movie_titles):
    #从嵌入矩阵中得到电影名对应的各个单词的嵌入向量
    movie_title_embed_layer=tf.keras.layers.Embedding(movie_title_max,embed_dim,input_length=15,name='movie_title_embed_layer')(movie_titles)
    sp=movie_title_embed_layer.shape
    movie_title_embed_layer_expand=tf.keras.layers.Reshape([sp[1],sp[2],1])(movie_title_embed_layer)

    #对文本嵌入层使用不同尺寸的卷据核做卷积核最大池化
    pool_layer_lst=[]
    for window_size in window_sizes:
        conv_layer=tf.keras.layers.Conv2D(filter_num,(window_size,embed_dim),1,activation='relu')(movie_title_embed_layer_expand)
        maxpool_layer=tf.keras.layers.MaxPooling2D(pool_size=(sentences_size-window_size+1,1),stride=1)(conv_layer)
        pool_layer_lst.append(maxpool_layer)
    #Dropout
    pool_layer=tf.keras.layers.concatenate(pool_layer_lst,3,name='pool_layer')
    max_num=len(window_sizes)*filter_num
    pool_layer_flat=tf.keras.layers.Reshape([1,max_num],name='pool_layer_flat')(pool_layer)

    dropout_layer=tf.keras.layers.Dropout(dropout_keep,name='dropout_layer')(pool_layer_flat)

    return pool_layer_flat,dropout_layer

def get_movie_feature_layer(movie_id_embed_layer,movie_categories_embed_layer,dropout_layer):
    #第一层全连接
    movie_id_fc_layer=tf.keras.layers.Dense(embed_dim,name='movie_id_fc_layer',activation='relu')(movie_id_embed_layer)
    movie_categories_fc_layer=tf.keras.layers.Dense(embed_dim,name='movie_categories_fc_layer',activation='relu')(movie_categories_embed_layer)
    #第二层连接
    movie_conbine_layer=tf.keras.layers.concatenate([movie_id_fc_layer,movie_categories_fc_layer,dropout_layer],2)
    movie_conbine_layer=tf.keras.layers.Dense(200,activation='tanh')(movie_conbine_layer)

    movie_combine_layer_flat=tf.keras.layers.Reshape([200],name='movie_combine_layer_flat')(movie_conbine_layer)
    return movie_conbine_layer,movie_combine_layer_flat

MODEL_DIR='./models'

class mv_network():
    def __init__(self,batch_size=256):
        self.batch_size=batch_size
        self.best_loss=9999
        self.losses={'train':[],'test':[]}

        #获取输入占位符
        uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles=get_inputs()
        #获取User的4个嵌入向量
        uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer=get_user_embedding(uid,user_gender,user_age,user_age)
        #得到用户特征
        user_combine_layer, user_combine_layer_flat=get_user_feature_layer(uid_embed_layer,gender_embed_layer,age_embed_layer,job_embed_layer)

        #获取电影ID的嵌入向量
        movie_id_embed_layer=get_movie_id_embed_layer(movie_id)
        #获取电影类型的嵌入向量
        movie_categories_embed_layer=get_movie_categories_layers(movie_categories)
        #获取电影名的特征向量
        pool_layer_flat,dropout_layer=get_movie_cnn_layer(movie_titles)
        #得到电影特征
        movie_combine_layer,movie_combine_layer_flat=get_movie_feature_layer(movie_id_embed_layer,movie_categories_embed_layer,dropout_layer)

        #计算出评分
        #将用户特征核电影特征做矩阵乘法得到一个预测评分的方案
        inference=tf.keras.layers.Lambda(lambda layer:
                                         tf.reduce_sum(layer[0]*layer[1],axis=1),name='inference')((user_combine_layer_flat,movie_combine_layer_flat))
        inference=tf.keras.layers.Lambda(lambda layer:tf.expand_dims(layer,axis=1))(inference)

        self.model=tf.keras.Model(
            inputs=[uid,user_gender,user_age,user_job,movie_id,movie_categories,movie_titles],
            outputs=[inference]
        )
        self.model.summary()

        self.optimizer=tf.keras.optimizer.Adam(learning_rate)
        #MSE损失，将计算值回归到评分
        self.ComputeLoss=tf.keras.losses.MeanSquareError()
        self.ComputeMetrics=tf.keras.metrics.MeanAbsoluteError()

        if tf.io.gfile.exists(MODEL_DIR):
            pass
        else:
            tf.io.gfile.makedirs(MODEL_DIR)

        checkpoint_dir=os.path.join(MODEL_DIR,'checkpoints')
        self.checkpoint_prefix=os.path.join(checkpoint_dir,'ckpt')
        self.checkpoint=tf.train.Checkpoint(model=self.model,optimizer=self.optimizer)

        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    def compute_loss(self,labels,logits):
        return tf.reduce_mean(tf.keras.losses.mse(labels,logits))

    def compute_metrics(self,labels,logits):
        return tf.keras.metrics.mae(labels,logits)

    @tf.function
    def train_step(self,x,y):
        with tf.GradientTape() as tape:
            logits=self.model([x[0],
                               x[1],
                               x[2],
                               x[3],
                               x[4],
                               x[5],
                               x[6]
                               ],training=True)
            loss=self.ComputeLoss(y,logits)
            self.ComputeMetrics(y,logits)

        grads=tape.gradient(loss,self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables))
        return loss,logits

    def training(self,features,targets_values,epochs=5,log_freq=50):
        for epoch_i in range(epochs):
            #将数据集分成训练集和测试集，随机种子不固定
            train_X,test_X,train_y,test_y=train_test_split(features,targets_values,test_size=0.2,random_state=0)
            train_batches=get_batches(train_X,train_y,self.batch_size)
            batch_num=(len(train_X)//self.batch_size)

            train_start=time.time()
            if True:
                start=time.time()
                avg_loss=tf.keras.metrics.Mean('loss',dtype=tf.float32)
                for batch_i in range(batch_num):
                    x,y=next(train_batches)
                    categories=np.zeros([self.batch_size,18])
                    for i in range(self.batch_size):
                        categories[i]=x.take(6,1)[i]
                    titles=np.zeros([self.batch_size,sentences_size])
                    for i in range(self.batch_size):
                        titles[i]=x.take(5,1)[i]
                    loss,logits=self.train_step([np.reshape(x.take(0,1),[self.batch_size,1]).astype(np.float32),
                                                 np.reshape(x.take(2,1),[self.batch_size,1]).astype(np.float32),
                                                 np.reshape(x.take(3,1),[self.batch_size,1]).astype(np.float32),
                                                 np.reshape(x.take(4,1),[self.batch_size,1]).astype(np.float32),
                                                 np.reshape(x.take(1,1),[self.batch_size,1]).astype(np.float32),
                                                 categories.astype(np.float32),
                                                 titles.astype(np.float32)
                                                 ],np.reshape(y,[self.batch_size,1]).astype(np.float32))

                    avg_loss(loss)
                    self.losses['train'].append(loss)

                    if tf.equal(self.optimizer.iterations % log_freq,0):
                        rate=log_freq/(time.time()-start)



def get_batches(Xs,ys,batch_size):
    for start in range(0,len(Xs),batch_size):
        end=min(start+batch_size,len(Xs))
        yield Xs[start:end],ys[start:end]

# coding: utf-8
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used
os.environ['TF_KERAS'] = '1'
import numpy as np
import pandas as pd
from glob import glob
import csv
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow import dtypes
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Flatten,
    Dense,
    concatenate,
    Input,
    Dropout,
    Activation
)
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.metrics import accuracy_score
import pickle
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

picSize=3
embedding_size = 9
input_image_shape = (picSize, picSize, 1)
bs = embedding_size

def categorical_squared_hinge(y_true, y_pred):
    y_true = 2. * y_true - 1
    vvvv = K.maximum(1. - y_true * y_pred, 0.)
    vv = K.sum(vvvv, 1, keepdims=False)
    v = K.mean(vv, axis=-1)
    return v

def pairwise_distance(feature, squared=False):
    pairwise_distances_squared = math_ops.add(
        math_ops.reduce_sum(math_ops.square(feature), axis=[1], keepdims=True),
        math_ops.reduce_sum(
            math_ops.square(array_ops.transpose(feature)),
            axis=[0],
            keepdims=True)) - 2.0 * math_ops.matmul(feature,
                                                    array_ops.transpose(feature))
    pairwise_distances_squared = math_ops.maximum(pairwise_distances_squared, 0.0)
    error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        # sqrt 前 clip，避免負數
        safe_val = tf.clip_by_value(pairwise_distances_squared + tf.cast(error_mask, tf.float32) * 1e-16, 0.0, np.inf)
        pairwise_distances = math_ops.sqrt(safe_val)
    pairwise_distances = math_ops.multiply(
        pairwise_distances, tf.cast(math_ops.logical_not(error_mask), tf.float32))
    num_data = array_ops.shape(feature)[0]
    mask_offdiagonals = array_ops.ones_like(pairwise_distances) - array_ops.diag(
        array_ops.ones([num_data]))
    pairwise_distances = math_ops.multiply(pairwise_distances, mask_offdiagonals)
    return pairwise_distances

def masked_maximum(data, mask, dim=1):
    axis_minimums = math_ops.reduce_min(data, dim, keepdims=True)
    masked_maximums = math_ops.reduce_max(
        math_ops.multiply(data - axis_minimums, mask), dim,
        keepdims=True) + axis_minimums
    return masked_maximums

def masked_minimum(data, mask, dim=1):
    axis_maximums = math_ops.reduce_max(data, dim, keepdims=True)
    masked_minimums = math_ops.reduce_min(
        math_ops.multiply(data - axis_maximums, mask), dim,
        keepdims=True) + axis_maximums
    return masked_minimums

def triplet_loss_adapted_from_tf(y_true, y_pred):
    del y_true
    margin = 1.
    labels = y_pred[:, :1] 
    labels = tf.cast(labels, dtype='int32')
    embeddings = y_pred[:, 1:]
    pdist_matrix = pairwise_distance(embeddings, squared=True)
    adjacency = math_ops.equal(labels, array_ops.transpose(labels))
    adjacency_not = math_ops.logical_not(adjacency)
    batch_size = array_ops.size(labels)
    pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1])
    mask = math_ops.logical_and(
        array_ops.tile(adjacency_not, [batch_size, 1]),
        math_ops.greater(
            pdist_matrix_tile, array_ops.reshape(
                array_ops.transpose(pdist_matrix), [-1, 1])))
    mask_final = array_ops.reshape(
        math_ops.greater(
            math_ops.reduce_sum(
                math_ops.cast(mask, dtype=dtypes.float32), 1, keepdims=True),
            0.0), [batch_size, batch_size])
    mask_final = array_ops.transpose(mask_final)
    adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32)
    mask = math_ops.cast(mask, dtype=dtypes.float32)
    negatives_outside = array_ops.reshape(
        masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
    negatives_outside = array_ops.transpose(negatives_outside)
    negatives_inside = array_ops.tile(
        masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
    semi_hard_negatives = array_ops.where(
        mask_final, negatives_outside, negatives_inside)
    loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives)
    mask_positives = math_ops.cast(
        adjacency, dtype=dtypes.float32) - array_ops.diag(
        array_ops.ones([batch_size]))
    num_positives = math_ops.reduce_sum(mask_positives)
    semi_hard_triplet_loss_distance = math_ops.truediv(
        math_ops.reduce_sum(
            math_ops.maximum(
                math_ops.multiply(loss_mat, mask_positives), 0.0)),
        num_positives,
        name='triplet_semihard_loss')
    return semi_hard_triplet_loss_distance

def create_base_network(image_input_shape, embedding_size):   
    model = Sequential()
    inputShape = (picSize, picSize, 1)
    chanDim = -1
    model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())    
    model.add(Dense(embedding_size))
    base_network=model
    return base_network

data = glob(os.path.join("tr0.xlsx"))
data = np.sort(data)
iN=9
k=0
u1 = pd.read_excel(data[0],usecols="A:I",sheet_name ="Sheet1")
n1Size=int(u1.size/iN)
x=[]
y=[]
for j in range(0,n1Size-1):
    u1i=u1[j:(j+1)]
    u1i = u1i.astype('uint32') # float16 , uint8
    u1i = u1i.values.reshape((u1i.shape[0],-1))
    if (k>0):
        tmp=np.squeeze(u1i)              
        xa=tmp.reshape([1,3,3,1])
        ya=[0]
        x=np.concatenate((x,xa), axis=0)
        y=np.concatenate((y,ya), axis=0)             
            
    if (k==0):
        tmp=np.squeeze(u1i)     
        x=tmp.reshape([1,3,3,1])
        y=[0]
        k=k+1          

x0=x
y0=y

data = glob(os.path.join("tr1.xlsx"))
data = np.sort(data)
iN=9
k=0
u1 = pd.read_excel(data[0],usecols="A:I",sheet_name ="Sheet1")
n1Size=int(u1.size/iN)
x=[]
y=[]
for j in range(0,n1Size-1):
    u1i=u1[j:(j+1)]
    u1i = u1i.astype('uint32') # float16 , uint8
    u1i = u1i.values.reshape((u1i.shape[0],-1))
    if (k>0):
        tmp=np.squeeze(u1i)              
        xa=tmp.reshape([1,3,3,1])
        ya=[1]
        x=np.concatenate((x,xa), axis=0)
        y=np.concatenate((y,ya), axis=0)             
            
    if (k==0):
        tmp=np.squeeze(u1i)     
        x=tmp.reshape([1,3,3,1])
        y=[1]
        k=k+1          

x1=x
y1=y

x=np.concatenate((x0,x1), axis=0)/266500.0
y=np.concatenate((y0,y1), axis=0)

# 讀取 tr0/tr1 完後，將 x, y 複製給 x_tr, y_tr
x_tr = x
y_tr = y

data = glob(os.path.join("ts0.xlsx"))
data = np.sort(data)
iN=9
k=0
u1 = pd.read_excel(data[0],usecols="A:I",sheet_name ="Sheet1")
n1Size=int(u1.size/iN)
x=[]
y=[]
for j in range(0,n1Size-1):
    u1i=u1[j:(j+1)]
    u1i = u1i.astype('uint32') # float16 , uint8
    u1i = u1i.values.reshape((u1i.shape[0],-1))
    if (k>0):
        tmp=np.squeeze(u1i)              
        xa=tmp.reshape([1,3,3,1])
        ya=[0]
        x=np.concatenate((x,xa), axis=0)
        y=np.concatenate((y,ya), axis=0)             
            
    if (k==0):
        tmp=np.squeeze(u1i)     
        x=tmp.reshape([1,3,3,1])
        y=[0]
        k=k+1          

x0=x
y0=y

data = glob(os.path.join("ts1.xlsx"))
data = np.sort(data)
iN=9
k=0
u1 = pd.read_excel(data[0],usecols="A:I",sheet_name ="Sheet1")
n1Size=int(u1.size/iN)
x=[]
y=[]
for j in range(0,n1Size-1):
    u1i=u1[j:(j+1)]
    u1i = u1i.astype('uint32') # float16 , uint8
    u1i = u1i.values.reshape((u1i.shape[0],-1))
    if (k>0):
        tmp=np.squeeze(u1i)              
        xa=tmp.reshape([1,3,3,1])
        ya=[1]
        x=np.concatenate((x,xa), axis=0)
        y=np.concatenate((y,ya), axis=0)             
            
    if (k==0):
        tmp=np.squeeze(u1i)     
        x=tmp.reshape([1,3,3,1])
        y=[1]
        k=k+1          

x1=x
y1=y

x=np.concatenate((x0,x1), axis=0)/266500.0
y=np.concatenate((y0,y1), axis=0)

# 讀取 ts0/ts1 完後，將 x, y 複製給 x_ts, y_ts
# 注意：此時 x, y 已被 ts0/ts1 覆蓋
x_ts = x
y_ts = y

dummy_gt_train = np.zeros((len(x_tr), embedding_size + 1))
x_tr = np.reshape(x_tr, (len(x_tr),  picSize, picSize, 1))
x_ts = np.reshape(x_ts, (len(x_ts), picSize, picSize, 1))

model=load_model("triplet.h5",custom_objects={'triplet_loss_adapted_from_tf':triplet_loss_adapted_from_tf})

embeddings = create_base_network(input_image_shape,embedding_size=embedding_size)
for layer_target, layer_source in zip(embeddings.layers, model.layers[2].layers):
  weights = layer_source.get_weights()
  layer_target.set_weights(weights)
  del weights

x_train_embeddings = embeddings.predict(x_tr)
x_tr_emb=np.array(x_train_embeddings)

x_test_embeddings = embeddings.predict(x_ts)
x_ts_emb=np.array(x_test_embeddings)

no_of_components = 3
class_labels = np.unique(y_tr)
pca = PCA(n_components=no_of_components)
decomposed_embeddings = pca.fit_transform(x_tr_emb)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
step = 1
for label in class_labels:
    decomposed_embeddings_class = decomposed_embeddings[y_tr == label]
    ax.scatter(decomposed_embeddings_class[::step, 0], decomposed_embeddings_class[::step, 1], decomposed_embeddings_class[::step, 2], label="Class "+str(int(label)))
    if int(label)==0:
        d0tr=decomposed_embeddings_class[::1, :]
        y0tr = np.ones((len(d0tr),),dtype='uint8')*0
    if int(label)==1:
        d1tr=decomposed_embeddings_class[::1, :]
        y1tr = np.ones((len(d1tr),),dtype='uint8')*1
    plt.legend()
plt.title('PCA features')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')    
#plt.savefig('pca3tr.jpg')

class_labels = np.unique(y_ts)
decomposed_embeddings = pca.transform(x_ts_emb)
for label in class_labels:
    decomposed_embeddings_class = decomposed_embeddings[y_ts == label]
    if int(label)==0:
        d0ts=decomposed_embeddings_class[::1, :]
        y0ts = np.ones((len(d0ts),),dtype='uint8')*0
    if int(label)==1:
        d1ts=decomposed_embeddings_class[::1, :]
        y1ts = np.ones((len(d1ts),),dtype='uint8')*1

# 不要覆蓋 x_tr, y_tr 為降維後的資料，只保留原始影像 shape
# 移除以下覆蓋：
# x_tr=np.concatenate((d0tr,d1tr), axis=0)
# y_tr=np.concatenate((y0tr,y1tr), axis=0)
# x_ts=np.concatenate((d0ts,d1ts), axis=0)
# y_ts=np.concatenate((y0ts,y1ts), axis=0)

# 5-fold cross-validation
from sklearn.model_selection import StratifiedKFold
# 印出 ts0.xlsx, ts1.xlsx 樣本數
ts0 = pd.read_excel('ts0.xlsx', usecols="A:I", sheet_name="Sheet1")
ts1 = pd.read_excel('ts1.xlsx', usecols="A:I", sheet_name="Sheet1")
n_ts0 = ts0.shape[0]
n_ts1 = ts1.shape[0]
print(f"Test set: ts0.xlsx (class 0): {n_ts0} samples, ts1.xlsx (class 1): {n_ts1} samples")
# 印出交叉驗證前初始資料集 class 分布
n_init_0 = np.sum(y_tr == 0)
n_init_1 = np.sum(y_tr == 1)
print(f"Initial dataset: class=0: {n_init_0}, class=1: {n_init_1}")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_tpr, all_tnr, all_acc, all_precision = [], [], [], []
all_probs = []
all_y_true = []
X = x_tr  # shape: (N, 3, 3, 1)
y = y_tr  # shape: (N,)
input_shape = (picSize, picSize, 1)
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"\n===== Fold {fold+1} =====")
    x_tr_cv, x_ts_cv = X[train_idx], X[test_idx]
    y_tr_cv, y_ts_cv = y[train_idx], y[test_idx]
    # 補充 ts1.xlsx 正例到每折驗證集
    ts1 = pd.read_excel('ts1.xlsx', usecols="A:I", sheet_name="Sheet1")
    n1Size = int(ts1.size / 9)
    x1ts = []
    for j in range(0, n1Size-1):
        u1i = ts1[j:(j+1)].astype('uint32').values.reshape((1, 3, 3, 1))
        if j == 0:
            x1ts = u1i
        else:
            x1ts = np.concatenate((x1ts, u1i), axis=0)
    x1ts = x1ts / 266500.0
    y1ts = np.ones(len(x1ts), dtype='uint8')
    # 合併到驗證集
    x_ts_cv = np.concatenate((x_ts_cv, x1ts), axis=0)
    y_ts_cv = np.concatenate((y_ts_cv, y1ts), axis=0)
    # 補充 ts0.xlsx 負例到每折驗證集
    ts0 = pd.read_excel('ts0.xlsx', usecols="A:I", sheet_name="Sheet1")
    n0Size = int(ts0.size / 9)
    x0ts = []
    for j in range(0, n0Size-1):
        u0i = ts0[j:(j+1)].astype('uint32').values.reshape((1, 3, 3, 1))
        if j == 0:
            x0ts = u0i
        else:
            x0ts = np.concatenate((x0ts, u0i), axis=0)
    x0ts = x0ts / 266500.0
    y0ts = np.zeros(len(x0ts), dtype='uint8')
    x_ts_cv = np.concatenate((x_ts_cv, x0ts), axis=0)
    y_ts_cv = np.concatenate((y_ts_cv, y0ts), axis=0)
    print(f"x_tr_cv: min={np.min(x_tr_cv):.6f}, max={np.max(x_tr_cv):.6f}, mean={np.mean(x_tr_cv):.6f}")
    # 僅載入triplet.h5，不微調
    model = load_model("triplet.h5", compile=False)
    embeddings = create_base_network(input_shape, embedding_size=embedding_size)
    for t, s in zip(embeddings.layers, model.layers[2].layers):
        t.set_weights(s.get_weights())
    x_tr_emb = embeddings.predict(x_tr_cv)
    x_ts_emb = embeddings.predict(x_ts_cv)
    print(f"x_tr_emb: min={np.min(x_tr_emb):.6f}, max={np.max(x_tr_emb):.6f}, mean={np.mean(x_tr_emb):.6f}")
    print(f"x_ts_emb: min={np.min(x_ts_emb):.6f}, max={np.max(x_ts_emb):.6f}, mean={np.mean(x_ts_emb):.6f}")
    # 載入固定的PCA模型
    with open('pca_model.pkl', 'rb') as f:
        pca = pickle.load(f)
    d_tr = pca.transform(x_tr_emb)
    d_ts = pca.transform(x_ts_emb)
    # 依 label 分群
    x0tr = d_tr[y_tr_cv == 0]
    y0tr = np.zeros(len(x0tr), dtype='uint8')
    x1tr = d_tr[y_tr_cv == 1]
    y1tr = np.ones(len(x1tr), dtype='uint8')
    x0ts = d_ts[y_ts_cv == 0]
    y0ts = np.zeros(len(x0ts), dtype='uint8')
    x1ts = d_ts[y_ts_cv == 1]
    y1ts = np.ones(len(x1ts), dtype='uint8')
    x_tr_xgb = np.concatenate((x0tr, x1tr), axis=0)
    y_tr_xgb = np.concatenate((y0tr, y1tr), axis=0).astype(np.int32)
    x_ts_xgb = np.concatenate((x0ts, x1ts), axis=0)
    y_ts_xgb = np.concatenate((y0ts, y1ts), axis=0).astype(np.int32)
    # 檢查標籤內容
    assert set(np.unique(y_tr_xgb)).issubset({0,1}), f"y_tr_xgb labels: {np.unique(y_tr_xgb)}"
    assert set(np.unique(y_ts_xgb)).issubset({0,1}), f"y_ts_xgb labels: {np.unique(y_ts_xgb)}"
    # 印出訓練集與測試集各類別樣本數
    n_tr_0 = np.sum(y_tr_xgb == 0)
    n_tr_1 = np.sum(y_tr_xgb == 1)
    n_ts_0 = np.sum(y_ts_xgb == 0)
    n_ts_1 = np.sum(y_ts_xgb == 1)
    print(f"Train class=0: {n_tr_0}, class=1: {n_tr_1}; Test class=0: {n_ts_0}, class=1: {n_ts_1}")
    # 每折微調XGBoost或載入現有模型
    dtrain = xgb.DMatrix(x_tr_xgb, label=y_tr_xgb)
    dtest = xgb.DMatrix(x_ts_xgb, label=y_ts_xgb)    
    # 僅用於預測，不要繼續訓練
    xgb_model = xgb.Booster()
    xgb_model.load_model('xgb_model.model')
    print('Loaded existing xgb_model.model')
    #print("XGBoost model config:\n", xgb_model.save_config())    
    # 使用最佳樹數量進行預測
    best_ntree_limit = None
    if 'best_ntree_limit' in xgb_model.attributes():
        best_ntree_limit = int(xgb_model.attributes()['best_ntree_limit'])
        print(f"Use best_ntree_limit={best_ntree_limit} for prediction")
    else:
        best_ntree_limit = xgb_model.num_boosted_rounds()
        print(f"No best_ntree_limit found, use all {best_ntree_limit} trees")
    # 預測分數（margin），再轉成機率（確保 ROC 真的是用連續值）
    # 若模型是 binary:hinge，predict() 只會回傳 0/1，必須用 output_margin
    raw_scores = xgb_model.predict(dtest, ntree_limit=best_ntree_limit, output_margin=True)
    raw_scores = np.asarray(raw_scores)
    # 若回傳為 (N,2) 或 (N,C)，只取正類的 margin
    if raw_scores.ndim > 1:
        raw_scores = raw_scores[:, -1]
    raw_scores = raw_scores.ravel()
    probs_ts = 1.0 / (1.0 + np.exp(-raw_scores))
    preds_ts = (probs_ts >= 0.5).astype(np.int32)
    preds0_ts = xgb_model.predict(xgb.DMatrix(x0ts), ntree_limit=best_ntree_limit).astype(int)
    preds1_ts = xgb_model.predict(xgb.DMatrix(x1ts), ntree_limit=best_ntree_limit).astype(int)
    acc = accuracy_score(y_ts_xgb, preds_ts)
    from sklearn.metrics import precision_score
    precision = precision_score(y_ts_xgb, preds_ts, zero_division=0)
    TP = np.sum(preds1_ts == 1)
    P = len(y1ts)
    TN = np.sum(preds0_ts == 0)
    N = len(y0ts)
    TPR = TP / P if P > 0 else 0
    TNR = TN / N if N > 0 else 0
    print(f"TPR={TPR:.4f}, TNR={TNR:.4f}, ACC={acc:.4f}, Precision={precision:.4f}")
    all_tpr.append(TPR)
    all_tnr.append(TNR)
    all_acc.append(acc)
    all_precision.append(precision)

    # 收集 test set 預測結果（用於整體 ROC）
    all_probs.append(probs_ts)
    all_y_true.append(y_ts_xgb)

    # 計算並儲存 ROC curve（每 fold）
    fpr, tpr, _ = roc_curve(y_ts_xgb, probs_ts)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'Fold {fold+1} ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Fold {fold+1}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'roc_fold_{fold+1}.png')
    plt.close()

# ==============================
# 單一 Test Set 的整體 ROC Curve
# ==============================
all_probs = np.concatenate(all_probs, axis=0)
all_y_true = np.concatenate(all_y_true, axis=0)
fpr_all, tpr_all, _ = roc_curve(all_y_true, all_probs)
auc_all = auc(fpr_all, tpr_all)

# ==============================
# Baseline: XGBoost only (RAW features, sklearn API, true predict_proba)
# ==============================
from xgboost import XGBClassifier

# 使用原始 3x3=9 維資料作為特徵
x_tr_raw = x_tr.reshape(len(x_tr), -1)
x_ts_raw = x_ts.reshape(len(x_ts), -1)

xgb_baseline = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    max_depth=6,
    learning_rate=0.1,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)

xgb_baseline.fit(x_tr_raw, y_tr.astype(np.int32))

# sklearn API：明確取得正類機率
baseline_probs = xgb_baseline.predict_proba(x_ts_raw)[:, 1]

# debug: 確認真的是連續機率
print('[Baseline XGB] prob stats:', 'min=', baseline_probs.min(), 'max=', baseline_probs.max(),
      'unique≈', len(np.unique(np.round(baseline_probs, 4))))

fpr_base, tpr_base, _ = roc_curve(y_ts, baseline_probs)
auc_base = auc(fpr_base, tpr_base)

# ==============================
# Plot comparison ROC
# ==============================
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# ==============================
# Train multiple RAW baselines: LR / RF / GB / XGB
# ==============================
x_tr_raw = x_tr.reshape(len(x_tr), -1)
x_ts_raw = x_ts.reshape(len(x_ts), -1)

baselines = {
    'LR': LogisticRegression(max_iter=2000, solver='lbfgs'),
    'RF': RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1),
    'GB': GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, random_state=42),
    'XGB': XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective='binary:logistic', eval_metric='auc',
        random_state=42, verbosity=0
    ),
}

baseline_rocs = {}
for name, clf in baselines.items():
    clf.fit(x_tr_raw, y_tr.astype(np.int32))
    probs = clf.predict_proba(x_ts_raw)[:, 1]
    fpr_b, tpr_b, _ = roc_curve(y_ts, probs)
    auc_b = auc(fpr_b, tpr_b)
    baseline_rocs[name] = (fpr_b, tpr_b, auc_b)
    print(f'[Baseline {name}] AUC={auc_b:.3f}, prob range=({probs.min():.4f},{probs.max():.4f})')

# ==============================
# Plot multi-baseline ROC comparison
# ==============================
plt.figure(figsize=(7,7))
plt.plot(fpr_all, tpr_all, label=f'Proposed (Triplet+PCA+XGB) AUC={auc_all:.3f}', linewidth=3, color='black')

colors = {'LR':'tab:blue','RF':'tab:orange','GB':'tab:green','XGB':'tab:red'}
for name, (fpr_b, tpr_b, auc_b) in baseline_rocs.items():
    plt.plot(fpr_b, tpr_b, label=f'{name} (raw) AUC={auc_b:.3f}', linewidth=2, color=colors.get(name,None))

plt.plot([0,1],[0,1],'k--', alpha=0.6)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Comparison: Proposed vs Multiple Raw Baselines')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_compare_multi_baselines.png')
plt.close()

# 統計5-fold平均與標準差
mean_tpr = np.mean(all_tpr)
std_tpr = np.std(all_tpr)
mean_tnr = np.mean(all_tnr)
std_tnr = np.std(all_tnr)
mean_acc = np.mean(all_acc)
std_acc = np.std(all_acc)
mean_precision = np.mean(all_precision)
std_precision = np.std(all_precision)
print(f"\n5-Fold 平均 TPR: {mean_tpr:.4f} ± {std_tpr:.4f}, TNR: {mean_tnr:.4f} ± {std_tnr:.4f}, ACC: {mean_acc:.4f} ± {std_acc:.4f}, Precision: {mean_precision:.4f} ± {std_precision:.4f}")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np



# import tensorflow as tf
# import pandas as pd
# import sklearn.model_selection as sk

# useCols = ['Rk','Player','Tm','FantPos','Age','G','GS','Cmp','Att','Yds','TD','Int','Att','Yds','Y/A','TD','Tgt','Rec','Yds','Y/R','TD','Fmb','FL','TD','2PM','2PP','FantPt','PPR','DKPt','FDPt','VBD','PosRank','OvRank']

# player_data = pd.read_csv('football.csv')

# X_train, X_test, y_train, y_test = sk.train_test_split(player_data, useCols, test_size=0.33, random_state=42)

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(33, activation=tf.nn.relu, input_shape=len(useCols)),
#     tf.keras.layers.Dense(33, activation=tf.nn.relu),
#     tf.keras.layers.Dense(15)
# ])
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# def loss(model, x, y, training):
#     y_ = model(x, training=training)
#     return loss_object(y_true=y, y_pred=y_)
# l = loss(model, X_train, useCols, training=False)

# def grad(model, inputs, targets):
#   with tf.GradientTape() as tape:
#     loss_value = loss(model, inputs, targets, training=True)
#   return loss_value, tape.gradient(loss_value, model.trainable_variables)

# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# loss_value, grads = grad(model, X_train, useCols)

# optimizer.apply_gradients(zip(grads, model.trainable_variables))
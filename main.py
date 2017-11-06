import tensorflow as tf
import numpy as np

data_file = "./data/all_scripts.txt"
text = open(data_file).read().strip()
vocab = sorted(list(set(text)))
vocab_length = len(vocab)
characters2id = dict((c, i) for i, c in enumerate(vocab))
id2characters = dict((i, c) for i, c in enumerate(vocab))
section_length = 50
sections = []
section_labels = []
for i in range(len(text)/section_length):
    sections.append(text[i*section_length:(i+1)*section_length])
    section_labels.append(text[(i+1)*section_length])

X_data = np.zeros((len(sections),section_length,vocab_length))
Y_data = np.zeros((len(sections),vocab_length))
for i,section in enumerate(sections):
    for j,letter in enumerate(section):
        X_data[i,j,characters2id[letter]] = 1
    Y_data[i,characters2id[section_labels[i]]] = 1

print X_data.shape,Y_data.shape

batch_size = 64
epochs = 50000
log_every = 1000

hidden_nodes = 32
start_text = "Hey cartman"

checkpoint_dir = "model"
if tf.gfile.Exists(checkpoint_dir):
    tf.gfile.DeleteRecursively(checkpoint_dir)
tf.gfile.MakeDirs(checkpoint_dir)
X = tf.placeholder(tf.float32,[batch_size,section_length,vocab_length])
Y = tf.placeholder(tf.float32,[batch_size,vocab_length])

cell = tf.nn.rnn_cell.LSTMCell(hidden_nodes,state_is_tuple=True)
all_outputs, state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

all_outputs = tf.transpose(all_outputs,[1,0,2])[-1]

W = tf.Variable(tf.truncated_normal([hidden_nodes,vocab_length],-0.1,0.1))
b = tf.Variable(tf.zeros([vocab_length]))

logits = tf.matmul(all_outputs,W)+b

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y))
optimizer = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    for epoch in range(epochs):
        # print epoch
        loss_val = 0
        for i in range(len(sections)/batch_size):
            batch_data = X_data[i*batch_size:(i+1)*batch_size]
            batch_labels = Y_data[i*batch_size:(i+1)*batch_size]
            _,loss_val = sess.run([optimizer,loss],feed_dict={X:batch_data,Y:batch_labels})
        if epoch%25==0 or epoch==0:
            print "epoch: {}  loss: {}\n".format(epoch,loss_val)

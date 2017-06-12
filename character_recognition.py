from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

image_size = 28
num_labels = 10

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  #Perform one-hot encoding
  labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
  return dataset, labels

def load_data(path):
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)
    return {'train_dataset':train_dataset, 'train_labels':train_labels, 'valid_dataset':valid_dataset,
            'valid_labels':valid_labels, 'test_dataset':test_dataset, 'test_labels':test_labels}

def apply_mini_batch(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels):
    batch_size = 128
    num_steps = 3001
    graph = tf.Graph()
    with graph.as_default():
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
        weights = tf.Variable(
            tf.truncated_normal([image_size*image_size, num_labels])
        )
        biases = tf.Variable(tf.zeros([num_labels]))
        logits = tf.matmul(tf_train_dataset, weights) + biases
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)
        )
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(
            tf.matmul(tf_valid_dataset, weights) + biases
        )
        test_prediction = tf.nn.softmax(
            tf.matmul(tf_test_dataset, weights) + biases
        )
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset+batch_size), :]
            batch_labels = train_labels[offset:(offset+batch_size), :]
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels:batch_labels}
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict = feed_dict
            )
            if (step % 500 == 0):
                print("Minibatch loss at step %d = %f" % (step, l))
                print("Minibatch accuracy %f" % accuracy(predictions, batch_labels))
                print("Validation accuracy %f" % accuracy(valid_prediction.eval(), valid_labels))
        print("Test accuracy %f" % accuracy(test_prediction.eval(), test_labels))

def apply_gradient_descent(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels):

    num_steps = 801
    graph = tf.Graph()
    with graph.as_default():
        tf_train_dataset = tf.constant(train_dataset, dtype=tf.float32)
        tf_train_labels = tf.constant(train_labels, dtype=tf.float32)
        tf_valid_dataset = tf.constant(valid_dataset, dtype=tf.float32)
        tf_valid_labels = tf.constant(valid_labels, dtype=tf.float32)
        tf_test_dataset = tf.constant(test_dataset, dtype=tf.float32)
        tf_test_labels = tf.constant(test_labels, dtype=tf.float32)

        weights = tf.Variable(
            tf.truncated_normal([image_size * image_size, num_labels])
        )
        biases = tf.Variable(tf.zeros([num_labels]))
        logits = tf.matmul(tf_train_dataset, weights) + biases
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)
        )
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(
            tf.matmul(tf_valid_dataset, weights) + biases
        )
        test_prediction = tf.nn.softmax(
            tf.matmul(tf_test_dataset, weights) + biases
        )
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        for step in range(num_steps):
            #print(step)
            _, l, predictions = session.run([optimizer, loss, train_prediction])
            if (step % 100 == 0):
                print('Loss at step %d = %f' % (step, l))
                print('Training accuracy %f' % accuracy(predictions, train_labels))
                print('Validation accuracy = %f' % accuracy(valid_prediction.eval(), valid_labels))
        print('Test accuracy = %f' % accuracy(test_prediction.eval(), test_labels))


def apply_neural_network(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels):
    graph = tf.Graph()
    hidden_nodes = 1024
    input_nodes = image_size * image_size
    output_nodes = num_labels
    epochs = 20
    batch_size = 250
    learning_rate = 0.05
    num_batch = int(train_labels.shape[0]/batch_size)

    with graph.as_default():
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        weights = {
            'hidden': tf.Variable(tf.random_normal([input_nodes, hidden_nodes])),
            'output': tf.Variable(tf.random_normal([hidden_nodes, output_nodes]))
        }
        biases = {
            'hidden': tf.Variable(tf.random_normal([hidden_nodes])),
            'output': tf.Variable(tf.random_normal([output_nodes]))
        }
        hidden_layer = tf.add(tf.matmul(tf_train_dataset, weights['hidden']), biases['hidden'])
        test_hidden_layer = tf.nn.relu(tf.add(tf.matmul(tf_test_dataset, weights['hidden']), biases['hidden']))
        test_output_layer = tf.add(tf.matmul(test_hidden_layer, weights['output']), biases['output'])
        hidden_layer = tf.nn.relu(hidden_layer)
        output_layer = tf.add(tf.matmul(hidden_layer, weights['output']), biases['output'])

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=tf_train_labels))
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

        prediction = tf.nn.softmax(output_layer)
        test_prediction = tf.nn.softmax(test_output_layer)

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        for epoch in range(epochs):
            for j in range(num_batch):
                offset = (j * batch_size) % (train_labels.shape[0] - batch_size)
                batch_data = train_dataset[offset:(offset+batch_size), :]
                batch_labels = train_labels[offset:(offset+batch_size), :]
                feed_dict = {tf_train_dataset:batch_data, tf_train_labels:batch_labels}

                _, l, predictions = session.run(
                    [optimizer, loss, prediction], feed_dict = feed_dict
                )
        #test_feed_dict = {tf_train_dataset: test_dataset[0:batch_size, :]}
        print("Test accuracy %f" % accuracy(test_prediction.eval(), test_labels))

def apply_neural_network_with_l2_reg(train_dataset, train_labels, valid_dataset,
                                    valid_labels, test_dataset, test_labels):
    train_subset = 10000
    beta = 0.01
    hidden_nodes = 1024
    input_nodes = image_size * image_size
    output_nodes = num_labels
    batch_size = 128
    learning_rate = 0.5
    num_batch = int(train_labels.shape[0]/batch_size)
    num_steps = 3001

    graph = tf.Graph()
    with graph.as_default():
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        weights = {
            'hidden': tf.Variable(tf.random_normal([input_nodes, hidden_nodes])),
            'output': tf.Variable(tf.random_normal([hidden_nodes, output_nodes]))
        }
        biases = {
            'hidden': tf.Variable(tf.random_normal([hidden_nodes])),
            'output': tf.Variable(tf.random_normal([output_nodes]))
        }
        hidden_layer = tf.add(tf.matmul(tf_train_dataset, weights['hidden']), biases['hidden'])
        hidden_layer = tf.nn.relu(hidden_layer)
        output_layer = tf.add(tf.matmul(hidden_layer, weights['output']), biases['output'])
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=tf_train_labels))
        regulizer = tf.nn.l2_loss(weights['hidden']) + tf.nn.l2_loss(weights['output'])
        loss = tf.reduce_mean(loss + beta * regulizer)
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        train_prediction = tf.nn.softmax(output_layer)

        #For validation
        validation_hidden_layer = tf.add(tf.matmul(tf_valid_dataset, weights['hidden']), biases['hidden'])
        validation_hidden_layer = tf.nn.relu(validation_hidden_layer)
        validation_output_layer = tf.add(tf.matmul(validation_hidden_layer, weights['output']), biases['output'])
        validation_prediction = tf.nn.softmax(validation_output_layer)

        #For testing
        test_hidden_layer = tf.add(tf.matmul(tf_test_dataset, weights['hidden']), biases['hidden'])
        test_hidden_layer = tf.nn.relu(test_hidden_layer)
        test_output_layer = tf.add(tf.matmul(test_hidden_layer, weights['output']), biases['output'])
        test_prediction = tf.nn.softmax(test_output_layer)

    with tf.Session(graph = graph) as session:
        tf.global_variables_initializer().run()
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict
                                            )
            if (step % 500 == 0):
                print("Minibatch loss = {}".format(l))
                print("Minibatch accuracy = {}".format(accuracy(predictions, batch_labels)))
                print("Validation accuracy = {}".format(accuracy(validation_prediction.eval(), valid_labels)))
        print("Test accuracy = {}".format(accuracy(test_prediction.eval(), test_labels)))

path = '../'
pickle_file = path + 'notMNIST.pickle'
print("reading dataset")
data = load_data(pickle_file)

train_dataset, train_labels = reformat(data['train_dataset'], data['train_labels'])
valid_dataset, valid_labels = reformat(data['valid_dataset'], data['valid_labels'])
test_dataset, test_labels = reformat(data['test_dataset'], data['test_labels'])
print("running batch gradient descent")
"""apply_gradient_descent(train_dataset=train_dataset[:10000, :], train_labels=train_labels[:10000],
                        valid_dataset=valid_dataset, valid_labels=valid_labels,
                        test_dataset=test_dataset, test_labels=test_labels
                        )"""
"""apply_mini_batch(train_dataset=train_dataset[:10000, :], train_labels=train_labels[:10000],
                        valid_dataset=valid_dataset, valid_labels=valid_labels,
                        test_dataset=test_dataset, test_labels=test_labels
                        )"""
"""apply_neural_network(train_dataset=train_dataset[:10000, :], train_labels=train_labels[:10000],
                        valid_dataset=valid_dataset, valid_labels=valid_labels,
                        test_dataset=test_dataset, test_labels=test_labels
                        )"""
apply_neural_network_with_l2_reg(train_dataset=train_dataset, train_labels=train_labels,
                                    valid_dataset=valid_dataset, valid_labels=valid_labels,
                                    test_dataset=test_dataset, test_labels=test_labels
                                )

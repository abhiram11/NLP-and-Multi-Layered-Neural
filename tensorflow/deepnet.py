import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
#one element is hot (ON), rest all are cold (OFF)

n_nodes_hl1 = 500
n_nodes_hl2 = 500 # number of nodes for hiddden layers
n_nodes_hl3 = 500 # the numbers can be random, not necessarily 500

n_classes = 10       #already defined in MNIST but still defined here
batch_size = 100

# matrix = height x width, here 28x28 pixel image converted to one line of 28x28 = 784 pixels(?) ki array

x = tf.placeholder('float', [None, 784]) #always good to describe its input type
y = tf.placeholder('float') # x is data y is label

def neural_network_model(data):

	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1])),
					'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))} #tensorflow variable which is tf random normal whose shape is defined

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1 ,n_nodes_hl2])),
					'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
					'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_1_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
					'biases':tf.Variable(tf.random_normal([n_classes]))}

#neurons work as : (input data ^ weights) + biases cuz if input data = 0 then neuron won't fire

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']) , hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1) #rectify linear/threshold function that takes the l1 as input value in its bracket

	l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']) , hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']) , hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3,output_1_layer['weights']) + output_1_layer['biases']
	return output


#specify how to run data thru that model
def train_neural_network(x):
	prediction = neural_network_model(x) #taking input data passing thr nnmodel, thru its layers, returns the output in prediciton;
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
	#will calculate the cost difference between the calculated prediction and the ACTUAL pre-present y value

#time to minimize the cost difference
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	#has parameter learning_rate=0.001 that can be defined too

	hm_epochs = 10 #how many epochs we want, less for lower performace CPU
#epochs  = cycles of feed forwads + backpropagations


#epoch = seen all of training data once
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		
		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)): # _ represents a variable that we dont care about
# implies how many times we wanna run the cycles depending on the dynamic entry of batch_size
				epoch_x,epoch_y= mnist.train.next_batch(batch_size) #input and labels as x and y
				_, c = sess.run([optimizer,cost], feed_dict={x:epoch_x,y:epoch_y}) # c represents cost
				epoch_loss+=c # for each c we resett epochloss but we need to keep count
			print(" Epoch ",epoch," completed out of ", hm_epochs, " loss : ", epoch_loss)


#data training completed

#now run them thru the model

			#try to make % completed
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1)) #first will return index of max value and matches with the other, i.e prediction = y

		accuracy = tf.reduce_mean(tf.cast(correct,'float')) #type cassting
		print('Accuracy: ',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)
































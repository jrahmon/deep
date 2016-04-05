import theano
import theano.tensor as T
import numpy as np


class AE_simple(object):

	def __init__(self, n_in, n_out, sentence_full, index_a, index_b, W1 = None, b1 = None, W2 = None, b2 = None, fan_in_activation = T.tanh, error_func = False ):

		"""
		outputs the reconstruction error for a single pair of inputs to the AE
	
		"""
		
		embedding_a = sentence_full[index_a]
		embedding_b = sentence_full[index_b]

		self.n_in = n_in #20
		self.n_out = n_out #10
		

		self.embedding_a = embedding_a #TensorVar
		self.embedding_b = embedding_b #TensorVar
	
		ipt_concat = T.concatenate([embedding_a,embedding_b])
	
		rng = np.random.RandomState(1234)
	
		if W1 is None:
			W_values = np.asarray(
				rng.uniform(
#					low = np.sqrt(-6. / (n_in + n_out)), #heuristic
#					high = np.sqrt(6. / (n_in + n_out)), #heuristic
					size = (n_in,n_out)
				),
				dtype = theano.config.floatX
			)
	
			W1 = theano.shared( value = W_values, name='W1', borrow=True ) 
	
		if b1 is None:
			b_values = np.zeros(n_out) # number of units in hidden layer
			b1 = theano.shared( value=b_values, name='b1', borrow=True )
		
		if W2 is None:
			W_values = np.asarray(
				rng.uniform(
#					low = np.sqrt(-6. / (n_in + n_out)), #heuristic
#					high = np.sqrt(6. / (n_in + n_out)), #heuristic
					size = (n_out,n_in)
				),
				dtype = theano.config.floatX
			)
	
			W2 = theano.shared( value = W_values, name='W2', borrow=True ) 
	
		if b2 is None:
			b_values = np.zeros(n_in) # number of units in hidden layer
			b2 = theano.shared( value=b_values, name='b2', borrow=True )
			
		fan_in_a = T.dot( ipt_concat, W1 ) + b1
		hidden = fan_in_activation(fan_in_a)
		
		self.reconstruction = T.dot( hidden, W2 ) + b2	 # op
		
		self.reconstruction_a = self.reconstruction[:n_out]
		self.reconstruction_b = self.reconstruction[n_out:]
		
		self.hidden = hidden
	
		self.params = [W1,b1,W2,b2]	
	
	
	def compute_reconstruction_error(self):
	
		ipt_A_recon = self.reconstruction[:self.n_out]
		ipt_B_recon = self.reconstruction[self.n_out:]
		
		ipt_A_error = T.mean((self.embedding_a - ipt_A_recon)**2)
		ipt_B_error = T.mean((self.embedding_b - ipt_B_recon)**2)
		
		error_total = ipt_A_error + ipt_B_error
		
		return error_total
	
	
	

	
def greedy( Sentence,W1,b1,W2,b2 ): #TensorVar

	sentence = T.matrix('sentence')
	index_a = T.scalar('index_a', dtype='int32')
	index_b = T.scalar('index_b', dtype='int32')
	
	ae = AE_simple(20,10,sentence,index_a,index_b,W1,b1,W2,b2)
	
	
	error_pair = theano.function([sentence,index_a,index_b],ae.compute_reconstruction_error())
	hidden_layer = theano.function([sentence,index_a,index_b],ae.hidden)
	recon = theano.function([sentence,index_a,index_b],ae.reconstruction)
	get_embedding = theano.function([sentence,index_a,index_b],[ae.embedding_a,ae.embedding_b])
		
	errors,hiddens,recons,embeddings = {},{},{},{}
	
	for i_h in range(len(Sentence)-1):
	
		indexed_a, indexed_b = i_h, i_h+1

		errors[i_h] = error_pair(Sentence,indexed_a,indexed_b)
	
		hiddens[i_h] =  hidden_layer(Sentence,indexed_a,indexed_b)
		
		recon_raw = recon(Sentence,indexed_a,indexed_b)
		first_recon = recon_raw[:10]
		second_recon = recon_raw[10:]
		
		recons[i_h] = [first_recon,second_recon]
		
		embeddings[i_h] = get_embedding(Sentence,indexed_a,indexed_b)
		
		
	for i_error in errors:
		
		if errors[i_error] == min(errors.values()):
		
			removal = [i_error, i_error+1]
			Sentence = [i for j,i in enumerate(Sentence) if j not in removal]
			Sentence.insert(i_error,list(hiddens[i_h]))
			print len(Sentence)

			return [Sentence,errors[i_error],recons[i_error],embeddings[i_error],removal]
		
		
		
			
my_sentence = []
for i in range(5):
	my_sentence.append([np.random.uniform() for j in range(10)])


	
def decisions ( Sentence,W1,b1,W2,b2 ):

	indexes_data = []
	new_sentences = []
	
	original_length = len(Sentence)

	for pair in range(original_length - 1):
		
		if len(Sentence) != original_length:
			padded = Sentence[:]
			for i in range(original_length-len(Sentence)):
				padded.append(list(np.zeros(len(Sentence[0]))))
			new_sentences.append(padded)
		else:
			new_sentences.append(Sentence)


		ae_sentence = greedy(Sentence,W1,b1,W2,b2)
		
		Sentence = ae_sentence[0]
		
		indexes_data.append(ae_sentence[4])
		
		continue

	
	return [indexes_data,new_sentences]
	
	

def sweep(index_pair,sentence_iter,W1,b1,W2,b2): #TensorVar
	
	ae = AE_simple(
		20,
		10,
		sentence_iter,
		index_pair[0],
		index_pair[1],
		W1=W1,
		b1=b1,
		W2=W2,
		b2=b2
		)
		
	cost_single = T.mean((ae.embedding_a - ae.reconstruction_a)**2) + T.mean((ae.embedding_b - ae.reconstruction_b)**2)
			
	return cost_single
	

def initialize_network(sentence):
	
	ae = AE_simple(20,10,sentence,0,1)
	parameters = ae.params
	if any(parameters) is None:
		print 'network error'
		return
	else:
		print 'network initilized'
		return parameters
	



###############
# TRAINING #
###############

def training(Sentence,learning_rate=0.1):

	indexes_symbolic = T.matrix('indexes_symbolic',dtype='int32')
	sentence_symbolic = T.tensor3('sentence_symbolic')

	network_params = initialize_network(my_sentence)
	W1,b1,W2,b2 = network_params[0],network_params[1],network_params[2],network_params[3]

	decision_tree = decisions(Sentence,W1,b1,W2,b2)
	indexes_instance = np.int32(np.array(decision_tree[0]))
	sentences_instance = np.float64(np.array(decision_tree[1]))

	cost_gather,updates_cost = theano.scan(sweep, 
						sequences = [indexes_symbolic,sentence_symbolic],
						non_sequences = [W1,b1,W2,b2]
						)
			
	cost = cost_gather.sum()



	#op_compute_gradient, reference "cost"
	gparams = [T.grad(cost.mean(),param) for param in network_params] # store these values in a theano variable called "gparams"
		
	#op_reference "gparams"
	updates = [
		(param,param - learning_rate * gparam)
		for param,gparam in zip(network_params, gparams)
		]
	
	#executable
	train_model = theano.function(
		inputs = [indexes_symbolic,sentence_symbolic],
		outputs = cost,
		updates = updates
		)	
	
	return [indexes_instance,sentences_instance,train_model]
	

	

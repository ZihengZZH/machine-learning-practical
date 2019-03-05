import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from dae_data import _get_training_data, _get_test_data


class BaseModel(object):
        
    def __init__(self, FLAGS):
        
        self.weight_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.2)
        self.bias_initializer=tf.zeros_initializer()
        self.FLAGS=FLAGS
    
    def _init_parameters(self):
        
        with tf.name_scope('weights'):
            self.W_1=tf.get_variable(name='weight_1', shape=(self.FLAGS.num_v,256), 
                                     initializer=self.weight_initializer)
            self.W_2=tf.get_variable(name='weight_2', shape=(256,128), 
                                     initializer=self.weight_initializer)
            self.W_3=tf.get_variable(name='weight_3', shape=(128,256), 
                                     initializer=self.weight_initializer)
            self.W_4=tf.get_variable(name='weight_4', shape=(256,self.FLAGS.num_v), 
                                     initializer=self.weight_initializer)
        
        with tf.name_scope('biases'):
            self.b1=tf.get_variable(name='bias_1', shape=(256), 
                                    initializer=self.bias_initializer)
            self.b2=tf.get_variable(name='bias_2', shape=(128), 
                                    initializer=self.bias_initializer)
            self.b3=tf.get_variable(name='bias_3', shape=(256), 
                                    initializer=self.bias_initializer)
    
    def inference(self, x):
        ''' 
        Making one forward pass. Predicting the networks outputs.
        @param x: input ratings
        @return : networks predictions
        '''
        
        with tf.name_scope('inference'):
             a1=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(x, self.W_1),self.b1))
             a2=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(a1, self.W_2),self.b2))
             a3=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(a2, self.W_3),self.b3))   
             a4=tf.matmul(a3, self.W_4) 
        return a4


class TrainModel(BaseModel):
    
    def __init__(self, FLAGS, name_scope):
        
        super(TrainModel,self).__init__(FLAGS)
        
        self._init_parameters()
        
    def _compute_loss(self, predictions, labels,num_labels):
        '''
        Computing the Mean Squared Error loss between the input and output of the network.
    	@param predictions: predictions of the stacked autoencoder
    	@param labels: input values of the stacked autoencoder which serve as labels at the same time
    	@param num_labels: number of labels !=0 in the data set to compute the mean	
    	@return mean squared error loss tf-operation
    	'''
        with tf.name_scope('loss'):
            
            loss_op=tf.div(tf.reduce_sum(tf.square(tf.subtract(predictions,labels))),num_labels)
            return loss_op
    
    def _validation_loss(self, x_train, x_test):
        ''' 
        Computing the loss during the validation time.	
    	@param x_train: training data samples
    	@param x_test: test data samples	
    	@return networks predictions
        @return root mean squared error loss between the predicted and actual ratings
        '''
        outputs=self.inference(x_train) # use training sample to make prediction
        mask=tf.where(tf.equal(x_test,0.0), tf.zeros_like(x_test), x_test) # identify the zero values in the test ste
        num_test_labels=tf.cast(tf.count_nonzero(mask),dtype=tf.float32) # count the number of non zero values
        bool_mask=tf.cast(mask,dtype=tf.bool) 
        outputs=tf.where(bool_mask, outputs, tf.zeros_like(outputs))
    
        MSE_loss=self._compute_loss(outputs, x_test, num_test_labels)
        RMSE_loss=tf.sqrt(MSE_loss)
        
        ab_ops=tf.div(tf.reduce_sum(tf.abs(tf.subtract(x_test,outputs))),num_test_labels)
        
        return outputs, x_test, RMSE_loss, ab_ops
    
    def train(self, x):
        '''
        Optimization of the network parameter through stochastic gradient descent.
        @param x: input values for the stacked autoencoder.
        @return: tensorflow training operation
        @return: ROOT!! mean squared error
        '''
        outputs=self.inference(x)
        mask=tf.where(tf.equal(x,0.0), tf.zeros_like(x), x) # indices of 0 values in the training set
        num_train_labels=tf.cast(tf.count_nonzero(mask),dtype=tf.float32) # number of non zero values in the training set
        bool_mask=tf.cast(mask,dtype=tf.bool) # boolean mask
        outputs=tf.where(bool_mask, outputs, tf.zeros_like(outputs)) # set the output values to zero if corresponding input values are zero

        MSE_loss=self._compute_loss(outputs,x,num_train_labels)
        
        if self.FLAGS.l2_reg==True:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            MSE_loss = MSE_loss +  self.FLAGS.lambda_ * l2_loss
        
        train_op=tf.train.AdamOptimizer(self.FLAGS.learning_rate).minimize(MSE_loss)
        RMSE_loss=tf.sqrt(MSE_loss)

        return train_op, RMSE_loss


class InferenceModel(BaseModel):
    
    def __init__(self, FLAGS):
        
        super(InferenceModel,self).__init__(FLAGS)
        self._init_parameters()


def _get_bias_initializer():
    return tf.zeros_initializer()


def _get_weight_initializer():
    return tf.random_normal_initializer(mean=0.0, stddev=0.05)


tf.app.flags.DEFINE_string('tf_records_train_path', 
                           os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'records/train/')),
                           'Path of the training data.')

tf.app.flags.DEFINE_string('tf_records_test_path', 
                           os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'records/test/')),
                           'Path of the test data.')

tf.app.flags.DEFINE_string('checkpoints_path', os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'checkpoints/model.ckpt')), 
                           'Path for the test data.')

tf.app.flags.DEFINE_integer('num_epoch', 1000,
                            'Number of training epochs.')

tf.app.flags.DEFINE_integer('batch_size', 16,
                            'Size of the training batch.')

tf.app.flags.DEFINE_float('learning_rate',0.0005,
                          'Learning_Rate')

tf.app.flags.DEFINE_boolean('l2_reg', False,
                            'L2 regularization.')

tf.app.flags.DEFINE_float('lambda_',0.01,
                          'Wight decay factor.')

tf.app.flags.DEFINE_integer('num_v', 3952,
                            'Number of visible neurons (Number of movies the users rated.)')

tf.app.flags.DEFINE_integer('num_h', 128,
                            'Number of hidden neurons.)')

tf.app.flags.DEFINE_integer('num_samples', 5953,
                            'Number of training samples (Number of users, who gave a rating).')

FLAGS = tf.app.flags.FLAGS


def main():
    '''
    Building the graph, opening of a session and starting the training od the neural network.
    '''
    num_batches=int(FLAGS.num_samples/FLAGS.batch_size)

    with tf.Graph().as_default():
        train_data, train_data_infer=_get_training_data(FLAGS)
        test_data=_get_test_data(FLAGS)
        
        iter_train = train_data.make_initializable_iterator()
        iter_train_infer=train_data_infer.make_initializable_iterator()
        iter_test=test_data.make_initializable_iterator()
        
        x_train= iter_train.get_next()
        x_train_infer=iter_train_infer.get_next()
        x_test=iter_test.get_next()

        model=TrainModel(FLAGS, 'training')

        train_op, train_loss_op=model.train(x_train)
        prediction, labels, test_loss_op, mae_ops=model._validation_loss(x_train_infer, x_test)
        
        saver=tf.train.Saver()
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_loss=0
            test_loss=[]
            mae=[]

            for epoch in range(FLAGS.num_epoch):
                sess.run(iter_train.initializer)
                sess.run(iter_train_infer.initializer)
                sess.run(iter_test.initializer)

                for batch_nr in range(num_batches):
                    _, loss_=sess.run((train_op, train_loss_op))
                    train_loss+=loss_
                
                for i in range(FLAGS.num_samples):
                    pred, labels_, loss_, mae_=sess.run((prediction, labels, test_loss_op,mae_ops))

                    test_loss.append(loss_)
                    mae.append(mae_)
                    
                print('epoch_nr: %i, train_loss: %.3f, test_loss: %.3f, mean_abs_error: %.3f' % (epoch,(train_loss/num_batches),np.mean(test_loss), np.mean(mae)))
                
                if np.mean(mae)<0.9:
                    saver.save(sess, FLAGS.checkpoints_path)

                train_loss=0
                test_loss=[]
                mae=[]


if __name__ == "__main__":
    
    tf.app.run()
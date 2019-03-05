import os
import sys
import numpy as np
import pandas as pd
import gc
from pathlib import Path
import tensorflow as tf


def _get_training_data(FLAGS):
    ''' 
    Buildind the input pipeline for training and inference using TFRecords files.
    @return data only for the training
    @return data for the inference
    '''
    filenames = [FLAGS.tf_records_train_path + '/' + f for f in os.listdir(FLAGS.tf_records_train_path)]

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse)
    dataset = dataset.shuffle(buffer_size=1)
    dataset = dataset.repeat()
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.prefetch(buffer_size=1)

    dataset2 = tf.data.TFRecordDataset(filenames)
    dataset2 = dataset2.map(parse)
    dataset2 = dataset2.shuffle(buffer_size=1)
    dataset2 = dataset2.repeat()
    dataset2 = dataset2.batch(FLAGS.batch_size)
    dataset2 = dataset2.prefetch(buffer_size=1)

    return dataset, dataset2


def _get_test_data(FLAGS):
    '''
    Buildind the input pipeline for test data.
    '''
    filenames = [FLAGS.tf_records_test_path + '/' + f for f in os.listdir(FLAGS.tf_records_test_path)]

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse)
    dataset = dataset.shuffle(buffer_size=1)
    dataset = dataset.repeat()
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(buffer_size=1)

    return dataset


def parse(serialized):
    '''
    Parser for the TFRecords file.
    '''
    features = {'movie_ratings':tf.FixedLenFeature([3952], tf.float32)}
    parsed_example = tf.parse_single_example(serialized, features=features)
    movie_ratings = tf.cast(parsed_example['movie_ratings'], tf.float32)

    return movie_ratings


p = Path(__file__).parents[1]
ROOT_DIR = os.path.abspath(os.path.join(p, '..', '/'))


def convert(data, num_users, num_movies):
    '''
    Making a User-Movie-Matrix
    '''
    new_data = []
    
    for id_user in range(1, num_users+1):

        id_movie = data[:,1][data[:,0] == id_user]
        id_rating = data[:,2][data[:,0] == id_user]
        ratings = np.zeros(num_movies, dtype=np.uint32)
        ratings[id_movie - 1] = id_rating
        if sum(ratings) == 0:
            continue

        new_data.append(ratings)

        del id_movie
        del id_rating
        del ratings

    return new_data
    

def get_dataset_1M():
    ''' 
    For each train.dat and test.dat making a User-Movie-Matrix.
    '''
    
    gc.enable()
    
    training_set=pd.read_csv(ROOT_DIR+'/ml-1m/train.dat', sep='::', header=None, engine='python', encoding='latin-1')
    training_set=np.array(training_set, dtype=np.uint32)
    
    test_set=pd.read_csv(ROOT_DIR+'/ml-1m/test.dat', sep='::', header=None, engine='python', encoding='latin-1')
    test_set=np.array(test_set, dtype=np.uint32)
    
    num_users=int(max(max(training_set[:,0]), max(test_set[:,0])))
    num_movies=int(max(max(training_set[:,1]), max(test_set[:,1])))

    training_set=convert(training_set,num_users, num_movies)
    test_set=convert(test_set,num_users, num_movies)
    
    return training_set, test_set
    

def _get_dataset():

    return get_dataset_1M()


ROOT_DIR_TF = os.path.abspath(os.path.join(p, '..', 'records/'))
TF_RECORD_TRAIN_PATH='/tf_records/train'
TF_RECORD_TEST_PATH='/tf_records/test'


def _add_to_tfrecord(data_sample,tfrecord_writer):
    
    data_sample=list(data_sample.astype(dtype=np.float32))
    
    example = tf.train.Example(features=tf.train.Features(feature={'movie_ratings': float_feature(data_sample)}))                                          
    tfrecord_writer.write(example.SerializeToString())
    

def _get_output_filename(output_dir, idx, name):
    return '%s/%s_%03d.tfrecord' % (ROOT_DIR_TF+output_dir, name, idx)


def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def main():
    ''' Writes the .txt training and testing data into binary TF_Records.'''

    SAMPLES_PER_FILES=100
    training_set, test_set=get_dataset_1M()

    for data_set, name, dir_ in zip([training_set, test_set], ['train', 'test'], [TF_RECORD_TRAIN_PATH, TF_RECORD_TEST_PATH]):
        num_samples=len(data_set)
        i = 0
        fidx = 1
        while i < num_samples:
            tf_filename = _get_output_filename(dir_, fidx,  name=name)
            with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
                j = 0
                while i < num_samples and j < SAMPLES_PER_FILES:
                    sys.stdout.write('\r>> Converting sample %d/%d' % (i+1, num_samples))
                    sys.stdout.flush()
                    sample = data_set[i]
                    _add_to_tfrecord(sample, tfrecord_writer)
                    i += 1
                    j += 1
                fidx += 1

    print('\nFinished converting the dataset!')


OUTPUT_DIR_TRAIN=os.path.abspath(os.path.join(p, '..', 'ml-1m/train.dat'))
OUTPUT_DIR_TEST=os.path.abspath(os.path.join(p, '..', 'ml-1m/test.dat'))
ROOT_DIR_RATING=os.path.abspath(os.path.join(p, '..', 'ml-1m/ratings.dat'))
NUM_USERS=6040
NUM_TEST_RATINGS=10


def count_rating_per_user():
    rating_per_user={}
    for line in open(ROOT_DIR_RATING):
        line=line.split('::')
        user_nr=int(line[0])
        if user_nr in rating_per_user:
            rating_per_user[user_nr]+=1                       
        else:
            rating_per_user[user_nr]=1

    return rating_per_user


def train_test_split():
    user_rating=count_rating_per_user()
    test_counter=0
    next_user=1
    
    train_writer=open(OUTPUT_DIR_TRAIN, 'w')
    test_writer=open(OUTPUT_DIR_TEST, 'w')
    
    for line in open(ROOT_DIR_RATING):
        splitted_line=line.split('::')
        user_nr=int(splitted_line[0])
        
        if user_rating[user_nr]<=NUM_TEST_RATINGS*2:
            next_user+=1
            continue
        
        try:
            if user_nr==next_user:
                write_test_samples=True
                next_user+=1
            if write_test_samples==True:
                test_writer.write(line)
                test_counter+=1

                if test_counter>=NUM_TEST_RATINGS:
                    test_counter=0
                    write_test_samples=False        
            else:
                train_writer.write(line)
        
        except KeyError:   
            print('Key not found')
            continue


if __name__ == "__main__":
    # train_test_split()
    main()
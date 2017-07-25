
# coding: utf-8

# # Kaggle Competition: New York City Taxi Trip Duration

# The purpose of this analysis is to accurately predict the duration of taxi trips in New York City. This work is for a [Kaggle competition](https://www.kaggle.com/c/nyc-taxi-trip-duration). To make our predictions, we will use a feed-forward neural network using TensorFlow, a RandomForest Regressor, Lightgbm, and Catboost. Random search will be used to find the optimal network architecture and hyperparameter values for each model.
# 
# The sections of this analysis are:
# - Loading the Data
# - Cleaning the Data
# - Building the Neural Network
# - Training the Neural Network
# - Training the Other Models
# - Making Predictions
# - Summary

# In[131]:

import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import tensorflow as tf

from sklearn.ensemble import RandomForestRegressor as RFR
import lightgbm as lgb
from catboost import CatBoostRegressor

from collections import namedtuple
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

import time
import operator
import haversine
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import timedelta
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

print(tf.__version__)


# ## Loading the Data

# In[132]:

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[133]:

train.head()


# In[134]:

test.head()


# In[135]:

print(train.shape)
print(test.shape)


# In[136]:

# Check for any duplicates
print(train.duplicated().sum())
print(train.id.duplicated().sum())
print(test.id.duplicated().sum())


# In[137]:

# Sanity check to ensure all trips are valid
sum(train.dropoff_datetime < train.pickup_datetime)


# ## Cleaning the Data

# In[138]:

# drop feature since it will not be used to make any predictions.
# it is not included in the test dataframe
train = train.drop('dropoff_datetime',1)


# In[139]:

# Some of the journeys are very long
train.trip_duration.describe()


# In[140]:

# Values are in minutes
print(np.percentile(train.trip_duration, 99)/60)
print(np.percentile(train.trip_duration, 99.5)/60)
print(np.percentile(train.trip_duration, 99.6)/60)
print(np.percentile(train.trip_duration, 99.8)/60)
print(np.percentile(train.trip_duration, 99.85)/60)
print(np.percentile(train.trip_duration, 99.9)/60)
print(np.percentile(train.trip_duration, 99.99)/60)
print(np.percentile(train.trip_duration, 99.999)/60)
print(np.percentile(train.trip_duration, 99.9999)/60)
print(train.trip_duration.max() / 60)


# In[141]:

# Check how many trips remain with each limit
print(len(train[train.trip_duration <= np.percentile(train.trip_duration, 99.9)]))
print(len(train[train.trip_duration <= np.percentile(train.trip_duration, 99.99)]))
print(len(train[train.trip_duration <= np.percentile(train.trip_duration, 99.999)]))


# In[142]:

# Remove outliers
train = train[train.trip_duration <= np.percentile(train.trip_duration, 99.999)]


# In[143]:

# Plot locations - look for outliers
n = 100000 # number of data points to display

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
ax1.scatter(train.pickup_longitude[:n], 
            train.pickup_latitude[:n],
            alpha = 0.1)
ax1.set_title('Pickup')
ax2.scatter(train.dropoff_longitude[:n], 
            train.dropoff_latitude[:n],
            alpha = 0.1)
ax2.set_title('Dropoff')


# In[144]:

# The values are not too wild, but we'll trim them back a little to be conservative
print(train.pickup_latitude.max())
print(train.pickup_latitude.min())
print(train.pickup_longitude.max())
print(train.pickup_longitude.min())
print()
print(train.dropoff_latitude.max())
print(train.dropoff_latitude.min())
print(train.dropoff_longitude.max())
print(train.dropoff_longitude.min())


# In[145]:

# Find limits of location
max_value = 99.999
min_value = 0.001

max_pickup_lat = np.percentile(train.pickup_latitude, max_value)
min_pickup_lat = np.percentile(train.pickup_latitude, min_value)
max_pickup_long = np.percentile(train.pickup_longitude, max_value)
min_pickup_long = np.percentile(train.pickup_longitude, min_value)

max_dropoff_lat = np.percentile(train.dropoff_latitude, max_value)
min_dropoff_lat = np.percentile(train.dropoff_latitude, min_value)
max_dropoff_long = np.percentile(train.dropoff_longitude, max_value)
min_dropoff_long = np.percentile(train.dropoff_longitude, min_value)


# In[146]:

# Remove extreme values
train = train[(train.pickup_latitude <= max_pickup_lat) & (train.pickup_latitude >= min_pickup_lat)]
train = train[(train.pickup_longitude <= max_pickup_long) & (train.pickup_longitude >= min_pickup_long)]

train = train[(train.dropoff_latitude <= max_dropoff_lat) & (train.dropoff_latitude >= min_dropoff_lat)]
train = train[(train.dropoff_longitude <= max_dropoff_long) & (train.dropoff_longitude >= min_dropoff_long)]


# In[147]:

# Replot to see the differences - minimal, but there is some change
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
ax1.scatter(train.pickup_longitude[:n], 
            train.pickup_latitude[:n],
            alpha = 0.1)
ax1.set_title('Pickup')
ax2.scatter(train.dropoff_longitude[:n], 
            train.dropoff_latitude[:n],
            alpha = 0.1)
ax2.set_title('Dropoff')


# In[148]:

# Concatenate the datasets for feature engineering
df = pd.concat([train,test])


# In[149]:

df.shape


# In[150]:

# Check for null values
# trip_duration nulls to due to them not being present in the test set
df.isnull().sum()


# In[151]:

df.vendor_id.value_counts()


# In[153]:

print(train.pickup_datetime.max())
print(train.pickup_datetime.min())
print()
print(test.pickup_datetime.max())
print(test.pickup_datetime.min())
print()
print(df.pickup_datetime.max())
print(df.pickup_datetime.min())


# In[154]:

# Convert to datetime
df.pickup_datetime = pd.to_datetime(df.pickup_datetime)


# In[155]:

# Calculate what minute in a day the pickup is at
df['pickup_minute_of_the_day'] = df.pickup_datetime.dt.hour*60 + df.pickup_datetime.dt.minute


# In[156]:

# Rather than use the standard 24 hours, group the trips into 24 groups that are sorted by KMeans
# This should help 'rush-hour' rides to be in the same groups
kmeans_pickup_time = KMeans(n_clusters=24, random_state=2).fit(df.pickup_minute_of_the_day[:500000].values.reshape(-1,1))


# In[157]:

df['kmeans_pickup_time'] = kmeans_pickup_time.predict(df.pickup_minute_of_the_day.values.reshape(-1,1))


# In[158]:

# Compare the distribution of kmeans_pickup_time and the standard 24 hour breakdown
n = 50000 # number of data points to plot
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 5))

ax1.scatter(x = df.pickup_minute_of_the_day[:n]/60, 
            y = np.random.uniform(0,1, n), 
            cmap = 'Set1',
            c = df.kmeans_pickup_time[:n])
ax1.set_title('KMeans Pickup Time')

ax2.scatter(x = df.pickup_minute_of_the_day[:n]/60, 
            y = np.random.uniform(0,1, n), 
            cmap = 'Set1',
            c = df.pickup_datetime.dt.hour[:n])
ax2.set_title('Pickup Hour')


# In[159]:

# Load a list of holidays in the US
calendar = USFederalHolidayCalendar()
holidays = calendar.holidays()

# Load business days
us_bd = CustomBusinessDay(calendar = USFederalHolidayCalendar())
# Set business_days equal to the work days in our date range.
business_days = pd.DatetimeIndex(start = df.pickup_datetime.min(), 
                                 end = df.pickup_datetime.max(), 
                                 freq = us_bd)
business_days = pd.to_datetime(business_days).date


# In[160]:

# Create features relating to time
df['pickup_month'] = df.pickup_datetime.dt.month
df['pickup_weekday'] = df.pickup_datetime.dt.weekday
df['pickup_is_weekend'] = df.pickup_weekday.map(lambda x: 1 if x >= 5 else 0)
df['pickup_holiday'] = pd.to_datetime(df.pickup_datetime.dt.date).isin(holidays)
df['pickup_holiday'] = df.pickup_holiday.map(lambda x: 1 if x == True else 0)

# If day is before or after a holiday
df['pickup_near_holiday'] = (pd.to_datetime(df.pickup_datetime.dt.date).isin(holidays + timedelta(days=1)) |
                             pd.to_datetime(df.pickup_datetime.dt.date).isin(holidays - timedelta(days=1)))
df['pickup_near_holiday'] = df.pickup_near_holiday.map(lambda x: 1 if x == True else 0)
df['pickup_businessday'] = pd.to_datetime(df.pickup_datetime.dt.date).isin(business_days)
df['pickup_businessday'] = df.pickup_businessday.map(lambda x: 1 if x == True else 0)

# Calculates what minute of the week it is
df['week_delta'] = (df.pickup_weekday + ((df.pickup_datetime.dt.hour + 
                                              (df.pickup_datetime.dt.minute / 60.0)) / 24.0))


# In[161]:

# Determines number of rides that occur during each specific time
# Should help to determine traffic
ride_counts = df.groupby(['pickup_month', 'pickup_weekday','pickup_holiday','pickup_near_holiday',
            'pickup_businessday','kmeans_pickup_time']).size()
ride_counts = pd.DataFrame(ride_counts).reset_index()
ride_counts['ride_counts'] = ride_counts[0]
ride_counts = ride_counts.drop(0,1)

# Add `ride_counts` to dataframe
df = df.merge(ride_counts, on=['pickup_month',
                          'pickup_weekday',
                          'pickup_holiday',
                          'pickup_near_holiday',
                          'pickup_businessday',
                          'kmeans_pickup_time'], how='left')


# In[163]:

# Dont' need this feature any more
df = df.drop('pickup_datetime', 1)


# In[164]:

# Group pickup and dropoff locations into 15 groups
kmeans_pickup = KMeans(n_clusters=15, random_state=2).fit(df[['pickup_latitude','pickup_longitude']][:500000])
kmeans_dropoff = KMeans(n_clusters=15, random_state=2).fit(df[['dropoff_latitude','dropoff_longitude']][:500000])

df['kmeans_pickup'] = kmeans_pickup.predict(df[['pickup_latitude','pickup_longitude']])
df['kmeans_dropoff'] = kmeans_dropoff.predict(df[['dropoff_latitude','dropoff_longitude']])


# In[166]:

# Plot these 15 groups

n = 100000 # Number of data points to plot
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
ax1.scatter(df.pickup_longitude[:n], 
            df.pickup_latitude[:n],
            cmap = 'viridis',
            c = df.kmeans_pickup[:n])
ax1.set_title('Pickup')
ax2.scatter(df.dropoff_longitude[:n], 
            df.dropoff_latitude[:n],
            cmap = 'viridis',
            c = df.kmeans_dropoff[:n])
ax2.set_title('Dropoff')


# In[168]:

# Reduce pickup and dropoff locations to one value
pca = PCA(n_components=1)
df['pickup_pca'] = pca.fit_transform(df[['pickup_latitude','pickup_longitude']])
df['dropoff_pca'] = pca.fit_transform(df[['dropoff_latitude','dropoff_longitude']])


# In[167]:

# Create distance features
df['distance'] = np.sqrt(np.power(df['dropoff_longitude'] - df['pickup_longitude'], 2) + 
                         np.power(df['dropoff_latitude'] - df['pickup_latitude'], 2))
df['haversine_distance'] = df.apply(lambda r: haversine.haversine((r['pickup_latitude'],r['pickup_longitude']),
                                                                  (r['dropoff_latitude'], r['dropoff_longitude'])), 
                           axis=1)
df['manhattan_distance'] = (abs(df.dropoff_longitude - df.pickup_longitude) +
                            abs(df.dropoff_latitude - df.pickup_latitude))
df['log_distance'] = np.log(df['distance'] + 1)
df['log_haversine_distance'] = np.log(df['haversine_distance'] + 1)
df['log_manhattan_distance'] = np.log(df.manhattan_distance + 1)


# In[169]:

def calculate_bearing(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    '''Calculate the direction of travel in degrees'''
    pickup_lat_rads = np.radians(pickup_lat)
    pickup_long_rads = np.radians(pickup_long)
    dropoff_lat_rads = np.radians(dropoff_lat)
    dropoff_long_rads = np.radians(dropoff_long)
    long_delta_rads = np.radians(dropoff_long_rads - pickup_long_rads)
    
    y = np.sin(long_delta_rads) * np.cos(dropoff_lat_rads)
    x = (np.cos(pickup_lat_rads) * 
         np.sin(dropoff_lat_rads) - 
         np.sin(pickup_lat_rads) * 
         np.cos(dropoff_lat_rads) * 
         np.cos(long_delta_rads))
    
    return np.degrees(np.arctan2(y, x))


# In[170]:

df['bearing'] = calculate_bearing(df.pickup_latitude,
                                  df.pickup_longitude,
                                  df.dropoff_latitude,
                                  df.dropoff_longitude)


# In[171]:

df.passenger_count.value_counts()


# In[172]:

# Group passenger_count by type of group
df['no_passengers'] = df.passenger_count.map(lambda x: 1 if x == 0 else 0)
df['one_passenger'] = df.passenger_count.map(lambda x: 1 if x == 1 else 0)
df['few_passengers'] = df.passenger_count.map(lambda x: 1 if x > 1 and x <= 4 else 0)
df['many_passengers'] = df.passenger_count.map(lambda x: 1 if x >= 5 else 0)


# In[173]:

df.store_and_fwd_flag = df.store_and_fwd_flag.map(lambda x: 1 if x == 'Y' else 0)


# In[174]:

# Create dummy features for these features, then drop these features
dummies = ['kmeans_pickup_time','pickup_month','pickup_weekday','kmeans_pickup','kmeans_dropoff']
for feature in dummies:
    dummy_features = pd.get_dummies(df[feature], prefix=feature)
    for dummy in dummy_features:
        df[dummy] = dummy_features[dummy]
    df = df.drop([feature], 1)


# In[175]:

# Check that all features look okay
df.head()


# In[176]:

# Don't need this feature any more
df = df.drop(['id'],1)


# In[177]:

# Transform each feature to have a mean of 0 and standard deviation of 1
# Help to train the neural network
for feature in df:
    if feature == 'trip_duration':
        continue
    mean, std = df[feature].mean(), df[feature].std()
    df.loc[:, feature] = (df[feature] - mean)/std


# In[178]:

# Check that the transformation was carried out correctly
df.head()


# In[179]:

# Return data into a training and testing set
trainFinal = df[:-len(test)]
testFinal = df[-len(test):]


# In[180]:

# Check lengths of dataframes
print(len(trainFinal))
print(len(testFinal))
print(len(test))


# In[181]:

# Give trip_duration its own dataframe
# Drop it from the other dataframes
yFinal = pd.DataFrame(trainFinal.trip_duration)
trainFinal = trainFinal.drop('trip_duration',1)
testFinal = testFinal.drop('trip_duration',1)


# In[182]:

# Sort data into training and testing sets
x_trainFinal, x_testFinal, y_trainFinal, y_testFinal = train_test_split(trainFinal, 
                                                                        np.log(yFinal+1), 
                                                                        test_size=0.15, 
                                                                        random_state=2)

x_train, x_test, y_train, y_test = train_test_split(x_trainFinal, 
                                                    y_trainFinal, 
                                                    test_size=0.15,
                                                    random_state=2)


# ## Build the Neural Network

# In[183]:

def create_weights_biases(num_layers, n_inputs, multiplier, max_nodes):
    '''Use the inputs to create the weights and biases for a network'''
    
    # Empty dictionaries to store the weights and biases for each layer
    weights = {}
    biases = {}
    
    # Create weights and biases for all layers, but the final layer
    for layer in range(1,num_layers):
        # The first layer needs to use the number of features that are in the dataframe
        if layer == 1:
            weights["h"+str(layer)] = tf.Variable(tf.random_normal([num_features, n_inputs],
                                                                   stddev=np.sqrt(1/num_features)))
            biases["b"+str(layer)] = tf.Variable(tf.random_normal([n_inputs],stddev=0))
            # n_previous keeps track of the number of nodes in the previous layer
            n_previous = n_inputs
            
        else:    
            # To alter number of nodes in each layer, multiply n_previous by multiplier 
            n_current = int(n_previous * multiplier)
            
            # Limit the number of nodes to the maximum amount
            if n_current >= max_nodes:
                n_current = max_nodes
                
            weights["h"+str(layer)] = tf.Variable(tf.random_normal([n_previous, n_current],
                                                                       stddev=np.sqrt(1/n_previous)))
            biases["b"+str(layer)] = tf.Variable(tf.random_normal([n_current],stddev=0))
            n_previous = n_current
            
    # Create weights for the final layer
    n_current = int(n_previous * multiplier)
    if n_current >= max_nodes:
        n_current = max_nodes
            
    # The final layer only has 1 node since this is a regression task
    weights["out"] = tf.Variable(tf.random_normal([n_previous, 1], stddev=np.sqrt(1/n_previous)))
    biases["out"] = tf.Variable(tf.random_normal([1],stddev=0))
                                                    
    return weights, biases


# In[184]:

def network(num_layers, n_inputs, weights, biases, rate, is_training, activation_function):
    '''Add the required number of layers to the network'''
    
    for layer in range(1, num_layers):
        if layer == 1:
            current_layer = eval(activation_function + "(tf.matmul(n_inputs, weights['h1']) + biases['b1'])")
            current_layer = tf.nn.dropout(current_layer, 1-rate)
            previous_layer = current_layer
        else:
            current_layer = eval(activation_function + "(tf.matmul(previous_layer,            weights['h'+str(layer)]) + biases['b'+str(layer)])")
            current_layer = tf.nn.dropout(current_layer, 1-rate)
            previous_layer = current_layer

    # Output layer with linear activation - because regression
    out_layer = tf.matmul(previous_layer, weights['out']) + biases['out']
    return out_layer


# In[185]:

def model_inputs():
    '''Create placeholders for model's inputs '''
    
    inputs = tf.placeholder(tf.float32, [None, None], name='inputs')
    targets = tf.placeholder(tf.float32, [None, 1], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')
    is_training = tf.placeholder(tf.bool, name='is_training')
    
    return inputs, targets, learning_rate, dropout_rate, is_training


# In[186]:

def build_graph(num_layers,n_inputs,weights_multiplier,dropout_rate,learning_rate,max_nodes,activation_function):
    '''Use inputs to build the graph and export the required features for training'''
    
    # Reset the graph to ensure it is ready for training
    tf.reset_default_graph()
    
    # Get the inputs
    inputs, targets, learning_rate, dropout_rate, is_training = model_inputs()
    
    # Get the weights and biases
    weights, biases = create_weights_biases(num_layers, n_inputs, weights_multiplier, max_nodes)
    
    # Construct the network
    preds = network(num_layers, inputs, weights, biases, dropout_rate, is_training, activation_function)    
            
    with tf.name_scope("cost"):
        # Cost function
        cost = tf.sqrt(tf.losses.mean_squared_error(labels=targets, predictions=preds))
        tf.summary.scalar('cost', cost)

    with tf.name_scope("optimze"):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Merge all of the summaries
    merged = tf.summary.merge_all()    

    # Export the nodes 
    export_nodes = ['inputs','targets','dropout_rate','is_training','cost','preds','merged',
                    'optimizer','learning_rate']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])

    return graph


# ## Training the Neural Network

# In[187]:

def train(model, epochs, log_string, learning_rate):
    '''Train the Network and return the average RMSE for each iteration of the model'''
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Used to determine when to stop the training early
        testing_loss_summary = []

        iteration = 0 # Keep track of which batch iteration is being trained
        stop_early = 0 # Keep track of how many consective epochs have not achieved a record low RMSE
        stop = 5 # If the batch_loss_testing does not decrease in 5 consecutive epochs, stop training
        per_epoch_training = 2 # Check training progress 2 times per epcoh
        per_epoch_testing = 1 # Check testing progress 1 time per epoch
        
        # Decay learning rate after consective epochs of no improvements
        learning_rate_decay_threshold = np.random.choice([2,3]) 
        original_learning_rate = learning_rate # Keep track of orginial learning rate for each split

        print()
        print("Training Model: {}".format(log_string))

        # Record progress to view with TensorBoard
        train_writer = tf.summary.FileWriter('./logs/1/train/{}'.format(log_string), sess.graph)
        test_writer = tf.summary.FileWriter('./logs/1/test/{}'.format(log_string))
        
        training_check = (len(x_train)//batch_size//per_epoch_training)-1 # Check training progress after this many batches
        testing_check = (len(x_train)//batch_size//per_epoch_testing)-1 # Check testing results after this many batches

        for epoch_i in range(1, epochs+1): 
            batch_loss = 0
            batch_time = 0

            for batch in range(int(len(x_train)/batch_size)):
                batch_x = x_train[batch*batch_size:(1+batch)*batch_size]
                batch_y = y_train[batch*batch_size:(1+batch)*batch_size]

                start_time = time.time()

                summary, loss, _ = sess.run([model.merged,
                                             model.cost, 
                                             model.optimizer], 
                                             {model.inputs: batch_x,
                                              model.targets: batch_y,
                                              model.learning_rate: learning_rate,
                                              model.dropout_rate: dropout_rate,
                                              model.is_training: True})


                batch_loss += loss
                end_time = time.time()
                batch_time += end_time - start_time

                # Record the progress of training
                train_writer.add_summary(summary, iteration)

                iteration += 1

                if batch % training_check == 0 and batch > 0:
                    print('Epoch {:>3}/{} Batch {:>4}/{} - RMSE: {:>6.3f}, Seconds: {:>4.2f}'
                          .format(epoch_i,
                                  epochs, 
                                  batch, 
                                  len(x_train) // batch_size, 
                                  (batch_loss / training_check), 
                                  batch_time))
                    batch_loss = 0
                    batch_time = 0

                #### Testing ####
                if batch % testing_check == 0 and batch > 0:
                    batch_loss_testing = 0
                    batch_time_testing = 0
                    for batch in range(int(len(x_test)/batch_size)):
                        batch_x = x_test[batch*batch_size:(1+batch)*batch_size]
                        batch_y = y_test[batch*batch_size:(1+batch)*batch_size]

                        start_time_testing = time.time()
                        summary, loss = sess.run([model.merged,
                                                  model.cost], 
                                                     {model.inputs: batch_x,
                                                      model.targets: batch_y,
                                                      model.learning_rate: learning_rate,
                                                      model.dropout_rate: 0,
                                                      model.is_training: False})

                        batch_loss_testing += loss
                        end_time_testing = time.time()
                        batch_time_testing += end_time_testing - start_time_testing

                        # Record the progress of testing
                        test_writer.add_summary(summary, iteration)

                    n_batches_testing = batch + 1
                    print('Testing RMSE: {:>6.3f}, Seconds: {:>4.2f}'
                          .format(batch_loss_testing / n_batches_testing, 
                                  batch_time_testing))

                    batch_time_testing = 0

                    # If the batch_loss_testing is at a new minimum, save the model
                    testing_loss_summary.append(batch_loss_testing)
                    if batch_loss_testing <= min(testing_loss_summary):
                        print('New Record!') 
                        lowest_loss_testing = batch_loss_testing/n_batches_testing
                        stop_early = 0 # Reset stop_early if new minimum loss is found
                        checkpoint = "./{}.ckpt".format(log_string)
                        saver = tf.train.Saver()
                        saver.save(sess, checkpoint)

                    else:
                        print("No Improvement.")
                        stop_early += 1 # Increase stop_early if no new minimum loss is found
                        if stop_early % learning_rate_decay_threshold == 0:
                            learning_rate *= learning_rate_decay
                            print("New learning rate = ", learning_rate)
                        elif stop_early == stop:
                            break

            if stop_early == stop:
                print("Stopping training for this iteration.")
                print("Lowest RMSE =", lowest_loss_testing)
                print()
                early_stop = 0
                testing_loss_summary = []
                break
        
    return lowest_loss_testing


# In[188]:

# Use random search to choose the values for each iteration

num_iterations = 15
results = {} # Save the log_string and RMSE of each iteration
for i in range(num_iterations):
    # (Randomly) choose the value for each input
    num_features = x_train.shape[1]
    epochs = 50
    learning_rate = np.random.uniform(0.001, 0.1)
    learning_rate_decay = np.random.uniform(0.1,0.5)
    weights_multiplier = np.random.uniform(0.5,2)
    n_inputs = np.random.randint(int(num_features)*0.1,int(num_features)*2)
    num_layers = np.random.choice([2,3,4])
    dropout_rate = np.random.uniform(0,0.3)
    batch_size = np.random.choice([256,512,1024])
    max_nodes = np.random.randint(16, 512)
    activation_function = np.random.choice(['tf.nn.sigmoid',
                                            'tf.nn.relu',
                                            'tf.nn.elu'])

    print("Starting iteration #",i+1)
    log_string = 'LR={},LRD={},WM={},NI={},NL={},DR={},BS={},MN={},AF={}'.format(learning_rate,
                                                                                 learning_rate_decay,
                                                                                 weights_multiplier,
                                                                                 n_inputs,
                                                                                 num_layers,
                                                                                 dropout_rate,
                                                                                 batch_size,
                                                                                 max_nodes,
                                                                                 activation_function) 
    
    model = build_graph(num_layers, n_inputs, weights_multiplier, 
                        dropout_rate,learning_rate,max_nodes,activation_function)
    result = train(model, epochs, log_string, learning_rate)
    results[log_string] = result


# In[189]:

def find_inputs(model):
    '''Use the log_string from the model to extract the values for all of the model's inputs'''
    
    learning_rate_start = model.find('LR=') + 3
    learning_rate_end = model.find(',LRD', learning_rate_start)
    learning_rate = float(model[learning_rate_start:learning_rate_end])
    
    learning_rate_decay_start = model.find('LRD=') + 4
    learning_rate_decay_end = model.find(',WM', learning_rate_decay_start)
    learning_rate_decay = float(model[learning_rate_decay_start:learning_rate_decay_end])
    
    weights_multiplier_start = model.find('WM=') + 3
    weights_multiplier_end = model.find(',NI', weights_multiplier_start)
    weights_multiplier = float(model[weights_multiplier_start:weights_multiplier_end])
    
    n_inputs_start = model.find('NI=') + 3
    n_inputs_end = model.find(',NL', n_inputs_start)
    n_inputs = int(model[n_inputs_start:n_inputs_end])
    
    num_layers_start = model.find('NL=') + 3
    num_layers_end = model.find(',DR', num_layers_start)
    num_layers = int(model[num_layers_start:num_layers_end])
    
    dropout_rate_start = model.find('DR=') + 3
    dropout_rate_end = model.find(',BS', dropout_rate_start)
    dropout_rate = float(model[dropout_rate_start:dropout_rate_end])
    
    batch_size_start = model.find('BS=') + 3
    batch_size_end = model.find(',MN', batch_size_start)
    batch_size = int(model[batch_size_start:batch_size_end])
    
    max_nodes_start = model.find('MN=') + 3
    max_nodes_end = model.find(',AF', max_nodes_start)
    max_nodes = int(model[max_nodes_start:max_nodes_end])
    
    activation_function_start = model.find('AF=') + 3
    activation_function = str(model[activation_function_start:])
    
    return (learning_rate, learning_rate_decay, weights_multiplier, n_inputs,
            num_layers, dropout_rate, batch_size, max_nodes, activation_function)


# In[190]:

# Sort results by RMSE (lowest - highest)
sorted_results_nn = sorted(results.items(), key=operator.itemgetter(1))


# In[191]:

# Create an empty dataframe to contain all of the inputs for each iteration of the model
results_nn = pd.DataFrame(columns=["learning_rate", 
                                   "learning_rate_decay", 
                                   "weights_multiplier", 
                                   "n_inputs",
                                   "num_layers", 
                                   "dropout_rate", 
                                   "batch_size", 
                                   "max_nodes", 
                                   "activation_function"])

for result in sorted_results_nn:
    # Find the input values for each iteration
    learning_rate, learning_rate_decay, weights_multiplier, n_inputs,        num_layers, dropout_rate, batch_size, max_nodes, activation_function = find_inputs(result[0])
    
    # Find the Mean Squared Error for each iteration
    RMSE = result[1]
    
    # Create a dataframe with the values above
    new_row = pd.DataFrame([[RMSE,
                             learning_rate, 
                             learning_rate_decay, 
                             weights_multiplier, 
                             n_inputs,
                             num_layers, 
                             dropout_rate, 
                             batch_size, 
                             max_nodes, 
                             activation_function]],
                     columns = ["RMSE",
                                "learning_rate", 
                                "learning_rate_decay", 
                                "weights_multiplier", 
                                "n_inputs",
                                "num_layers", 
                                "dropout_rate", 
                                "batch_size", 
                                "max_nodes", 
                                "activation_function"])
    
    # Append the dataframe as a new row in results_df
    results_nn = results_nn.append(new_row, ignore_index=True)


# In[192]:

# Look at the top five iterations
results_nn.head()


# In[193]:

def make_predictions(data, batch_size):
    '''
    Restore a session to make predictions, then return these predictions
    data: the data that will be used to make predictions.
    '''
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint)
        predictions = [] # record the predictions

        for batch in range(int(len(data)/batch_size)):
            batch_x = data[batch*batch_size:(1+batch)*batch_size]

            batch_predictions = sess.run([model.preds],
                                   {model.inputs: batch_x,
                                    model.learning_rate: learning_rate,
                                    model.dropout_rate: 0,
                                    model.is_training: False})

            for prediction in batch_predictions[0]:
                predictions.append(prediction)

        return predictions


# In[194]:

initial_preds = {} # stores the RMSE and predictions for x_testFinal
final_preds = {} # store the predictions for testFinal, with x_testFinal's RMSE

iteration = 1 

for model, result in sorted_results_nn:
    checkpoint = str(model) + ".ckpt" 
    
    # Aquire the inputs from the log_string
    _, _, weights_multiplier, n_inputs, num_layers, _, _, max_nodes, activation_function = find_inputs(model)
    
    model = build_graph(num_layers,n_inputs,weights_multiplier,dropout_rate,
                        learning_rate,max_nodes,activation_function)
    
    y_preds_nn = make_predictions(x_testFinal, 659)
    RMSE_nn = np.sqrt(mean_squared_error(y_testFinal, y_preds_nn))
    print("RMSE for iteration #{} is {}.".format(iteration, RMSE_nn))
    print()
    initial_preds[RMSE_nn] = y_preds_nn
    testFinal_preds_nn = make_predictions(testFinal, 258)
    final_preds[RMSE_nn] = [testFinal_preds_nn]
    iteration += 1


# ## Training the Other Models

# In[195]:

# Create an empty dataframe to contain all of the inputs for each iteration of the model
results_rfr = pd.DataFrame(columns=["RMSE",
                                    "n_estimators", 
                                    "max_depth", 
                                    "min_samples_split"])

for i in range(num_iterations):
    # Use random search to choose the inputs' values
    n_estimators = np.random.randint(10,20)
    max_depth = np.random.randint(6,12)
    min_samples_split = np.random.randint(2,50)

    rfr = RFR(n_estimators = n_estimators,
          max_depth = max_depth,
          min_samples_split = min_samples_split,
          verbose = 2,
          random_state = 2)
    
    rfr = rfr.fit(x_train, y_train.values)

    y_preds_rfr = rfr.predict(x_testFinal)
    RMSE_rfr = np.sqrt(mean_squared_error(y_testFinal, y_preds_rfr))
    print("RMSE for iteration #{} is {}.".format(i+1, RMSE_rfr))
    print("NE={}, MD={}, MSS={}".format(n_estimators,
                                        max_depth,
                                        min_samples_split))
    print()
    initial_preds[RMSE_rfr] = y_preds_rfr
    testFinal_preds_rfr = rfr.predict(testFinal)
    final_preds[RMSE_rfr] = [testFinal_preds_rfr]
    
    # Create a dataframe with the values above
    new_row = pd.DataFrame([[RMSE_rfr,
                             n_estimators, 
                             max_depth, 
                             min_samples_split]],
                     columns = ["RMSE",
                                "n_estimators", 
                                "max_depth", 
                                "min_samples_split"])
    
    # Append the dataframe as a new row in results_df
    results_rfr = results_rfr.append(new_row, ignore_index=True)


# In[196]:

# Check the results
results_rfr


# In[197]:

# Create an empty dataframe to contain all of the inputs for each iteration of the model
results_lgb = pd.DataFrame(columns=["RMSE",
                                    "num_leaves", 
                                    "max_depth", 
                                    "feature_fraction",
                                    "bagging_fraction",
                                    "bagging_freq",
                                    "learning_rate"])

for i in range(num_iterations):
    
    num_leaves = np.random.randint(100,250)
    max_depth = np.random.randint(6,12)
    feature_fraction = np.random.uniform(0.7,1)
    bagging_fraction = np.random.uniform(0.8,1)
    bagging_freq = np.random.randint(3,10)
    learning_rate = np.random.uniform(0.2,1)
    n_estimators = 100
    early_stopping_rounds = 5

    gbm = lgb.LGBMRegressor(objective = 'regression',
                            boosting_type = 'gbdt',
                            num_leaves = num_leaves,
                            max_depth = max_depth,
                            feature_fraction = feature_fraction,
                            bagging_fraction = bagging_fraction,
                            bagging_freq = bagging_freq,
                            learning_rate = learning_rate,
                            n_estimators = n_estimators)
    
    gbm.fit(x_train.values, y_train.values.ravel(),
            eval_set = [(x_test.values, y_test.values.ravel())],
            eval_metric = 'rmse',
            early_stopping_rounds = early_stopping_rounds)

    y_preds_gbm = gbm.predict(x_testFinal, num_iteration = gbm.best_iteration)
    RMSE_gbm = np.sqrt(mean_squared_error(y_testFinal, y_preds_gbm))
    print("RMSE for iteration #{} is {}.".format(i+1, RMSE_gbm))
    print("NL={}, MD={}, FF={}, BF={}, BQ={}, LR={}, NE={}, ESR={}".format(num_leaves,
                                                                           max_depth,
                                                                           feature_fraction,
                                                                           bagging_fraction,
                                                                           bagging_freq,
                                                                           learning_rate,
                                                                           n_estimators,
                                                                           early_stopping_rounds))
    print()
    initial_preds[RMSE_gbm] = y_preds_gbm
    testFinal_preds_gbm = gbm.predict(testFinal, num_iteration = gbm.best_iteration)
    final_preds[RMSE_gbm] = [testFinal_preds_gbm]
    
    # Create a dataframe with the values above
    new_row = pd.DataFrame([[RMSE_gbm,
                             num_leaves, 
                             max_depth, 
                             feature_fraction,
                             bagging_fraction,
                             bagging_freq,
                             learning_rate]],
                     columns = ["RMSE",
                                "num_leaves", 
                                "max_depth", 
                                "feature_fraction",
                                "bagging_fraction",
                                "bagging_freq",
                                "learning_rate"])
    
    # Append the dataframe as a new row in results_df
    results_lgb = results_lgb.append(new_row, ignore_index=True)


# In[198]:

results_lgb


# In[199]:

# Create an empty dataframe to contain all of the inputs for each iteration of the model
results_cbr = pd.DataFrame(columns=["RMSE",
                                    "iterations", 
                                    "depth", 
                                    "learning_rate",
                                    "rsm"])

for i in range(num_iterations):

    iterations = np.random.randint(50,250)
    depth = np.random.randint(5,12)
    learning_rate = np.random.uniform(0.5,1)
    rsm = np.random.uniform(0.8,1)

    cbr = CatBoostRegressor(iterations = iterations, 
                            depth = depth, 
                            learning_rate = learning_rate,  
                            rsm = rsm,
                            loss_function='RMSE',
                            use_best_model=True)
    
    cbr.fit(x_train, y_train,
            eval_set = (x_test, y_test),
            use_best_model=True)

    y_preds_cbr = cbr.predict(x_testFinal)
    RMSE_cbr = np.sqrt(mean_squared_error(y_testFinal, y_preds_cbr))
    print("RMSE for iteration #{} is {}.".format(i+1, RMSE_cbr))
    print("I={}, D={}, LR={}, RSM={}".format(iterations,
                                             depth,
                                             learning_rate,
                                             rsm))
    print()
    initial_preds[RMSE_cbr] = y_preds_cbr
    testFinal_preds_cbr = cbr.predict(testFinal)
    final_preds[RMSE_cbr] = [testFinal_preds_cbr]
    
    # Create a dataframe with the values above
    new_row = pd.DataFrame([[RMSE_cbr,
                             iterations, 
                             depth, 
                             learning_rate,
                             rsm]],
                     columns = ["RMSE",
                                "iterations", 
                                "depth",
                                "learning_rate",
                                "rsm"])
    
    # Append the dataframe as a new row in results_df
    results_cbr = results_cbr.append(new_row, ignore_index=True)


# In[200]:

results_cbr


# In[201]:

sorted_initial_RMSE = sorted(initial_preds)
print(sorted_initial_RMSE)


# ## Making Predictions

# In[208]:

best_models = [] # Records teh RMSE of the models to be used for the final predictions
best_RMSE = 99999999999 # records the best RMSE
best_predictions = np.array([0]*len(x_testFinal)) # records the best predictions for each row
current_model = 1 # Used to equally weight the predictions from each iteration

for model in sorted_initial_RMSE:
    
    predictions = initial_preds[model]
    
    RMSE = np.sqrt(mean_squared_error(y_testFinal, predictions))
    print("RMSE = ", RMSE)
    
    # Equally weight each prediction
    combined_predictions = (best_predictions*(current_model-1) + predictions) / current_model
    
    # Find the RMSE with the new predictions
    new_RMSE = np.sqrt(mean_squared_error(y_testFinal, combined_predictions))
    print("New RMSE = ", new_RMSE)
    
    if new_RMSE <= best_RMSE:
        best_predictions = combined_predictions
        best_RMSE = new_RMSE
        best_models.append(model)
        current_model += 1
        print("Improvement!")
        print()
    else:
        print("No improvement.")
        print()


# In[209]:

best_predictions = pd.DataFrame([0]*len(testFinal)) # Records the predictions to be used for submission to Kaggle
current_model = 1

for model in best_models:
    print(model)
    predictions = final_preds[model][0]
    predictions = pd.DataFrame(np.exp(predictions)-1)
    
    combined_predictions = (best_predictions*(current_model-1) + predictions) / current_model
    best_predictions = combined_predictions
    current_model += 1


# In[210]:

# Prepare the dataframe for submitting to Kaggle
best_predictions['id'] = test.id
best_predictions['trip_duration'] = best_predictions[0]
best_predictions = best_predictions.drop([0],1)

best_predictions.to_csv("submission_combined.csv", index=False)


# In[211]:

# Preview the predictions
best_predictions.head()


# In[212]:

# Compare the predicted values with the training values - the distribution should be similar
best_predictions.trip_duration.describe()


# In[207]:

yFinal.describe()


# ## Summary

# This ensemble approach with random search has worked rather well. Currently, I am ranked in the top 13% of this competition. Creating numerous features and fine-tuning the range for the random searches were critical to the success of this work. 

# In[ ]:




import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

#function that works for evaluating only one file.
def learning_one_file(input_len, output_len, trainingFile, testingFile):
    #read train and result data seperately
    train_data, result_data = read_one_file(trainingFile)
    test_data, test_result = read_one_file(testingFile)

    #crete a sequential model with 12 neurons in input layer and 4 hidden 8-neurons layers. 
    #And output layer with softmax function. 
    model = Sequential()
    model.add(Dense(12, input_dim=input_len, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(output_len, activation='softmax'))

    #use rmsprop optimizer categorical_crossentropy loss function because they are working best 
    #in our dataset.
    model.compile(loss='categorical_crossentropy', optimizer="RMSprop", metrics=['accuracy'])
    model.fit(train_data, result_data, epochs=150, batch_size=50)

    #evaluate the result with real result.
    results, accuracy = model.evaluate(test_data, test_result)
    print("Correct prediction rate: {}".format(accuracy*100))

#function that works for evaluating two files.
def learning_two_files(input_len, output_len, training_file1,testing_file1,training_file2,testing_file2):    
    #read train and result data seperately
    train_data, result_data = read_two_files(training_file1,training_file2)
    test_data, test_result = read_two_files(testing_file1,testing_file2)
    #crete a sequential model with 12 neurons in input layer and 4 hidden 8-neurons layers. 
    #And output layer with softmax function. 
    model = Sequential()
    model.add(Dense(12, input_dim=input_len, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(output_len, activation='softmax'))

    #use rmsprop optimizer categorical_crossentropy loss function because they are working best 
    #in our dataset.
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    model.fit(train_data, result_data, epochs=150, batch_size=50)

    #evaluate the result with real result.
    results, accuracy = model.evaluate(test_data, test_result)
    print("Correct prediction rate: {}".format(accuracy*100))

#function that works for reading one file.
def read_one_file(filename):
    #create empty lists.
    dataset = list()
    resultset=list()
    #things that we should skip while reading.
    skip = ['@RELATION', '@ATTRIBUTE', '@DATA']

    #open file
    with open(filename) as file:
        #loop through lines.
        for line in file:
            #if line contains any skip item, passs that line.
            if not any(x in line for x in skip):
                #split by , and clean the spaces, store it in result.
                result = [x.strip() for x in line.split(',')]

                #if result is not empy, the first lines are dataset, last line is result set.
                if len(result) > 1:
                    dataset.append(result[:len(result) - 1])
                    resultset.append([result[len(result) - 1]])
    

    #for better accuracy, we transform dataset and result sets in more appropriate forms.
    encoder = OneHotEncoder()
    resultset = encoder.fit_transform(resultset).toarray()
    
    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataset)

    dataset = np.array(dataset,dtype=np.float)
    resultset = np.array(resultset,dtype=np.float)

    return [dataset,resultset]

#function that works for reading two files.
def read_two_files(filename1,filename2):
    #create empty lists.
    dataset = list()
    dataset1 = list()
    dataset2 = list()
    resultset=list()
    #things that we should skip while reading.
    skip = ['@RELATION', '@ATTRIBUTE', '@DATA']
    #open file
    with open(filename1) as file:
        #loop through lines.
        for line in file:
            #if line contains any skip item, passs that line.
            if not any(x in line for x in skip):
                #split by , and clean the spaces, store it in result.
                result = [x.strip() for x in line.split(',')]
                #if result is not empy, the first lines are dataset, last line is result set.
                if len(result) > 1:
                    dataset1.append(result[:len(result) - 1])
                    resultset.append([result[len(result) - 1]])

    #same for first file reading.
    with open(filename2) as file:
        for line in file:
            if not any(x in line for x in skip):
                result = [x.strip() for x in line.split(',')]
                if len(result) > 1:
                    dataset2.append(result[:len(result) - 1])
    
    #concatanate each elements of datasets together
    for i,j in zip(dataset1,dataset2):
        dataset.append(i+j)

    #for better accuracy, we transform dataset and result sets in more appropriate forms.
    encoder = OneHotEncoder()
    resultset = encoder.fit_transform(resultset).toarray()
    
    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataset)

    dataset = np.array(dataset,dtype=np.float)
    resultset = np.array(resultset,dtype=np.float)

    return [dataset,resultset]


if __name__ == '__main__':
    #learning_one_file(5,3,'IrisGeometicFeatures_TrainingSet.txt','IrisGeometicFeatures_TestingSet.txt')
    #learning_one_file(9600, 3,'IrisTextureFeatures_TrainingSet.txt','IrisTextureFeatures_TestingSet.txt')
    learning_two_files(9605, 3,'IrisGeometicFeatures_TrainingSet.txt','IrisGeometicFeatures_TestingSet.txt','IrisTextureFeatures_TrainingSet.txt','IrisTextureFeatures_TestingSet.txt')
    

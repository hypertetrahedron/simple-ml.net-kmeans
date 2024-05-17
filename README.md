# simple-ml.net-kmeans
A simple command line app written with ML.NET to cluster data from a CSV file. 

# Command Line Usage
kMeans.exe --inputCSVFilename \<inputCsvFilename\> --outputCSVFilename \<outputCsvFilename\> --numberOfClusters \<numberOfClusters\> --headerRow \<hasHeaderRow\>

\<inputCsvFilename\> - path to a CSV file containing the features to be classified. The first column should be an identifying label for the data (so you can track it in the output) and all further columns must be features in the form of floating point values. 

\<outputCsvFilename\> - path where a CSV file will be written containing the cluster results. The columns of this csv are Label (matches the original label provided in the input file), Features(1-n), ClusterID, DistanceToCluster(1-n).

\<numberOfClusters\> - the number of clusters to use. 

\<hasHeaderRow\> - a boolean indicated if the first non-empty row of the csv contains a header, which will be ignored. 

# Example Data
The folder 'example' contains an example input file and the folader 'example-output' contains an example of the output to expect after running the application. 

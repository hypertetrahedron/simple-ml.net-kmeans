using Microsoft.ML.Data;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace kMeans
{
    public class CSVDataFile
    {
        /// <summary>
        /// Confirm the specified CSV file contains a consistant number of features for every row.
        /// </summary>
        /// <param name="filename">Filename of the CSV to validate</param>
        /// <param name="hasolumnHeader">Does the CSV contain a column header row. If true the first non-null row will be ignored.</param>
        /// <returns>Number of features each row contains</returns>
        /// <exception cref="Exception">An exception is thrown if the number of features among the rows are not equal. The error message of the exception will contain a list of all non-conforming rows.</exception>
        public static int Validate(string filename, bool hasolumnHeader)
        {
            // verify our file exists
            if (!File.Exists(filename)) { throw new Exception($"The csv file specified {filename}, could not be found"); }

            // setup our csv parser
            var parser = new Microsoft.VisualBasic.FileIO.TextFieldParser(filename);
            parser.TextFieldType = Microsoft.VisualBasic.FileIO.FieldType.Delimited;
            parser.SetDelimiters(new string[] { ",", "\t" });

            var headerRow = hasolumnHeader;
            int? numberOfFeatures = null;
            var errors = new List<string>();
            int rowIndex = 0;
            while (!parser.EndOfData)
            {
                // kep track of which row we are on for error reporting
                rowIndex++;

                // get the next row
                string[] row = parser.ReadFields();

                // ignore any rows which are empty
                if (row == null) { continue; }

                // skip the header row
                if (headerRow) { headerRow = false; continue; }

                // column 0 must be the label
                var label = row[0] != null ? row[0] : "notSpecified";

                // validate all remaining columns contain the same number of featues. 
                var currentNumberOfFeatrues = row.Length - 1;
                if (numberOfFeatures == null) { numberOfFeatures = currentNumberOfFeatrues; }
                if (numberOfFeatures != currentNumberOfFeatrues)
                {
                    errors.Add($"Row {rowIndex} has {currentNumberOfFeatrues} columns of features instead of the expected {numberOfFeatures}");
                }
            }

            // throw an error if any are present
            if (errors.Count > 0)
            {
                var listOfErrors = errors.Aggregate((aggregate, currentValue) => aggregate + Environment.NewLine + currentValue );
                throw new Exception($"Errors were encountered while processing the file {filename}. These errors are as follows: {Environment.NewLine + listOfErrors}");
            }

            // if the file is valid then return the number of features found
            return numberOfFeatures.Value;
        }


        /// <summary>
        /// Creates an IDataView from the specified CSV file. 
        /// </summary>
        /// <param name="mlContext">MLContext</param>
        /// <param name="filename">Name of the CSV file to be loaded.</param>
        /// <param name="numberOfFeatures">Number of features to extract from the CSV file. Features cannot occur in the 0th column since the label is exptected in that location. Any number of contigous features can be loaded from columns 1 through n. All features must be floating point values. All rows must have the same number of featues.</param>
        /// <returns></returns>
        public static IDataView Load(MLContext mlContext, string filename, int numberOfFeatures)
        {
            // create the feature column based on the number of features specified. 
            // we assume the first column (index 0) is the label and all remaining columns are features.

            var featureRange = new TextLoader.Range[numberOfFeatures];
            for (int i = 0; i < numberOfFeatures; i++)
            {
                // adding 1 to the index to offset the range past the label at index 0
                featureRange[i] = new TextLoader.Range(i + 1);
            }

            // load the data
            var data = mlContext.Data.LoadFromTextFile(filename,
            new TextLoader.Options()
            {
                Separators = [',', '\t'],
                TrimWhitespace = true,
                HasHeader = true,
                Columns =
                [
                    new TextLoader.Column("Label",DataKind.String,0),
                new TextLoader.Column("Features", DataKind.Single, featureRange)
                ]
            });
            return data;
        }

        /// <summary>
        /// Normalize the data using MinMax
        /// </summary>
        /// <param name="mlContext">MLContext</param>
        /// <param name="data">Data whose features will be normalized</param>
        /// <returns>Normalized data</returns>
        public static IDataView Normalize(MLContext mlContext, IDataView data)
        {
            var featureColumns = data.Schema
            .Select( col => col.Name)
            .Where(col => col != "Label")
            .Select( col => new InputOutputColumnPair(col, col))
            .ToArray();
            var scaling = mlContext.Transforms.NormalizeMinMax(featureColumns);
            var scaledData = scaling.Fit(data).Transform(data);
            return scaledData;
        }

        /// <summary>
        /// Writes the results of this clustiner to the filename specified. 
        /// </summary>
        /// <param name="data">KMeansResults to write out.</param>
        /// <param name="outputFilename">Filename of the file to write the specified KMeansResults out to. This file will be in a csv format. It is recommended users specifiy a filename with a .csv extension.</param>
        /// <exception cref="Exception"></exception>
        public static void Write(List<KMeansResult> data, string outputFilename)
        {
            // we cannot write out the data if there is no data to write out. we cannot even infer the number of featues
            if (data.Count < 1)
            {
                throw new Exception("Cannot write out an empty data set.");
            }

            // extract the feature count from the first data element
            var featureCount = data[0].Features.Length;

            // extract the number of cluster
            var clusterCount = data[0].Score.Length;

            // open our output file
            Directory.CreateDirectory(Path.GetDirectoryName(outputFilename));
            using (var outputFile = new StreamWriter(outputFilename))
            {
                // write the column row
                outputFile.Write("Label,");
                for (int i = 0; i < featureCount; i++) { outputFile.Write($"Feature{i},"); }
                outputFile.Write($"ClusterID,");
                // adding 1 to the index to offset it from a 0 based to 1 based index. this is done here because the cluster ID
                // is a 1 based index, so this change will keep the column names in line with the cluster ID values. 
                for (int i = 0; i < clusterCount; i++) { outputFile.Write($"DistanceToCluster{i+1},"); }
                outputFile.Write(Environment.NewLine);

                // write the data rows
                foreach (var dataElement in data)
                {
                    outputFile.Write($"{dataElement.Label}");
                    for (int i = 0; i < featureCount; i++) { outputFile.Write($",{dataElement.Features.GetValues()[i]}"); }
                    outputFile.Write($",{dataElement.PredictedLabel}");
                    for (int i = 0; i < clusterCount; i++) { outputFile.Write($",{dataElement.Score.GetValues()[i]}"); }
                    outputFile.Write(Environment.NewLine);
                }
            }
        }
    }
}

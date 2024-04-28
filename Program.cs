using kMeans;
using Microsoft.ML;
using Microsoft.ML.Data;
using CommandLine;
using static System.Runtime.InteropServices.JavaScript.JSType;

internal class Program
{
    public class CommandLineArgs
    {
        [Option('i', "inputCSVFilename", Required = true, HelpText = "Filename of a CSV file containing labels and features for clustering.")]
        public string InputCSVFilename { get; set; }
        
        [Option('o', "outputCSVFilename", Required = true, HelpText = "Filename where the output CSV file containing labels, features, clusterIDs, and scores will be written.")]
        public string OutputCSVFilename { get; set; }

        [Option('c', "numberOfClusters", Required = true, HelpText = "The number of clusters to generate.")]
        public int NumberOfClusters { get; set; }

        [Option('h', "headerRow", Required = false, Default =true, HelpText = "Boolean indicating if the CSV specified by inputCSVFilename contains a header row.")]
        public bool HasHeaderRow { get; set; }
    }

    private static void Main(string[] args)
    {
        Parser.Default.ParseArguments<CommandLineArgs>(args)
                   .WithParsed<CommandLineArgs>(args =>
                   {
                       KMeansClustering(args.InputCSVFilename, args.NumberOfClusters, args.OutputCSVFilename, args.HasHeaderRow);
                   });
    }

    public static void KMeansClustering(string inputCsvFilename, int numberOfClusters, string outputCsvFilename, bool inputCsvHasHeaderColumn)
    {
        // verify our file is good and get its feature count
        var numberOfFeatures = CSVDataFile.Validate(inputCsvFilename, inputCsvHasHeaderColumn);

        Console.WriteLine($"Detected {numberOfFeatures} features in input CSV.");

        // context with a fixed seed so results will repeat. 
        var mlContext = new MLContext(seed: 0);

        // load the data from the csv file
        var data = CSVDataFile.Load(mlContext, inputCsvFilename, numberOfFeatures);

        // setup our kmeans pipeline
        var pipeline = mlContext.Clustering.Trainers.KMeans(numberOfClusters: numberOfClusters);

        // execute pipeline
        System.Console.WriteLine($"Running KMeans with {numberOfClusters} clusters.");
        var model = pipeline.Fit(data);

        // since this is a clustering task we will run the new model over the data
        System.Console.WriteLine($"Identifying clusters of data.");
        var clusteredData = model.Transform(data);

        // Convert IDataView object to a list.
        var clusterPredictions = mlContext.Data.CreateEnumerable<KMeansResult>(clusteredData, reuseRowObject: false).ToList();

        Console.WriteLine($"Generated {clusterPredictions.Count} results.");

        // write our output
        CSVDataFile.Write(clusterPredictions, outputCsvFilename);

        // print out metrics
        var metrics = mlContext.Clustering.Evaluate(clusteredData, null, "Score", "Features");
        PrintMetrics(clusterPredictions, metrics);        
    }   

    /// <summary>
    /// Display basic metrics of a clustering operation.
    /// </summary>
    /// <param name="kMeansResults">KMeans results</param>
    private static void PrintMetrics(List<KMeansResult> kMeansResults, ClusteringMetrics metrics)
    {
        Console.WriteLine($"Average Distance: " + $"{metrics.AverageDistance:F2}");
        
        kMeansResults.Sort((a, b) => a.Score.GetValues()[(int)a.PredictedLabel - 1].CompareTo(b.Score.GetValues()[(int)b.PredictedLabel - 1]));
        var bestFit = kMeansResults.First();
        var worstFit = kMeansResults.Last();
        Console.WriteLine($"Best fit Label={bestFit.Label} Distance={bestFit.Score.GetValues()[(int)bestFit.PredictedLabel - 1]}");
        Console.WriteLine($"Wost fit Label={worstFit.Label} Distance={worstFit.Score.GetValues()[(int)worstFit.PredictedLabel-1]}");

        // display cluster populations
        Console.WriteLine("Cluster Populations");
        foreach (var line in kMeansResults.GroupBy(x => x.PredictedLabel)
                        .Select(group => new {
                            PredictedLabel = group.Key,
                            Count = group.Count()
                        })
                        .OrderBy(x => x.PredictedLabel))
        {
            Console.WriteLine($"ClusterId:{line.PredictedLabel} : {line.Count} members");
        }
    }

    

    
    

    
}
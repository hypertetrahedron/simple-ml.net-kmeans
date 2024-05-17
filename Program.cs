using kMeans;
using Microsoft.ML;
using Microsoft.ML.Data;
using CommandLine;
internal class Program
{
    public class CommandLineArgs
    {
        [Option('i', "inputCSVFilename", Required = true, HelpText = "Filename of a CSV file containing labels and features for clustering.")]
        public string InputCSVFilename { get; set; }
        
        [Option('h', "headerRow", Required = false, Default =true, HelpText = "Boolean indicating if the CSV specified by inputCSVFilename contains a header row.")]
        public bool HasHeaderRow { get; set; }

        [Option('o', "outputCSVFilename", Required = true, HelpText = "Filename where the output CSV file containing labels, features, clusterIDs, and scores will be written.")]
        public string OutputCSVFilename { get; set; }

        [Option('c', "numberOfClusters", Required = false, Default =0, HelpText = "The number of clusters to generate.")]
        public int NumberOfClusters { get; set; }

        [Option('s', "startClusterCount", Required = false, Default = 0, HelpText = "The starting number of clusters to start with during a sweep.")]
        public int StartClusterCount { get; set; }

        [Option('e', "endClusterCount", Required = false, Default =0 , HelpText = "The ending number of clusters to start with during a sweep.")]
        public int EndClusterCount { get; set; }

        [Option('n', "normalizeData", Required = false, Default = true, HelpText = "If set all features will be normalized.")]
        public bool NormalizeData { get; set; }
    }

    private static void Main(string[] args)
    {
        
        try
        {
            // context with a fixed seed so results will repeat. 
            var mlContext = new MLContext(seed: 0);

            Parser.Default.ParseArguments<CommandLineArgs>(args)
                    .WithParsed<CommandLineArgs>(args =>
                    {
                            // set cluster values
                            if( args.NumberOfClusters == 0 && args.StartClusterCount == 0 && args.EndClusterCount == 0)
                            {
                                throw new Exception( "A value must be provided for either numberOfClusters, or for startClusterCount and endClusterCount");
                            }
                            if( args.NumberOfClusters != 0)
                            {
                                args.StartClusterCount = args.NumberOfClusters;
                                args.EndClusterCount = args.NumberOfClusters;
                            }

                            // run kmeans sweep
                            Sweep(mlContext, args.InputCSVFilename, args.StartClusterCount, args.EndClusterCount, args.OutputCSVFilename, args.HasHeaderRow, args.NormalizeData);
                    });
        }
        catch(Exception ex)
        {
            Console.WriteLine($"Exception: {ex.ToString()}");
        }
    }

    private static void Sweep(MLContext mlContext, string inputCsvFilename, int clusterStartRange, int clusterEndRange, string outputCsvFilename, bool inputCsvHasHeaderColumn, bool normalizeData)
    {
        // load input data 
        // verify our file is good and get its feature count
        Console.WriteLine($"Validating {inputCsvFilename}");
        var numberOfFeatures = CSVDataFile.Validate(inputCsvFilename, inputCsvHasHeaderColumn);
        Console.WriteLine($"Detected {numberOfFeatures} features in input CSV.");       

        // load the data from the csv file
        Console.WriteLine($"Loading {inputCsvFilename}");
        var data = CSVDataFile.Load(mlContext, inputCsvFilename, numberOfFeatures);

        // scale data
        if( normalizeData)
        {
            Console.WriteLine($"Scaling data");
            data = CSVDataFile.Normalize(mlContext, data);
        }

        using(var metricsFile = new StreamWriter(outputCsvFilename + "-clusterMetrics.csv"))
        {
            metricsFile.WriteLine("ClusterCount,AverageDistance,BestDistance,WorstDistance");
            Parallel.For( clusterStartRange, clusterEndRange+1, clusterSize =>
            {
                var outputFile = $"{outputCsvFilename}-{clusterSize}.csv";
                KMeansClustering(mlContext, data, clusterSize, outputFile, metricsFile);
            });
        }
    }

    public static void KMeansClustering(MLContext mlContext, IDataView data, int numberOfClusters, string outputCsvFilename, TextWriter metricsFile = null)
    {
        // setup our kmeans pipeline
        var pipeline = mlContext.Clustering.Trainers.KMeans(numberOfClusters: numberOfClusters);

        // execute pipeline
        Console.WriteLine($"Running KMeans with {numberOfClusters} clusters.");
        var model = pipeline.Fit(data);

        // since this is a clustering task we will run the new model over the data
        Console.WriteLine($"Identifying clusters of data.");
        var clusteredData = model.Transform(data);

        // Convert IDataView object to a list.
        var clusterPredictions = mlContext.Data.CreateEnumerable<KMeansResult>(clusteredData, reuseRowObject: false).ToList();

        Console.WriteLine($"Generated {clusterPredictions.Count} results.");

        // write our output
        CSVDataFile.Write(clusterPredictions, outputCsvFilename);

        // print out metrics
        var metrics = mlContext.Clustering.Evaluate(clusteredData, null, "Score", "Features");
        PrintMetrics(clusterPredictions, metrics, metricsFile);        
    }   

    private static object metricsFileLock = new object();
    /// <summary>
    /// Display basic metrics of a clustering operation.
    /// </summary>
    /// <param name="kMeansResults">KMeans results</param>
    private static void PrintMetrics(List<KMeansResult> kMeansResults, ClusteringMetrics metrics, TextWriter metricsFile = null)
    {   
        var totalDistances = kMeansResults.Aggregate( 0f, (total, next) => total += next.Score.GetValues()[(int)next.PredictedLabel-1]);
        var averageDistance = totalDistances / kMeansResults.Count;
        Console.WriteLine($"Average Distance: " + $"{averageDistance}");

        kMeansResults.Sort((a, b) => a.Score.GetValues()[(int)a.PredictedLabel - 1].CompareTo(b.Score.GetValues()[(int)b.PredictedLabel - 1]));
        var bestFit = kMeansResults.First();
        var worstFit = kMeansResults.Last();
        var bestFitDistance = bestFit.Score.GetValues()[(int)bestFit.PredictedLabel - 1];
        var worstFitDistance = worstFit.Score.GetValues()[(int)worstFit.PredictedLabel-1];
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
        if( metricsFile != null)
        {
            lock(metricsFileLock)
            {
                metricsFile.WriteLine( $"{kMeansResults.GroupBy(x => x.PredictedLabel).Count()},{averageDistance},{bestFitDistance},{worstFitDistance}" );
                metricsFile.Flush();
            }
        }
    }

    

    
    

    
}
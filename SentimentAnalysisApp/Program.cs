using Microsoft.ML;
using System;
using System.IO;

namespace SentimentAnalysisApp
{
   
    class Program
    {
        private static readonly string BaseModelsRelativePath = @"../../../../MLModels";
        private static readonly string ModelRelativePath = $"{BaseModelsRelativePath}/SentimentModel.zip";
        private static readonly string ModelPath = GetAbsolutePath(ModelRelativePath);

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();
            ITransformer mlModel = mlContext.Model.Load(ModelPath, out _);
            var predEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(mlModel);
            SentimentData sampleData = new SentimentData { SentimentText = "i am loving it" };
            var predictionResult = predEngine.Predict(sampleData);
            Console.WriteLine($"prediction ::  {predictionResult.Prediction} , probability :: {(Convert.ToBoolean(predictionResult.Probability) ? "Positive" : "Negative")} , score :: {predictionResult.Score}");
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}

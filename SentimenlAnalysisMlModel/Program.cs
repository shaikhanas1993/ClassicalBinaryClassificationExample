using Microsoft.ML;
using System;
using System.IO;
using static Microsoft.ML.DataOperationsCatalog;

namespace SentimenlAnalysisMlModel
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");
        private static readonly string BaseModelsRelativePath = @"../../../../MLModels";
        private static readonly string ModelRelativePath = $"{BaseModelsRelativePath}/SentimentModel.zip";

        private static readonly string ModelPath = GetAbsolutePath(ModelRelativePath);

        static void Main(string[] args)
        {
            MLContext mLContext = new MLContext();

            //load data step
            IDataView dataView =  mLContext.Data.LoadFromTextFile<SentimentData>(_dataPath ,hasHeader:false);

            //split the data set into train data set and test data set
            TrainTestData splitDataView = mLContext.Data.TrainTestSplit(dataView,0.2);

            var trainDataSet = splitDataView.TrainSet;
            var testDataSet = splitDataView.TestSet;

            //process data and vectorize data
            var dataProcessPipeline = mLContext.Transforms.Text.FeaturizeText(outputColumnName: "Features",inputColumnName: nameof(SentimentData.SentimentText));

            //define a trainer
            var trainer = mLContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName:"Label",featureColumnName: "Features");

            //add trainer to the pipeline
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            //train the model
            ITransformer trainedModel = trainingPipeline.Fit(trainDataSet);



            //evaluate the model matrics
            var predictions = trainedModel.Transform(testDataSet);
            var metrics = mLContext.BinaryClassification.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");
            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");

            //save model
            mLContext.Model.Save(trainedModel,trainDataSet.Schema, ModelPath);

            //Test your model
            //SentimentData sampleData = new SentimentData { SentimentText = "i am loving it" };

            ////create a prediction engine
            //var predictionEngine = mLContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(trainedModel);

            //var predictionResult = predictionEngine.Predict(sampleData);
            //Console.WriteLine($"prediction ::  {predictionResult.Prediction} , probability :: {(Convert.ToBoolean(predictionResult.Probability) ? "Positive":"Negative" )} , score :: {predictionResult.Score}");
            

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

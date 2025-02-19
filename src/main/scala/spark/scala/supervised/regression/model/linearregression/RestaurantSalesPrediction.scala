package spark.scala.supervised.regression.model.linearregression

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession
import spark.scala.common.logger.Logging

object RestaurantSalesPrediction extends Logging {
  def main(args: Array[String]): Unit = {
    // Initialize Spark Session
    val spark = SparkSession.builder()
      .appName("LinearRegressionExample")
      .master("local[*]") // Use all available cores
      .getOrCreate()

    import spark.implicits._

    logger.info("Sample dataset: Ad Spend ($) vs Sales Revenue ($)")
    val data = Seq(
      (2000.0, 25000.0),
      (3000.0, 30000.0),
      (5000.0, 50000.0),
      (4000.0, 40000.0),
      (6000.0, 55000.0),
      (7000.0, 62000.0),
      (10000.0, 85000.0),
      (8000.0, 73000.0),
      (9000.0, 80000.0),
      (11000.0, 92000.0)
    ).toDF("Ad_Spend", "Sales")

    logger.info("Convert feature column into Vector format (required for Spark ML)")
    val assembler = new VectorAssembler()
      .setInputCols(Array("Ad_Spend"))
      .setOutputCol("features")

    val finalData = assembler.transform(data)

    logger.info("Split data into training (80%) and testing (20%) sets")
    val Array(trainingData, testData) = finalData.randomSplit(Array(0.8, 0.2), seed = 42)

    logger.info("Define and train the Linear Regression model")
    val lr = new LinearRegression()
      .setLabelCol("Sales")
      .setFeaturesCol("features")

    val model = lr.fit(trainingData)

    logger.info("Print the coefficients and intercept")
    logger.info(s"\nLinear Regression Model: Sales = ${model.coefficients(0)} * Ad_Spend + ${model.intercept}")

    // Make predictions
    val predictions = model.transform(testData)

    // Show predictions
    predictions.select("Ad_Spend", "Sales", "prediction").show()

    logger.info("Predict sales for $7,000 Ad Spend")
    val newInput = Seq(7000.0).toDF("Ad_Spend")
    val newInputTransformed = assembler.transform(newInput)
    val predictedSales = model.transform(newInputTransformed).select("prediction").as[Double].collect()(0)
    logger.info(f"Predicted Sales for $$7000 Ad Spend: $$$predictedSales%.2f")

    // Stop Spark session
    spark.stop()
  }
}

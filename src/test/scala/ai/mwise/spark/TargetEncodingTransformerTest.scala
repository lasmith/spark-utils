package ai.mwise.spark

import java.io.File

import org.apache.commons.io.FileDeleteStrategy
import org.apache.spark.ml.feature.{TargetEncodingModel, TargetEncodingTransformer}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.scalactic.{Equality, TolerantNumerics}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}


/**
 * @author Laurence Smith
 */
class TargetEncodingTransformerTest extends FlatSpec with Matchers with BeforeAndAfter {
  var sparkSession: SparkSession = _

  before {
    sparkSession = SparkSessionFactory.createLocalSparkContext()
  }

  behavior of "fit"

  it should " handle an empty data frame and do nothing" in {
    // Given
    val schema = StructType(
      StructField("label", StringType, nullable = false) ::
        StructField("feature", IntegerType, nullable = false) ::
        StructField("expected", DoubleType, nullable = false) :: Nil
    )
    val df = sparkSession.createDataFrame(sparkSession.sparkContext.emptyRDD[Row], schema)
    val woe = new TargetEncodingTransformer().setInputCol("feature").setOutputCol("woe")

    // When
    val model = woe.fit(df)

    // Then
    model should not be null
    val rows = model.transform(df).collect()
    assert(rows.length == 0)
  }

  it should " return a new dataframe with the target encoded column given boolean target" in {
    // Given
    val df = sparkSession.createDataFrame(Seq(
      (true, "ab", 0.6666),
      (false, "ab", 0.6666),
      (true, "ab", 0.6666),
      (false, "cd", 0.5),
      (true, "cd", 0.5),
      (true, "ef", 1d),
      (false, "gh", 0d)
    )).toDF("label", "feature", "expected")
    runFitAndCheck(df, smoothingEnabled = false)
  }

  it should " return a new dataframe with the target encoded column given int(1|0) target" in {
    // Given
    val df = getBasicDataFrame
    runFitAndCheck(df, smoothingEnabled = false)
  }

  it should " return a new dataframe with the target encoded column given boolean target and smoothing" in {
    // Given
    val globalMean = 0.5714285714285714
    val weight = 100
    val df: DataFrame = getSmoothedDataFrame(globalMean, weight)
    runFitAndCheck(df, smoothingEnabled = true, smoothingWeight = weight)
  }

  behavior of "persistence API"

  it should " save a model to file" in {
    // Given
    val df = getBasicDataFrame
    val model = runFit(df, smoothingEnabled = false, smoothingWeight = 100)

    val file = new File(TestUtils.getTestResourceDir, "out_model")
    FileDeleteStrategy.FORCE.delete(file)
    println(s"Saving model to $file")
    model.save(file.getAbsolutePath)
  }

  it should " load a model from file and correctly fit / encode basic" in {
    // Given
    val df = getBasicDataFrame
    val model = runFit(df, smoothingEnabled = false, smoothingWeight = 100)

    // When
    val file = new File(TestUtils.getTestResourceDir, "out_model")
    FileDeleteStrategy.FORCE.delete(file)
    println(s"Saving model to $file")
    model.save(file.getAbsolutePath)
    val modelFromDisk = TargetEncodingModel.load(file.getAbsolutePath)

    // Then
    checkDataFrame(df, modelFromDisk)

  }

  it should " load a model from file and correctly fit / encode smoothed" in {
    // Given
    val globalMean = 0.5714285714285714
    val weight = 100
    val df: DataFrame = getSmoothedDataFrame(globalMean, weight)
    val model = runFit(df, smoothingEnabled = true, smoothingWeight = 100)

    // When
    val file = new File(TestUtils.getTestResourceDir, "out_model")
    FileDeleteStrategy.FORCE.delete(file)
    println(s"Saving model to $file")
    model.save(file.getAbsolutePath)
    val modelFromDisk = TargetEncodingModel.load(file.getAbsolutePath)

    // Then
    checkDataFrame(df, modelFromDisk)

  }

  // ---------------------------------------- HELPERS ----------------------------------------

  private def getSmoothedMean(globalMean: Double, weight: Int, noOfRecords: Int, mean: Double) = {
    (noOfRecords * mean + weight * globalMean) / (noOfRecords + weight)
  }

  private def runFit(df: DataFrame, smoothingEnabled: Boolean, smoothingWeight: Double) = {
    val transformer = new TargetEncodingTransformer()
      .setInputCol("feature")
      .setOutputCol("te")
      .setSmoothingEnabled(smoothingEnabled)
      .setSmoothingWeight(smoothingWeight)

    // When
    val model = transformer.fit(df)
    model
  }

  private def runFitAndCheck(df: DataFrame, smoothingEnabled: Boolean, smoothingWeight: Double = 100d): Unit = {
    val model: TargetEncodingModel = runFit(df, smoothingEnabled, smoothingWeight)
    checkDataFrame(df, model)
  }

  private def checkDataFrame(df: DataFrame, model: TargetEncodingModel): Unit = {
    // Then
    model should not be null
    val epsilon = 1e-4f
    implicit val doubleEq: Equality[Double] = TolerantNumerics.tolerantDoubleEquality(epsilon)
    val rows = model.transform(df).collect()
    for (r <- rows) {
      val encVal = r.getAs[Double]("te")
      val expected = r.getAs[Double]("expected")
      assert(encVal === expected, r.getAs[String]("feature")) // roughly equal to 4dp based on epsilon above to avoid floating point issues..
    }
  }

  private def getBasicDataFrame = {
    sparkSession.createDataFrame(Seq(
      (1, "ab", 0.6666),
      (0, "ab", 0.6666),
      (1, "ab", 0.6666),
      (0, "cd", 0.5),
      (1, "cd", 0.5),
      (1, "ef", 1d),
      (0, "gh", 0d)
    )).toDF("label", "feature", "expected")
  }

  private def getSmoothedDataFrame(globalMean: Double, weight: Int) = {
    val ab = getSmoothedMean(globalMean, weight, 3, 0.6666)
    val cd = getSmoothedMean(globalMean, weight, 2, 0.5)
    val ef = getSmoothedMean(globalMean, weight, 1, 1)
    val gh = getSmoothedMean(globalMean, weight, 1, 0)
    val df = sparkSession.createDataFrame(Seq(
      (true, "ab", ab),
      (false, "ab", ab),
      (true, "ab", ab),
      (false, "cd", cd),
      (true, "cd", cd),
      (true, "ef", ef),
      (false, "gh", gh)
    )).toDF("label", "feature", "expected")
    df
  }
}

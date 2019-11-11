package com.msxi.mwise.autopay

import org.apache.spark.ml.feature.TargetEncodingTransformer
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.scalactic.TolerantNumerics
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}


/**
 * <br> <br>
 * Copyright:    Copyright (c) 2019 <br>
 * Company:      MSX-International  <br>
 *
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
    runTransformAndCheck(df, smoothingEnabled = false)
  }

  it should " return a new dataframe with the target encoded column given int(1|0) target" in {
    // Given
    val df = sparkSession.createDataFrame(Seq(
      (1, "ab", 0.6666),
      (0, "ab", 0.6666),
      (1, "ab", 0.6666),
      (0, "cd", 0.5),
      (1, "cd", 0.5),
      (1, "ef", 1d),
      (0, "gh", 0d)
    )).toDF("label", "feature", "expected")
    runTransformAndCheck(df, smoothingEnabled = false)
  }


  it should " return a new dataframe with the target encoded column given boolean target and smoothing" in {
    // Given
    val globalMean = 0.5714285714285714
    val weight = 100
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
    runTransformAndCheck(df, smoothingEnabled = true, smoothingWeight = weight)
  }


  private def getSmoothedMean(globalMean: Double, weight: Int, noOfRecords: Int, mean: Double) = {
    (noOfRecords * mean + weight * globalMean) / (noOfRecords + weight)
  }

  private def runTransformAndCheck(df: DataFrame, smoothingEnabled: Boolean, smoothingWeight: Double = 100d) = {
    val transformer = new TargetEncodingTransformer()
      .setInputCol("feature")
      .setOutputCol("te")
      .setSmoothingEnabled(smoothingEnabled)
      .setSmoothingWeight(smoothingWeight)

    // When
    val model = transformer.fit(df)

    // Then
    model should not be null
    val epsilon = 1e-4f
    implicit val doubleEq = TolerantNumerics.tolerantDoubleEquality(epsilon)
    val rows = model.transform(df).collect()
    for (r <- rows) {
      val encVal = r.getAs[Double]("te")
      val expected = r.getAs[Double]("expected")
      assert(encVal === expected, r.getAs[String]("feature")) // roughly equal to 4dp based on epsilon above to avoid floating point issues..
    }
  }
}

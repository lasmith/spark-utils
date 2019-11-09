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
    runTransformAndCheck(df)
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
    runTransformAndCheck(df)
  }


  private def runTransformAndCheck(df: DataFrame) = {
    val transformer = new TargetEncodingTransformer().setInputCol("feature").setOutputCol("te")

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

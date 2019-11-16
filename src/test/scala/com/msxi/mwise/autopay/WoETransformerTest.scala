package com.msxi.mwise.autopay

import org.apache.spark.ml.feature.{WoEModel, WoETransformer}
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
class WoETransformerTest extends FlatSpec with Matchers with BeforeAndAfter {
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
    val woe = new WoETransformer().setInputCols(Array("feature")).setOutputColPostFix("woe")

    // When
    val model = woe.fit(df)

    // Then
    model should not be null
    val rows = model.transform(df).collect()
    assert(rows.length == 0)
  }

  it should " given 1 categorical, return a new dataframe with the WoE encoded column" in {
    // Given
    val df = sparkSession.createDataFrame(Seq(
      (true, "ab", 0.2827),
      (false, "ab", 0.2827),
      (true, "ab", 0.2827),
      (false, "cd", -0.4055),
      (true, "cd", -0.4055)
    )).toDF("label", "feature", "expected")
    val woe = new WoETransformer().setInputCols(Array("feature")).setOutputColPostFix("woe")

    // When
    val model = woe.fit(df)

    // Then
    checkModel(df, model, "feature_woe", "expected", "feature")
  }
  it should " given 2 categoricals, return a new dataframe with the WoE encoded columns" in {
    // Given
    val df = sparkSession.createDataFrame(Seq(
      (true, "ab", "XY", 0.2827, -0.4054),
      (false, "ab", "XY", 0.2827, -0.4054),
      (true, "ab", "YZ", 0.2827, 0.2827),
      (false, "cd", "YZ", -0.4055, 0.2827),
      (true, "cd", "YZ", -0.4055, 0.2827)
    )).toDF("label", "feature", "feature2", "expected", "expected2")
    val woe = new WoETransformer().setInputCols(Array("feature", "feature2")).setOutputColPostFix("woe")

    // When
    val model = woe.fit(df)

    // Then
    checkModel(df, model, "feature_woe", "expected", "feature")
    checkModel(df, model, "feature2_woe", "expected2", "feature2")

  }

  private def checkModel(df: DataFrame, model: WoEModel, actualCol: String, expectedCol: String, featureCol: String): Any = {
    model should not be null
    val epsilon = 1e-4f
    implicit val doubleEq = TolerantNumerics.tolerantDoubleEquality(epsilon)
    val rows = model.transform(df).collect()

    for (r <- rows) {
      val originalVal = r.getAs[String](featureCol)
      val actualVal = r.getAs[Double](actualCol)
      val expected = r.getAs[Double](expectedCol)
      withClue(s"$featureCol: $originalVal") {
        assert(actualVal === expected) // roughly equal to 4dp based on epsilon above to avoid floating point issues..
      }
      assert(WoETransformer.getInformationValue(df, featureCol, "label") === 0.1147)
    }
  }
}

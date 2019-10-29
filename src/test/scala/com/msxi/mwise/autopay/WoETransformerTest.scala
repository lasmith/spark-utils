package com.msxi.mwise.autopay

import org.apache.spark.ml.feature.WoETransformer
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SparkSession}
import org.scalactic.TolerantNumerics
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

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
    val woe = new WoETransformer().setInputCol("feature").setOutputCol("woe")

    // When
    val model = woe.fit(df)

    // Then
    model should not be null
    val rows = model.transform(df).collect()
    assert(rows.length == 0)
  }

  it should " return a new dataframe with the encoded column" in {
    // Given
    val df = sparkSession.createDataFrame(Seq(
      (true, "ab", 0.2827),
      (false, "ab", 0.2827),
      (true, "ab", 0.2827),
      (false, "cd", -0.4055),
      (true, "cd", -0.4055)
    )).toDF("label", "feature", "expected")
    val woe = new WoETransformer().setInputCol("feature").setOutputCol("woe")

    // When
    val model = woe.fit(df)

    // Then
    model should not be null
    val epsilon = 1e-4f
    implicit val doubleEq = TolerantNumerics.tolerantDoubleEquality(epsilon)
    assert(model.transform(df).collect().forall(r => {
      val woe = r.getAs[Double]("woe")
      val expected = r.getAs[Double]("expected")
      woe === expected // roughly equal to 2dp based on epsilon above to avoid floating point issues..
    }
    ))

    assert(WoETransformer.getInformationValue(df, "feature", "label") === 0.1147)
  }
}

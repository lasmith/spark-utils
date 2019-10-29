package com.msxi.mwise.autopay

import org.apache.spark.ml.feature.WoETransformer
import org.scalatest.{FlatSpec, Matchers}

class WoETransformerTest extends FlatSpec with Matchers{

  "Given a data frame, it " should " return a new dataframe with the encoded column" in {
    val sparkSession = SparkSessionFactory.createLocalSparkContect()
    val df = sparkSession.createDataFrame(Seq(
      (true, "ab", 0.2827),
      (false, "ab", 0.2827),
      (true, "ab", 0.2827),
      (false, "cd", -0.4055),
      (true, "cd", -0.4055)
    )).toDF("label", "feature", "expected")

    val woe = new WoETransformer().setInputCol("feature").setOutputCol("woe")
    val model = woe.fit(df)
    model should not be null
    val rows = model.transform(df).collect()
    // TODO: More asserts...
  }
}

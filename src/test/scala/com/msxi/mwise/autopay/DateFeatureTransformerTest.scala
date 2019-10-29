package com.msxi.mwise.autopay

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DateType, StringType, StructField, StructType}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

/**
 * <br> <br> 
 * Copyright:    Copyright (c) 2019 <br> 
 * Company:      MSX-International  <br>
 *
 * @author Laurence Smith
 */
class DateFeatureTransformerTest extends FlatSpec with Matchers with BeforeAndAfter {


  var sparkSession: SparkSession = _
  before {
    sparkSession = SparkSessionFactory.createLocalSparkContext()
  }


  behavior of "transform"

  it should " work on an empty dataset and do nothing" in {

  }

  val sampleFile = getClass.getClassLoader.getResource("sample_dates.csv")

  it should " work on an date column and generate the correct columns" in {

    // Given
    val schema = StructType(
      StructField("date_time", DateType, nullable = false) ::
        StructField("test_col", StringType, nullable = false) :: Nil
    )
    val df = sparkSession.read.format(source = "csv")
      .option("header", "true")
      .option("mode", "FAILFAST")
      .schema(schema)
      //.option("inferSchema", "true")
      .load(sampleFile.getFile)
    val trans = new DateFeatureTransformer()
      .setInputCol("date_time")
      .setOutputCol("dt_gen")

    // When
    val res = trans.transform(df)

    res should not be null
    val rows = df.collect()
    // TODO: Asserts on components.
  }

}

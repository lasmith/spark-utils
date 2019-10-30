package com.msxi.mwise.autopay

import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SparkSession}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

/**
 * <br> <br> 
 * Copyright:    Copyright (c) 2019 <br> 
 * Company:      MSX-International  <br>
 *
 * @author Laurence Smith
 */
class TimestampFeatureTransformerTest extends FlatSpec with Matchers with BeforeAndAfter {


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
      StructField("date_time", TimestampType, nullable = false) ::
        StructField("test_col", StringType, nullable = false) ::
        StructField("exp_date_time_year", IntegerType, nullable = false) ::
        StructField("exp_date_time_month", IntegerType, nullable = false) ::
        StructField("exp_date_time_day", IntegerType, nullable = false) ::
        StructField("exp_date_time_hour", IntegerType, nullable = false) ::
        StructField("exp_date_time_minute", IntegerType, nullable = false) ::
        Nil
    )
    val df = sparkSession.read.format(source = "csv")
      .option("header", "true")
      .option("mode", "FAILFAST")
      .schema(schema)
      //.option("inferSchema", "true")
      .load(sampleFile.getFile)
    val trans = new TimestampFeatureTransformer()
      .setInputCol("date_time")
      .setOutputCol("dt_gen")

    // When
    val res = trans.transform(df)

    res should not be null

    def checkActualVsExpected(r: Row, fieldName: String) = {
      val expected = r.getAs[Int](s"exp_$fieldName")
      val actual = r.getAs[Int](s"$fieldName")
      expected == actual
    }

    assert(res.collect().forall(r => {
      checkActualVsExpected(r, "date_time_year")
    }
    ))
    res.collect().forall(r => {
      assert(checkActualVsExpected(r, "date_time_month"))
      assert(checkActualVsExpected(r, "date_time_day"))
      assert(checkActualVsExpected(r, "date_time_hour"))
      assert(checkActualVsExpected(r, "date_time_minute"))
      true
    }
    )
  }

}

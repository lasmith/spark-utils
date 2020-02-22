package ai.mwise.spark

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
    val schema = getTestFileSchema
    val df = sparkSession.createDataFrame(sparkSession.sparkContext.emptyRDD[Row], schema)
    val trans = new TimestampFeatureTransformer()
      .setInputCols(Array("date_time"))
    val res = trans.transform(df)
    res should not be null
    val rows = res.collect()
    assert(rows.length == 0)
  }
  it should " work on an date column and generate the correct columns" in {
    // Given
    val df = loadTestData
    val trans = new TimestampFeatureTransformer()
      .setInputCols(Array("date_time"))

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
  it should "throw an exception with a column that is not a date" in {
    // Given
    val df = loadTestData
    val trans = new TimestampFeatureTransformer()
      .setInputCols(Array("test_col"))
    // When / Then
    intercept[IllegalArgumentException] {
      trans.transform(df)
    }
  }

  behavior of "generateStatement"
  it should " accept one column" in {
    // Given
    val tf = new TimestampFeatureTransformer()
    // When
    val sql = tf.generateStatement(Array("foo"))
    // Then
    sql should not be null
    sql shouldEqual ("SELECT *,  " +
      "year(foo) AS foo_year, month(foo) AS foo_month, dayofmonth(foo) AS foo_day, hour(foo) as foo_hour, minute(foo) as foo_minute " +
      "FROM __THIS__")
  }
  behavior of "generateStatement"
  it should " accept two columns" in {
    // Given
    val tf = new TimestampFeatureTransformer()
    // When
    val sql = tf.generateStatement(Array("foo", "bar"))
    // Then
    sql should not be null
    sql shouldEqual ("SELECT *,  " +
      "year(foo) AS foo_year, month(foo) AS foo_month, dayofmonth(foo) AS foo_day, hour(foo) as foo_hour, minute(foo) as foo_minute,   " +
      "year(bar) AS bar_year, month(bar) AS bar_month, dayofmonth(bar) AS bar_day, hour(bar) as bar_hour, minute(bar) as bar_minute " +
      "FROM __THIS__")
  }

  /**
   * Utility func to load the sample dates file with the correct schema
   */
  private def loadTestData = {
    val sampleFile = getClass.getClassLoader.getResource("sample_dates.csv")
    val schema = getTestFileSchema
    sparkSession.read.format(source = "csv")
      .option("header", "true")
      .option("mode", "FAILFAST")
      .schema(schema)
      //.option("inferSchema", "true")
      .load(sampleFile.getFile)
  }

  private def getTestFileSchema = {
    StructType(
      StructField("date_time", TimestampType, nullable = false) ::
        StructField("test_col", StringType, nullable = false) ::
        StructField("exp_date_time_year", IntegerType, nullable = false) ::
        StructField("exp_date_time_month", IntegerType, nullable = false) ::
        StructField("exp_date_time_day", IntegerType, nullable = false) ::
        StructField("exp_date_time_hour", IntegerType, nullable = false) ::
        StructField("exp_date_time_minute", IntegerType, nullable = false) ::
        Nil
    )
  }
}

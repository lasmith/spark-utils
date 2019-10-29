package com.msxi.mwise.autopay

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.feature.SQLTransformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

/**
 * A transformer to handle splitting dates into their component parts (year / month etc). Most of
 * the code comes from the SQLTransformer withing the spark source.
 *
 * TODO: Handle multi column, add options for which components to extract (eg date / time)
 *
 * @author Laurence Smith
 */
class DateFeatureTransformer(override val uid: String) extends Transformer {

  def this() = this(Identifiable.randomUID("dateFeature"))

  /**
   * Param for input column name.
   */
  final val inputCol: Param[String] = new Param[String](this, "inputCol", "input column name")

  final def getInputCol: String = $(inputCol)

  final def setInputCol(value: String): DateFeatureTransformer = {
    set(inputCol, value)
    val statement =
      s"""SELECT *,
         | year(${value}) AS ${value}_year,
         | month(${value}) AS ${value}_month,
         | dayofmonth(${value}) AS ${value}_day,
         | hour(date_time) as ${value}_hour,
         | minute(date_time) as ${value}_minute
         | FROM __THIS__""".stripMargin.filter(_ != '\n')
    setStatement(statement)
  }

  /**
   * Param for output column name.
   */
  final val outputCol: Param[String] = new Param[String](this, "outputCol", "output column name")

  final def getOutputCol: String = $(outputCol)

  final def setOutputCol(value: String): DateFeatureTransformer = set(outputCol, value)

  final val statement: Param[String] = new Param[String](this, "statement", "SQL statement")


  private def setStatement(value: String): this.type = set(statement, value)

  private val tableIdentifier: String = "__THIS__"

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val tableName = Identifiable.randomUID(uid)
    dataset.createOrReplaceTempView(tableName)
    val realStatement = $(statement).replace(tableIdentifier, tableName)
    val result = dataset.sparkSession.sql(realStatement)
    // Call SessionCatalog.dropTempView to avoid unpersisting the possibly cached dataset.
    dataset.sparkSession.sessionState.catalog.dropTempView(tableName)
    result
  }

  override def transformSchema(schema: StructType): StructType = {
    val spark = SparkSession.builder().getOrCreate()
    val dummyRDD = spark.sparkContext.parallelize(Seq(Row.empty))
    val dummyDF = spark.createDataFrame(dummyRDD, schema)
    val tableName = Identifiable.randomUID(uid)
    val realStatement = $(statement).replace(tableIdentifier, tableName)
    dummyDF.createOrReplaceTempView(tableName)
    val outputSchema = spark.sql(realStatement).schema
    spark.catalog.dropTempView(tableName)
    outputSchema
  }

  override def copy(extra: ParamMap): SQLTransformer = defaultCopy(extra)
}

object DateFeatureTransformer extends DefaultParamsReadable[DateFeatureTransformer] {

  override def load(path: String): DateFeatureTransformer = super.load(path)
}

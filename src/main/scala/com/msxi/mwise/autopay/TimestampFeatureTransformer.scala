package com.msxi.mwise.autopay

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.feature.SQLTransformer
import org.apache.spark.ml.param.shared.HasInputCols
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.types.{DataTypes, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

/**
 * A transformer to handle splitting dates into their component parts (year / month etc). Most of
 * the code comes from the SQLTransformer withing the spark source.
 *
 * TODO: add options for which components to extract (eg date / time)
 *
 * @author Laurence Smith
 */
class TimestampFeatureTransformer(override val uid: String) extends Transformer with HasInputCols {

  def this() = this(Identifiable.randomUID("dateFeature"))

  /** @group setParam */
  def setInputCols(values: Array[String]): this.type = {
    set(inputCols, values)
    setStatement(generateStatement(values))
  }

  final def generateStatement(values: Array[String]): String = {
    val strings: Array[String] = values.map(value =>
      s""" year(${value}) AS ${value}_year,
         | month(${value}) AS ${value}_month,
         | dayofmonth(${value}) AS ${value}_day,
         | hour(${value}) as ${value}_hour,
         | minute(${value}) as ${value}_minute, """)
    s"""SELECT *, ${strings.mkString(" ").dropRight(2)} FROM __THIS__""".stripMargin.filter(_ != '\n')
  }

  final val statement: Param[String] = new Param[String](this, "statement", "SQL statement")


  private def setStatement(value: String): this.type = set(statement, value)

  private val tableIdentifier: String = "__THIS__"

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = false)
    val tableName = Identifiable.randomUID(uid)
    dataset.createOrReplaceTempView(tableName)
    val realStatement = $(statement).replace(tableIdentifier, tableName)
    val result = dataset.sparkSession.sql(realStatement)
    // Call SessionCatalog.dropTempView to avoid unpersisting the possibly cached dataset.
    dataset.sparkSession.sessionState.catalog.dropTempView(tableName)
    result
  }

  override def transformSchema(schema: StructType): StructType = {
    $(inputCols).toSeq.foreach(inputCol => {
      val actualDataType = schema(inputCol).dataType
      require(actualDataType.equals(DataTypes.TimestampType),
        s"Column ${inputCol} must be TimestampType but was actually $actualDataType.")
    })

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

object TimestampFeatureTransformer extends DefaultParamsReadable[TimestampFeatureTransformer] {

  override def load(path: String): TimestampFeatureTransformer = super.load(path)
}

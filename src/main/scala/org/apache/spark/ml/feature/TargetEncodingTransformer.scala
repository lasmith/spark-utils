package org.apache.spark.ml.feature

import org.apache.hadoop.fs.Path
import org.apache.spark.annotation.{Experimental, Since}
import org.apache.spark.ml.feature.TargetEncodingModel.TargetEncodingModelWriter
import org.apache.spark.ml.param.shared.{HasInputCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

private[feature] trait TargetEncodingParams
  extends Params with HasInputCol with HasLabelCol with HasOutputCol {

  /** Validates and transforms the input schema. */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    require(isDefined(inputCol),
      s"TargetEncodingTransformer requires input column parameter: $inputCol")
    require(isDefined(outputCol),
      s"TargetEncodingTransformer requires output column parameter: $outputCol")
    require(isDefined(labelCol),
      s"TargetEncodingTransformer requires output column parameter: $labelCol")

    val outputColName = $(outputCol)
    if (schema.fieldNames.contains(outputColName)) {
      throw new IllegalArgumentException(s"Output column $outputColName already exists.")
    }
    StructType(schema.fields :+ StructField(outputColName, DoubleType, nullable = true))
  }
}

/**
 * Target (or mean) encoding takes for each category the mean of the target. This can optionally be smoothed
 * which helps reduce leakage of the target into the training data. By default smoothing is enabled.
 *
 * See here: https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02
 */
@Experimental
@Since("2.4.0")
class TargetEncodingTransformer(override val uid: String)
  extends Estimator[TargetEncodingModel] with TargetEncodingParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("TargetEncoding"))

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /**
   * Set the column name which is used as binary classes label column. The data type can be
   * boolean or numeric, which has at most two distinct values.
   *
   * @group setParam */
  def setLabelCol(value: String): this.type = set(labelCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def fit(dataset: Dataset[_]): TargetEncodingModel = {
    transformSchema(dataset.schema, logging = true)
    val table = TargetEncodingTransformer.getTargetEncodingTable(dataset, $(inputCol), $(labelCol))
    copyValues(new TargetEncodingModel(uid, table).setParent(this))
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): TargetEncodingTransformer = defaultCopy(extra)
}

@Experimental
@Since("2.4.0")
object TargetEncodingTransformer {

  private def getTargetEncodingTable(dataset: Dataset[_], categoryCol: String, labelCol: String): DataFrame = {
    val data = dataset.select(categoryCol, labelCol)
    val tmpTableName = "TargetEncoding_temp"
    data.createOrReplaceTempView(tmpTableName)
    val err = 0.01
    val query =
      s"""
         |SELECT
         |$categoryCol,
         |SUM (IF(CAST ($labelCol AS DOUBLE)=1, 1, 0)) AS 1count,
         |SUM (IF(CAST ($labelCol AS DOUBLE)=0, 1, 0)) AS 0count
         |FROM $tmpTableName
         |GROUP BY $categoryCol
        """.stripMargin
    val groupResult = data.sqlContext.sql(query)

    val total0 = groupResult.selectExpr("SUM(0count)").first().getAs[Long](0).toDouble
    val total1 = groupResult.selectExpr("SUM(1count)").first().getAs[Long](0).toDouble
    val mean = total1 / (total0 + total1)
    // TODO: Add smoothing
    groupResult.selectExpr(
      categoryCol,
      s"1count / (1count + 0count) AS te"
    )
  }
}

@Experimental
@Since("2.4.0")
class TargetEncodingModel private[ml](override val uid: String,
                                      val table: DataFrame)
  extends Model[TargetEncodingModel] with TargetEncodingParams with MLWritable {

  override def transform(dataset: Dataset[_]): DataFrame = {
    // validateParams()

    val encMap = table.rdd.map(r => {
      val category = r.get(0)
      val encVal = r.getAs[Double]("te")
      (category, encVal)
    }).collectAsMap()

    val trans = udf { (factor: String, target: Any) => encMap.get(factor) }
    dataset.withColumn($(outputCol), trans(col($(inputCol)), col($(labelCol))))
  }

  @Since("2.4.0")
  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  @Since("2.4.0")
  override def copy(extra: ParamMap): TargetEncodingModel = {
    val copied = new TargetEncodingModel(uid, table)
    copyValues(copied, extra).setParent(parent)
  }

  @Since("2.4.0")
  override def write: MLWriter = new TargetEncodingModelWriter(this)
}


@Since("2.4.0")
object TargetEncodingModel extends MLReadable[TargetEncodingModel] {

  private[TargetEncodingModel]
  class TargetEncodingModelWriter(instance: TargetEncodingModel) extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val dataPath = new Path(path, "data").toString
      instance.table.repartition(1).write.parquet(dataPath)
    }
  }

  private class TargetEncodingModelReader extends MLReader[TargetEncodingModel] {

    private val className = classOf[TargetEncodingModel].getName

    override def load(path: String): TargetEncodingModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val dataPath = new Path(path, "data").toString
      val data = sqlContext.read.parquet(dataPath)
      val model = new TargetEncodingModel(metadata.uid, data)
      metadata.getAndSetParams(model)
      model
    }
  }

  @Since("2.4.0")
  override def read: MLReader[TargetEncodingModel] = new TargetEncodingModelReader

  @Since("2.4.0")
  override def load(path: String): TargetEncodingModel = super.load(path)
}
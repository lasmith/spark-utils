// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package org.apache.spark.ml.feature

import org.apache.hadoop.fs.Path
import org.apache.spark.annotation.{Experimental, Since}
import org.apache.spark.ml.feature.TargetEncodingModel.TargetEncodingModelWriter
import org.apache.spark.ml.param.shared.{HasInputCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.util.Try

private[feature] trait TargetEncodingParams
  extends Params with HasInputCol with HasLabelCol with HasOutputCol {

  /** Validates and transforms the input schema. */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    require(isDefined(inputCol),
      s"TargetEncodingTransformer requires input column parameter: $inputCol")
    require(isDefined(outputCol),
      s"TargetEncodingTransformer requires output column parameter: $outputCol")
    require(isDefined(labelCol),
      s"TargetEncodingTransformer requires label column parameter: $labelCol")

    val outputColName = $(outputCol)
    if (schema.fieldNames.contains(outputColName)) {
      throw new IllegalArgumentException(s"Output column $outputColName already exists.")
    }
    StructType(schema.fields :+ StructField(outputColName, DoubleType, nullable = true))
  }
}

/**
 * Target (or mean) encoding takes for each category the mean of the target. This can optionally be smoothed
 * which helps reduce leakage of the target into the training data. By default smoothing is enabled but can be turned
 * off via the parameter.
 *
 * The basic calculation is:
 * output_column = 1count / catNoOfRecords = catMean
 *
 * The smoothed version is:
 *   output_column = (catNoOfRecords * catMean + weight * globalMean) / (catNoOfRecords + weight)
 *
 * Where:
 *   catNoOfRecords = The number of records for the category
 *   1count = The number of positive records for the category (eg number of claims with savings, number of positive cancer records)
 *   catMean = the mean target value for a given category
 *   weight = An optional weight, defaults to 100. Used to control the impact of the global mean on the output value
 *   globalMean = The average positive records in the whole data set (ie global1count / globalNoOfRecords)
 *
 *
 * See here: https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02
 *
 */
@Experimental
@Since("2.4.0")
class TargetEncodingTransformer(override val uid: String)
  extends Estimator[TargetEncodingModel] with TargetEncodingParams with DefaultParamsWritable {

  def this() = {
    this(Identifiable.randomUID("TargetEncoding"))
    setSmoothingEnabled(true)
    setSmoothingWeight(100d)
  }

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

  final val smoothingEnabled: Param[String] = new Param[String](this, "smoothingEnabled", "Is smoothing enabled")
  def setSmoothingEnabled(enabled: Boolean): this.type = set(smoothingEnabled, enabled.toString)

  final val smoothingWeight: Param[String] = new Param[String](this, "smoothingWeight", "The smoothing weight")
  def setSmoothingWeight(enabled: Double): this.type = set(smoothingWeight, enabled.toString)

  override def fit(dataset: Dataset[_]): TargetEncodingModel = {
    transformSchema(dataset.schema, logging = true)
    val table = TargetEncodingTransformer.getTargetEncodingTable(dataset, $(inputCol), $(labelCol),
                                                                 $(smoothingEnabled), $(smoothingWeight))
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

  private def getTargetEncodingTable(dataset: Dataset[_], categoryCol: String, labelCol: String,
                                     smoothingEnabled: String, smoothingWeight:String): DataFrame = {
    val data = dataset.select(categoryCol, labelCol)
    val tmpTableName = "TargetEncoding_temp"
    data.createOrReplaceTempView(tmpTableName)
    val err = 0.01
    val query =
      s"""
         |SELECT
         |$categoryCol,
         |SUM (IF(CAST ($labelCol AS DOUBLE)=1, 1, 0)) AS 1count,
         |SUM (IF(CAST ($labelCol AS DOUBLE)=0, 1, 0)) AS 0count,
         |count($labelCol) AS no_of_recs
         |FROM $tmpTableName
         |GROUP BY $categoryCol
        """.stripMargin
    val groupResult = data.sqlContext.sql(query)

    if (Try(smoothingEnabled.toBoolean).getOrElse(false)) {
      val globalMean = dataset.selectExpr(s"avg(CAST ($labelCol AS DOUBLE)) as mean").first().getAs[Double](0)
      val weight:Double = Try(smoothingWeight.toDouble).getOrElse(100d)
      groupResult.selectExpr(
        categoryCol,
//        s"no_of_recs  AS no_of_recs",
//        s"(no_of_recs + ${weight})  AS denom",
//        s"( no_of_recs * (1count / no_of_recs) + ${weight * globalMean}) AS numer",
        s"( no_of_recs * (1count / no_of_recs) + ${weight * globalMean}) / (no_of_recs + $weight)  AS te"
      )
    }
    else{
      groupResult.selectExpr(
        categoryCol,
        s"1count / no_of_recs AS te"
      )
    }
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
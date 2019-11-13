package com.msxi.mwise.autopay

import com.typesafe.scalalogging.Logger
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{Dataset, Row}

/**
 * Statistics calculator for binary classification
 * <br> <br> 
 * Copyright:    Copyright (c) 2019 <br> 
 * Company:      MSX-International  <br>
 *
 * @author Laurence Smith
 */
object StatsCalculator {
  val logger = Logger(ModelTrainer.getClass)

  /**
   * Calculate the classification statistics. This covers areas currently not supported by the BinaryClassificationMetrics from Spark.
   * Additional metrics calculated are:
   *
   * - F1 Score (not just the curve)
   * - Mathews Correlation Coefficient (MCC)
   * - Sensitivity / Specificity metrics (see [[https://en.wikipedia.org/wiki/Sensitivity_and_specificity here]])
   *
   * @param dataset           - The data set including target and prediction columns
   * @param predictionColName - The column storing the probability (or prediction)
   * @param targetColName     - The target column
   * @return Stats - The stats wrapper
   */
  def calculateStats(dataset: Dataset[_], predictionColName: String, targetColName: String): Stats = {
    val predictionsAndLabels =
      dataset.select(col(predictionColName), col(targetColName).cast(DoubleType)).rdd.map {
        case Row(prediction: Double, label: Double) => (prediction, label)
      }
    // Map reduce to calculate some metrics used later
    val (tp, tn, fp, fn) = predictionsAndLabels.aggregate((0, 0, 0, 0))(
      seqOp = (t, pal) => {
        val (tp, tn, fp, fn) = t
        (if (pal._1 == pal._2 && pal._2 == 1.0) tp + 1 else tp,
          if (pal._1 == pal._2 && pal._2 == 0.0) tn + 1 else tn,
          if (pal._1 == 1.0 && pal._2 == 0.0) fp + 1 else fp,
          if (pal._1 == 0.0 && pal._2 == 1.0) fn + 1 else fn)
      },
      combOp = (t1, t2) => (t1._1 + t2._1, t1._2 + t2._2, t1._3 + t2._3, t1._4 + t2._4)
    )
    new Stats(tp, tn, fp, fn)
  }

  /**
   *
   * @param dataset           - The data set including target and prediction columns
   * @param predictionColName - The column storing the probability (or prediction)
   * @param targetColName     - The target column
   * @param logLabel          - A label to add to the log messages (training / holdout etc)
   * @return Stats - The stats wrapper
   */
  def generateAndPrintStats(dataset: Dataset[_], predictionColName: String, targetColName: String, logLabel: String): Stats = {
    val stats = calculateStats(dataset, predictionColName, targetColName)
    logger.info(
      s"""| $logLabel Metrics:
          | $logLabel Accuracy ${stats.accuracy}
          | $logLabel Recall ${stats.recall}
          | $logLabel F1 ${stats.F1}
          | $logLabel MCC ${stats.MCC}
          |""".stripMargin)
    stats
  }


  /**
   * Calculate the metrics using the spark API methods.
   *
   * Whilst not complete this will calculate the PR curves amongst other things. Which can then be plotted into a graph.
   *
   * @param dataset           - The data set including target and prediction columns
   * @param predictionColName - The column storing the probability (or prediction)
   * @param targetColName     - The target column
   * @param logLabel          - A label to add to the log messages (training / holdout etc)
   * @param threshold         - The cut off point
   * @return BinaryClassificationMetrics - The calculated metrics
   */
  def calculateClassificationMetricsSpark(dataset: Dataset[_], predictionColName: String, targetColName: String, logLabel: String,
                                          threshold: Double = 0.5): BinaryClassificationMetrics = {
    val scoreAndLabels =
      dataset.select(col(predictionColName), col(targetColName).cast(DoubleType)).rdd.map {
        case Row(prediction: org.apache.spark.ml.linalg.Vector, label: Double) => (prediction(1), label)
        case Row(prediction: Double, label: Double) => (prediction, label)
      }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    logger.info(s"$logLabel metrics (with threshold $threshold):")
    logger.info(s"$logLabel AUROC: ${metrics.areaUnderROC()}")
    logger.info(s"$logLabel AUPRC: ${metrics.areaUnderPR()}")

    // This approach does not work as there is not always a 0.5 threshold. So ideally this would work out the correct threshold and use that
    // The alternative is to use the custom method above...
    val precision = metrics.precisionByThreshold.collect().find(x => x._1 == threshold)
    logger.info(s"$logLabel Precision: ${precision.get._2}")

    val recall = metrics.recallByThreshold.collect().find(x => x._1 == threshold)
    logger.info(s"$logLabel Recall: ${recall.get._2}")

    val f1Score = metrics.fMeasureByThreshold().collect().find(x => x._1 == threshold)
    logger.info(s"$logLabel F-score: ${f1Score.get._2}, Beta = 1")

    metrics
  }

}

class Stats(val tp: Int, val tn: Int, val fp: Int, val fn: Int) {
  val TPR = tp / (tp + fn).toDouble
  val recall = TPR
  val sensitivity = TPR
  val TNR = tn / (tn + fp).toDouble
  val specificity = TNR
  val PPV = tp / (tp + fp).toDouble
  val precision = PPV
  val NPV = tn / (tn + fn).toDouble
  val FPR = 1.0 - specificity
  val FNR = 1.0 - recall
  val FDR = 1.0 - precision
  val ACC = (tp + tn) / (tp + fp + fn + tn).toDouble
  val accuracy = ACC
  val F1 = 2 * PPV * TPR / (PPV + TPR).toDouble
  val MCC = (tp * tn - fp * fn).toDouble / math.sqrt((tp + fp).toDouble * (tp + fn).toDouble * (fp + tn).toDouble * (tn + fn).toDouble)

}
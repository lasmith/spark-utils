package com.msxi.mwise.autopay

import org.apache.spark.sql.SparkSession
import org.scalactic.TolerantNumerics
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

/**
 * <br> <br> 
 * Copyright:    Copyright (c) 2019 <br> 
 * Company:      MSX-International  <br>
 *
 * @author Laurence Smith
 */
class StatsCalculatorTest extends FlatSpec with Matchers with BeforeAndAfter {
  var sparkSession: SparkSession = _
  val label = "label"
  val target = "target"

  before {
    sparkSession = SparkSessionFactory.createLocalSparkContext()
  }


  // --------------------------------------------------------------------------------
  behavior of "calculateStats"
  it should "should work on 1tp" in {

    // Given
    val df = sparkSession.createDataFrame(Seq(
      (1d, 1d)
    )).toDF(label, target)
    // When
    val stats = StatsCalculator.calculateStats(df, label, target)
    // Then
    validateStats(stats, 1d, 0d, 0d, 0d, 1d, 1d, 1d, Double.NaN, 1d, 1d, Double.NaN)
  }

  it should "should work on 1tn" in {
    // Given
    val df = sparkSession.createDataFrame(Seq(
      (0d, 0d)
    )).toDF(label, target)
    // When
    val stats = StatsCalculator.calculateStats(df, label, target)
    // Then
    validateStats(stats, 0d, 1d, 0d, 0d, Double.NaN, Double.NaN, Double.NaN, 1d, 1d, Double.NaN, Double.NaN)
  }

  it should "should work on all 1tn + 1tp" in {
    // Given
    val df = sparkSession.createDataFrame(Seq(
      (0d, 0d),
      (1d, 1d)
    )).toDF(label, target)
    // When
    val stats = StatsCalculator.calculateStats(df, label, target)
    // Then
    validateStats(stats, 1d, 1d, 0d, 0d, 1d, 1d, 1d, 1d, 1d, 1d, 1d)
  }

  it should "should work on all 1tn + 1tp + 1fp" in {
    // Given
    val df = sparkSession.createDataFrame(Seq(
      (0d, 0d),
      (1d, 0d),
      (1d, 1d)
    )).toDF(label, target)
    // When
    val stats = StatsCalculator.calculateStats(df, label, target)
    // Then
    validateStats(stats, 1d, 1d, 1d, 0d, 1d, 1d, 1d, 0.5d, 0.6666d, 0.6666d, 0.5d)
  }

  it should "should work on all 1tn + 1tp + 1fp + 1fn" in {
    // Given
    val df = sparkSession.createDataFrame(Seq(
      (0d, 0d),
      (0d, 1d),
      (1d, 0d),
      (1d, 1d)
    )).toDF(label, target)
    // When
    val stats = StatsCalculator.calculateStats(df, label, target)
    // Then
    validateStats(stats, 1d, 1d, 1d, 1d, 0.5d, 0.5d, 0.5d, 0.5d, 0.5d, 0.5d, 0d)
  }


  // TODO: Additional tests


  private def validateStats(stats: Stats, tp: Double, tn: Double, fp: Double, fn: Double, tpr: Double, recall: Double,
                            sensitivity: Double, specificity: Double, accuracy: Double, f1: Double, mcc: Double): Unit = {
    def checkNumberOrNan(actual: Double, expected: Double, clue: String) = {
      val epsilon = 1e-4f
      implicit val doubleEq = TolerantNumerics.tolerantDoubleEquality(epsilon)
      withClue(clue) {
        if (expected.isNaN) {
          assert(actual.isNaN)
        }
        else {
          assert(actual === expected)
        }
      }
    }

    stats should not be (null)
    checkNumberOrNan(stats.tp, tp, "tp")
    checkNumberOrNan(stats.tn, tn, "tn")
    checkNumberOrNan(stats.fp, fp, "fp")
    checkNumberOrNan(stats.fn, fn, "fn")
    checkNumberOrNan(stats.TPR, tpr, "tpr")
    checkNumberOrNan(stats.recall, recall, "recall")
    checkNumberOrNan(stats.sensitivity, sensitivity, "sensitivity")
    checkNumberOrNan(stats.specificity, specificity, "specificity")
    checkNumberOrNan(stats.accuracy, accuracy, "accuracy")
    checkNumberOrNan(stats.F1, f1, "f1")
    checkNumberOrNan(stats.MCC, mcc, "mcc")
  }
}

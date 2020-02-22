package ai.mwise.spark

import org.apache.spark.sql.SparkSession
import org.slf4j.LoggerFactory

/**
 * <br> <br>
 * Copyright:    Copyright (c) 2019 <br>
 * Company:      MSX-International  <br>
 *
 * @author Laurence Smith
 */
object SparkSessionFactory {

  private val logger = LoggerFactory.getLogger(this.getClass)
  var sparkSession: SparkSession = null

  /**
   * @return SparkContext - a local spark context
   */
  def createLocalSparkContext() = {
    sparkSession match {
      case null =>
        logger.info("Creating spark session...")
        sparkSession = SparkSession.builder
          .appName("MWISE-LocalSparkContext")
          .master("local[*]")
          .getOrCreate()
        logger.info("Spark session created!")
        sparkSession
      case _ => sparkSession
    }
  }

  def close(): Unit = if (sparkSession != null) {
    sparkSession.close();
    sparkSession.stop();
    sparkSession = null
  }
}

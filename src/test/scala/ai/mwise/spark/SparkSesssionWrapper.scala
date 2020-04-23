package ai.mwise.spark

import ai.mwise.spark.SparkSessionFactory.{logger, sparkSession}
import org.apache.spark.sql.SparkSession
import org.slf4j.LoggerFactory

trait SparkSessionTestWrapper {

  private val logger = LoggerFactory.getLogger(this.getClass)

  lazy val sparkSession: SparkSession = {
    logger.info("Creating spark session...")
    val ss = SparkSession.builder
      .appName("MWISE-LocalSparkContext")
      .master("local[*]")
      .config("spark.sql.shuffle.partitions", "1")
      .getOrCreate()
    logger.info("Spark session created!")
    ss
  }

}

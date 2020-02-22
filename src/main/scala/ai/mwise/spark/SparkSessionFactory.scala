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
package ai.mwise.spark

import org.apache.spark.sql.SparkSession
import org.slf4j.LoggerFactory

/**
 * @author Laurence Smith
 */
object SparkSessionFactory {

  private val logger = LoggerFactory.getLogger(this.getClass)
  var sparkSession: SparkSession = _

  /**
   * @return SparkContext - a local spark context
   */
  def createLocalSparkContext(): SparkSession = {
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
    sparkSession.close()
    sparkSession.stop()
    sparkSession = null
  }
}

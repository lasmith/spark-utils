package ai.mwise.spark

import java.io.File

/**
  *
  * @author Laurence Smith
  */
object TestUtils {

  def getFileAsString(fileName: String): String = {
    val stream = getClass.getClassLoader.getResourceAsStream(fileName)
    val source = scala.io.Source.fromInputStream(stream)
    try source.mkString finally source.close()
  }

  def getTestResourceDir: File = {
    val sampleUrl = getClass.getClassLoader.getResource("sample_claims.csv")
    new File(sampleUrl.getFile).getParentFile
  }
}

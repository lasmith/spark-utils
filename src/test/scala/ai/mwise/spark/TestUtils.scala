package ai.mwise.spark

import java.io.File

/**
  * <br> <br> 
  * Copyright:    Copyright (c) 2019 <br> 
  * Company:      MSX-International  <br>
  *
  * @author Laurence Smith
  */
object TestUtils {

  implicit def getFileAsString(fileName: String) = {
    val stream = getClass.getClassLoader.getResourceAsStream(fileName)
    val source = scala.io.Source.fromInputStream(stream)
    try source.mkString finally source.close()
  }

  def getTestResourceDir: File = {
    val sampleUrl = getClass.getClassLoader.getResource("sample_claims.csv")
    new File(sampleUrl.getFile).getParentFile
  }
}

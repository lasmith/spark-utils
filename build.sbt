name := "spark-utils"
version := "1.2.0"
scalaVersion := "2.12.4"

organization := "ai.mwise.spark"

artifactName := { (sv: ScalaVersion, module: ModuleID, artifact: Artifact) =>
  artifact.name + "." + artifact.extension
}
parallelExecution in Test := false

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.4.3",
  "org.apache.spark" %% "spark-sql" % "2.4.3",
  "org.apache.spark" %% "spark-mllib" % "2.4.3",
  "org.apache.spark" %% "spark-streaming" % "2.4.3",
  "org.apache.spark" %% "spark-catalyst" % "2.4.3",
  "org.jmockit" % "jmockit" % "1.34" % "test",
  "com.github.scopt" %% "scopt" % "3.5.0",
  "com.typesafe.scala-logging" %% "scala-logging" % "3.9.2",
  "ch.qos.logback" % "logback-classic" % "1.2.3",

  // Test Resources
  "org.scalatest" %% "scalatest" % "3.0.8" % "test",
  "org.jmockit" % "jmockit" % "1.34" % "test"
)
name := "com/msxi/mwise/autopay"
version := "1.1.0"
scalaVersion := "2.12.4"

organization := "com.msxi.mwise"

resolvers += "Azure repository" at "https://pkgs.dev.azure.com/msxi-berlin/_packaging/mwise/maven/v1"

credentials += Credentials(new File(sys.env.getOrElse("SBT_CREDENTIALS", sys.props.get("user.home").get + "/.sbt/.credentials")))

artifactName := { (sv: ScalaVersion, module: ModuleID, artifact: Artifact) =>
  artifact.name + "." + artifact.extension
}

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
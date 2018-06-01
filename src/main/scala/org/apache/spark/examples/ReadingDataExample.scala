package org.apache.spark.examples

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession

object ReadingDataExample {

  def main(args: Array[String]) {

    //		println(util.Properties.versionString)

    val traditionalDataset = getClass.getClassLoader.getResource("splice.arff").getPath
    val multilabelDataset = getClass.getClassLoader.getResource("emotions.arff").getPath
    val multilabelXML = getClass.getClassLoader.getResource("emotions.xml").getPath
    val multitargetDataset = getClass.getClassLoader.getResource("andro.arff").getPath
    val multiInstanceDataset = getClass.getClassLoader.getResource("musk1.arff").getPath

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val log = Logger.getLogger(getClass.getName)

    val sparkSession = SparkSession.builder
      .appName("ReadingDataExample")
      .master("local[2]")
      .getOrCreate()

    log.info("Loading the dataset with traditional arff format")
    val traditional_df = sparkSession
      .read
      .format("org.apache.spark.ml.source.arff")
      .load(traditionalDataset)

    traditional_df.printSchema()
    traditional_df.show()

    log.info("Loading the dataset with multilabel arff format")
    val multilabel_df = sparkSession
      .read
      .format("org.apache.spark.ml.source.arff")
      .option("xmlMultilabelFile", multilabelXML)
      .load(multilabelDataset)

    multilabel_df.printSchema()
    multilabel_df.show()

    log.info("Loading the dataset with multitarget arff format")
    val multitarget_df = sparkSession
      .read
      .format("org.apache.spark.ml.source.arff")
      .option("numOutputs", 6)
      .load(multitargetDataset)

    multitarget_df.printSchema()
    multitarget_df.show()

    log.info("Loading the multiInstance arff data")
    val multiInstance_df = sparkSession.read.format("org.apache.spark.ml.source.arff")
      .option("multiInstance", value = true)
      .load(multiInstanceDataset)

    multiInstance_df.printSchema()
    multiInstance_df.show()
  }

}
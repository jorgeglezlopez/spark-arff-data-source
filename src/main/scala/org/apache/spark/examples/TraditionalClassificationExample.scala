package org.apache.spark.examples

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.util.MetadataUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vector

object TraditionalClassificationExample {

  def main(args: Array[String]) {

    val traditionalClassificationDataset = getClass
      .getClassLoader
      .getResource("weather.arff")
      .getPath

    //    Optionally: turn off logger, so prints are more clear
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val log = Logger.getLogger(getClass.getName)

    val sparkSession = SparkSession.builder
      .appName("TraditionalClassificationExample")
      .master("local[*]")
      .getOrCreate()

    log.info("Loading the classification dataset with traditional ARFF format")
    val dataset = sparkSession
      .read
      .format("org.apache.spark.ml.source.arff")
      .load(traditionalClassificationDataset)

    log.info("Print schema")
    dataset.printSchema()
    //    root
    //    |-- label: double (nullable = true)
    //    |-- features: vector (nullable = true)

    log.info("Print schema metadata")
    dataset.schema.foreach(x => println(x.metadata))
    //    {"ml_attr":{"name":"play","vals":["yes","no"],"idx":0,"global_idx":[4],"type":"nominal"}}
    //    {"ml_attr":{"attrs":{"numeric":[
    //                            {"idx":1,"name":"temperature"},
    //                            {"idx":2,"name":"humidity"}],
    //                        "nominal":[
    //                            {"vals":["sunny","overcast","rainy"],"idx":0,"name":"outlook"},
    //                            {"vals":["TRUE","FALSE"],"idx":3,"name":"windy"}]
    //                        },"global_idx":[0,1,2,3],"num_attrs":4}}

    log.info("Show first 20 instances")
    dataset.show()
    //    +-----+-------------------+
    //    |label|           features|
    //    +-----+-------------------+
    //    |  1.0|[0.0,85.0,85.0,1.0]|
    //    |  1.0|[0.0,80.0,90.0,0.0]|
    //    |  0.0|[1.0,83.0,86.0,1.0]|
    //    |  0.0|[2.0,70.0,96.0,1.0]|
    //    |  0.0|[2.0,68.0,80.0,1.0]|
    //    |  1.0|[2.0,65.0,70.0,0.0]|
    //    |  0.0|[1.0,64.0,65.0,0.0]|
    //    |  1.0|[0.0,72.0,95.0,1.0]|
    //    |  0.0|[0.0,69.0,70.0,1.0]|
    //    |  0.0|[2.0,75.0,80.0,1.0]|
    //    |  0.0|[0.0,75.0,70.0,0.0]|
    //    |  0.0|[1.0,72.0,90.0,0.0]|
    //    |  0.0|[1.0,81.0,75.0,1.0]|
    //    |  1.0|[2.0,71.0,91.0,0.0]|
    //    +-----+-------------------+

    log.info("Extracts immediately the number of categories for each nominal attribute from the " +
      "schema.")
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema("features"))

    categoricalFeatures.foreach(println(_))
    //    (0,3)
    //    (3,2)

    log.info("Number of instances with each class")
    dataset.groupBy("label").count().show()
    //    +-----+-----+
    //    |label|count|
    //    +-----+-----+
    //    |  0.0|    9|
    //    |  1.0|    5|
    //    +-----+-----+

    log.info("Characteristics of the dataset")
    val numInstances = dataset.count()
    val numFeatures = dataset.first().getAs[Vector]("features").size
    val numClasses = MetadataUtils.getNumClasses(dataset.schema("label")).getOrElse(-1)

    //    If we didnt have the information in the metadata it would be..
    //    val numClasses = dataset.select("label").distinct().count()

    println(s"Dataset characteristics:\n instances:$numInstances features:$numFeatures " +
      s"classes:$numClasses")
    //    Dataset characteristics:
    //      instances:14 features:4 classes:2

  }

}

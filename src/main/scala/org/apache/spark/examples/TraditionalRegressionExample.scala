package org.apache.spark.examples

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.util.MetadataUtils
import org.apache.spark.sql.SparkSession

object TraditionalRegressionExample {

  def main(args: Array[String]): Unit = {

    val traditionalClassificationDataset = getClass
      .getClassLoader
      .getResource("servo.arff")
      .getPath

    //    Optionally: turn off logger, so prints are more clear
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val log = Logger.getLogger(getClass.getName)

    val sparkSession = SparkSession.builder()
      .appName("TraditionalRegressionExample")
      .master("local[*]")
      .getOrCreate()

    log.info("Loading the regression dataset with traditional ARFF format")
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
    //    {"ml_attr":{"global_idx":[4],"idx":0,"name":"class"}}
    //    {"ml_attr":{"attrs":{"nominal":[{"vals":["E","B","D","C","A"],"idx":0,"name":"motor"},
    //                                    {"vals":["E","D","A","B","C"],"idx":1,"name":"screw"},
    //                                    {"vals":["5","6","4","3"],"idx":2,"name":"pgain"},
    //                                    {"vals":["4","5","3","2","1"],"idx":3,"name":"vgain"}]},
    //     "global_idx":[0,1,2,3],"num_attrs":4}}


    log.info("Show first 20 instances")
    dataset.show()
    //    +--------+-----------------+
    //    |   label|         features|
    //    +--------+-----------------+
    //    |0.281251|        (4,[],[])|
    //    |0.506252|[1.0,1.0,1.0,1.0]|
    //    |0.356251|[2.0,1.0,2.0,2.0]|
    //    | 5.50003|[1.0,2.0,3.0,3.0]|
    //    |0.356251|[2.0,3.0,1.0,1.0]|
    //    |0.806255|[0.0,4.0,2.0,2.0]|
    //    | 5.10001|[3.0,2.0,3.0,3.0]|
    //    | 5.70004|[4.0,2.0,3.0,3.0]|
    //    |0.768754|[3.0,2.0,1.0,1.0]|
    //    | 1.03125|[2.0,2.0,2.0,4.0]|
    //    |0.468752|[1.0,0.0,1.0,1.0]|
    //    |0.393752|    (4,[1],[4.0])|
    //    |0.281251|[1.0,4.0,2.0,4.0]|
    //    |     1.1|[0.0,4.0,3.0,4.0]|
    //    |0.506252|[3.0,4.0,0.0,0.0]|
    //    | 1.89999|[0.0,3.0,3.0,3.0]|
    //    |0.900001|[2.0,4.0,3.0,4.0]|
    //    |0.468752|[1.0,4.0,0.0,0.0]|
    //    |0.543753|[1.0,3.0,0.0,0.0]|
    //    | 0.20625|[3.0,0.0,2.0,3.0]|
    //    +--------+-----------------+
    //    only showing top 20 rows

    log.info("Extracts immediately the number of categories for each nominal attribute from the " +
      "schema.")
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema("features"))

    categoricalFeatures.foreach(println(_))
    //    (0,5)
    //    (1,5)
    //    (2,4)
    //    (3,5)

    log.info("Compute statistics about the label")
    import org.apache.spark.sql.functions._
    dataset.agg(min("label").as("minimum"),
      max("label").as("maximum"),
      avg("label").as("average"),
      kurtosis("label").as("kurtosis"),
      skewness("label").as("skewness"))
      .show()
    //    +-------+-------+------------------+-----------------+-----------------+
    //    |minimum|maximum|           average|         kurtosis|         skewness|
    //    +-------+-------+------------------+-----------------+-----------------+
    //    |0.13125|7.10011|1.3897083592814365|1.975733284008018|1.774861952837843|
    //    +-------+-------+------------------+-----------------+-----------------+

    log.info("Characteristics of the dataset")
    val numInstances = dataset.count()
    val numFeatures = dataset.first().getAs[Vector]("features").size
    println(s"Dataset characteristics:\n instances:$numInstances features:$numFeatures " +
      s"classes:1")
    //    Dataset characteristics:
    //      instances:167 features:4 classes:1

  }

}

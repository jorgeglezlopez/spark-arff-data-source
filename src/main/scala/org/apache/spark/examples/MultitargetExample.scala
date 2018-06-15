package org.apache.spark.examples

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.linalg.{Matrix, Vector}
import org.apache.spark.sql.{Row, SparkSession}

object MultitargetExample {

  def main(args: Array[String]) {

    val multitargetDataset = getClass
      .getClassLoader
      .getResource("andro.arff")
      .getPath

    //    Optionally: turn off logger, so prints are more clear
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val log = Logger.getLogger(getClass.getName)

    val sparkSession = SparkSession.builder
      .appName("MultitargetExample")
      .master("local[*]")
      .getOrCreate()

    log.info("Loading the dataset with multitarget ARFF format")
    val dataset = sparkSession
      .read
      .format("org.apache.spark.ml.source.arff")
      .option("numOutputs", 6)
      .load(multitargetDataset)

    log.info("Print schema")
    dataset.printSchema()
    //    root
    //    |-- multilabel: vector (nullable = true)
    //    |-- features: vector (nullable = true)

    log.info("Print schema metadata")
    dataset.schema.foreach(x => println(x.metadata))
    //    {"ml_attr":{"attrs":{"numeric":[{"idx":0,"name":"Target"},{"idx":1,"name":"Target_2"},{"idx":2,"name":"Target_3"},{"idx":3,"name":"Target_4"},{"idx":4,"name":"Target_5"},{"idx":5,"name":"Target_6"}]},"global_idx":[30,31,32,33,34,35],"num_attrs":6}}
    //    {"ml_attr":{"attrs":{"numeric":[{"idx":0,"name":"Window0-Att0"},
    //                                    {"idx":1,"name":"Window0-Att1"},
    //                                    {"idx":2,"name":"Window0-Att2"},
    //                                    {"idx":3,"name":"Window0-Att3"},
    //                                    {"idx":4,"name":"Window0-Att4"},
    //                                    {"idx":5,"name":"Window0-Att5"},
    //                                    {"idx":6,"name":"Window1-Att0"},
    //                                    {"idx":7,"name":"Window1-Att1"},
    //                                    {"idx":8,"name":"Window1-Att2"},
    //                                    {"idx":9,"name":"Window1-Att3"},
    //                                    {"idx":10,"name":"Window1-Att4"},
    //                                    {"idx":11,"name":"Window1-Att5"},
    //                                    {"idx":12,"name":"Window2-Att0"},
    //                                    {"idx":13,"name":"Window2-Att1"},
    //                                    {"idx":14,"name":"Window2-Att2"},
    //                                    {"idx":15,"name":"Window2-Att3"},
    //                                    {"idx":16,"name":"Window2-Att4"},
    //                                    {"idx":17,"name":"Window2-Att5"},
    //                                    {"idx":18,"name":"Window3-Att0"},
    //                                    {"idx":19,"name":"Window3-Att1"},
    //                                    {"idx":20,"name":"Window3-Att2"},
    //                                    {"idx":21,"name":"Window3-Att3"},
    //                                    {"idx":22,"name":"Window3-Att4"},
    //                                    {"idx":23,"name":"Window3-Att5"},
    //                                    {"idx":24,"name":"Window4-Att0"},
    //                                    {"idx":25,"name":"Window4-Att1"},
    //                                    {"idx":26,"name":"Window4-Att2"},
    //                                    {"idx":27,"name":"Window4-Att3"},
    //                                    {"idx":28,"name":"Window4-Att4"},
    //                                    {"idx":29,"name":"Window4-Atta5"}]},"
    //     global_idx":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],"num_attrs":30}}

    log.info("Show first 20 instances")
    dataset.show()
    //    +--------------------+--------------------+
    //    |          multilabel|            features|
    //    +--------------------+--------------------+
    //    |(6,[0,1,2,3,4,5],...|[11.84,5.08,49.0,...|
    //    |(6,[0,1,2,3,4,5],...|[12.01,5.09,48.0,...|
    //    |(6,[0,1,2,3,4,5],...|[12.89,5.1,48.0,3...|
    //    |(6,[0,1,2,3,4,5],...|[13.95,5.11,47.0,...|
    //    |(6,[0,1,2,3,4,5],...|[13.75,5.13,45.0,...|
    //    |(6,[0,1,2,3,4,5],...|[13.34,5.13,46.0,...|
    //    |(6,[0,1,2,3,4,5],...|[13.63,5.11,48.0,...|
    //    |(6,[0,1,2,3,4,5],...|[14.27,5.12,47.0,...|
    //    |(6,[0,1,2,3,4,5],...|[14.19,5.15,45.0,...|
    //    |(6,[0,1,2,3,4,5],...|[14.75,5.15,45.0,...|
    //    |(6,[0,1,2,3,4,5],...|[15.17,5.14,46.0,...|
    //    |(6,[0,1,2,3,4,5],...|[14.85,5.11,47.0,...|
    //    |(6,[0,1,2,3,4,5],...|[14.72,5.1,48.0,3...|
    //    |(6,[0,1,2,3,4,5],...|[15.28,5.11,48.0,...|
    //    |(6,[0,1,2,3,4,5],...|[16.15,5.13,47.0,...|
    //    |(6,[0,1,2,3,4,5],...|[17.29,5.13,47.0,...|
    //    |(6,[0,1,2,3,4,5],...|[18.02,5.15,46.0,...|
    //    |(6,[0,1,2,3,4,5],...|[18.73,5.15,46.0,...|
    //    |(6,[0,1,2,3,4,5],...|[19.57,5.14,47.0,...|
    //    |(6,[0,1,2,3,4,5],...|[20.96,6.65,48.0,...|
    //    +--------------------+--------------------+
    //    only showing top 20 rows

    val numLabels = dataset.first().getAs[Vector]("multilabel")
      .size

    //      import org.apache.spark.sql.functions._
    import org.apache.spark.ml.stat.Correlation
    val Row(coeff: Matrix) = Correlation.corr(dataset, "multilabel").head
    val strCoeff = coeff.toString(600, 600)
    println(s"Pearson correlation matrix of labels:\n $strCoeff \n")
    //    Pearson correlation matrix of labels:
    //    1.0                  0.7778513566466512    -0.4888751661511569   -0.5233641345036824   0.285064208638145    0.1657287797789108
    //    0.7778513566466512   1.0                   -0.10579386374045373  -0.13914258558561302  0.5903471987660324   0.5084910594447184
    //    -0.4888751661511569  -0.10579386374045373  1.0                   0.9957962648852898    0.07719327642537346  0.10806043401478108
    //    -0.5233641345036824  -0.13914258558561302  0.9957962648852898    1.0                   0.08048627985599216  0.11641859854351705
    //    0.285064208638145    0.5903471987660324    0.07719327642537346   0.08048627985599216   1.0                  0.9906648622901055
    //    0.1657287797789108   0.5084910594447184    0.10806043401478108   0.11641859854351705   0.9906648622901055   1.0


    log.info("Characteristics of the dataset")
    val numInstances = dataset.count()
    val numFeatures = dataset.first().getAs[Vector]("features").size
    println(s"Dataset characteristics:\n instances:$numInstances features:$numFeatures " +
      s"targets:$numLabels")
    //    Dataset characteristics:
    //      instances:49 features:30 targets:6
  }
}

package org.apache.spark.examples

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.linalg.{SQLDataTypes, Vector, Vectors}
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types.{ArrayType, DataType, DoubleType, StructType}
import org.apache.spark.sql.{Row, SparkSession}

import scala.collection.mutable

object MultilabelExample {

  def main(args: Array[String]) {

    val multilabelDataset = getClass
      .getClassLoader
      .getResource("emotions.arff")
      .getPath
    val multilabelXML = getClass
      .getClassLoader
      .getResource("emotions.xml")
      .getPath

    //    Optionally: turn off logger, so prints are more clear
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val log = Logger.getLogger(getClass.getName)

    val sparkSession = SparkSession.builder
      .appName("MultilabelExample")
      .master("local[*]")
      .getOrCreate()

    log.info("Loading the classification dataset with multilabel ARFF format")
    val dataset = sparkSession
      .read
      .format("org.apache.spark.ml.source.arff")
      .option("xmlMultilabelFile", multilabelXML)
      .load(multilabelDataset)

    log.info("Print schema")
    dataset.printSchema()
    //    root
    //    |-- multilabel: vector (nullable = true)
    //    |-- features: vector (nullable = true)

    log.info("Print schema metadata")
    dataset.schema.foreach(x => println(x.metadata))
    //    {"ml_attr":{"attrs":{"nominal":[{"vals":["0","1"],"idx":0,"name":"amazed-suprised"},{"vals":["0","1"],"idx":1,"name":"happy-pleased"},{"vals":["0","1"],"idx":2,"name":"relaxing-calm"},{"vals":["0","1"],"idx":3,"name":"quiet-still"},{"vals":["0","1"],"idx":4,"name":"sad-lonely"},{"vals":["0","1"],"idx":5,"name":"angry-aggresive"}]},"global_idx":[72,73,74,75,76,77],"num_attrs":6}}
    //    {"ml_attr":{"attrs":{"numeric":[{"idx":0,"name":"Mean_Acc1298_Mean_Mem40_Centroid"},
    //        {"idx":1,"name":"Mean_Acc1298_Mean_Mem40_Rolloff"},
    //        {"idx":2,"name":"Mean_Acc1298_Mean_Mem40_Flux"},
    //        {"idx":3,"name":"Mean_Acc1298_Mean_Mem40_MFCC_0"},
    //        {"idx":4,"name":"Mean_Acc1298_Mean_Mem40_MFCC_1"},
    //        {"idx":5,"name":"Mean_Acc1298_Mean_Mem40_MFCC_2"},
    //        {"idx":6,"name":"Mean_Acc1298_Mean_Mem40_MFCC_3"},
    //        {"idx":7,"name":"Mean_Acc1298_Mean_Mem40_MFCC_4"},
    //        {"idx":8,"name":"Mean_Acc1298_Mean_Mem40_MFCC_5"},
    //        {"idx":9,"name":"Mean_Acc1298_Mean_Mem40_MFCC_6"},
    //        {"idx":10,"name":"Mean_Acc1298_Mean_Mem40_MFCC_7"},
    //        {"idx":11,"name":"Mean_Acc1298_Mean_Mem40_MFCC_8"},
    //        {"idx":12,"name":"Mean_Acc1298_Mean_Mem40_MFCC_9"},
    //        {"idx":13,"name":"Mean_Acc1298_Mean_Mem40_MFCC_10"},
    //        {"idx":14,"name":"Mean_Acc1298_Mean_Mem40_MFCC_11"},
    //        {"idx":15,"name":"Mean_Acc1298_Mean_Mem40_MFCC_12"},
    //        {"idx":16,"name":"Mean_Acc1298_Std_Mem40_Centroid"},
    //        {"idx":17,"name":"Mean_Acc1298_Std_Mem40_Rolloff"},
    //        {"idx":18,"name":"Mean_Acc1298_Std_Mem40_Flux"},
    //        {"idx":19,"name":"Mean_Acc1298_Std_Mem40_MFCC_0"},
    //        {"idx":20,"name":"Mean_Acc1298_Std_Mem40_MFCC_1"},
    //        {"idx":21,"name":"Mean_Acc1298_Std_Mem40_MFCC_2"},
    //        {"idx":22,"name":"Mean_Acc1298_Std_Mem40_MFCC_3"},
    //        {"idx":23,"name":"Mean_Acc1298_Std_Mem40_MFCC_4"},
    //        {"idx":24,"name":"Mean_Acc1298_Std_Mem40_MFCC_5"},
    //        {"idx":25,"name":"Mean_Acc1298_Std_Mem40_MFCC_6"},
    //        {"idx":26,"name":"Mean_Acc1298_Std_Mem40_MFCC_7"},
    //        {"idx":27,"name":"Mean_Acc1298_Std_Mem40_MFCC_8"},
    //        {"idx":28,"name":"Mean_Acc1298_Std_Mem40_MFCC_9"},
    //        {"idx":29,"name":"Mean_Acc1298_Std_Mem40_MFCC_10"},
    //        {"idx":30,"name":"Mean_Acc1298_Std_Mem40_MFCC_11"},
    //        {"idx":31,"name":"Mean_Acc1298_Std_Mem40_MFCC_12"},
    //        {"idx":32,"name":"Std_Acc1298_Mean_Mem40_Centroid"},
    //        {"idx":33,"name":"Std_Acc1298_Mean_Mem40_Rolloff"},
    //        {"idx":34,"name":"Std_Acc1298_Mean_Mem40_Flux"},
    //        {"idx":35,"name":"Std_Acc1298_Mean_Mem40_MFCC_0"},
    //        {"idx":36,"name":"Std_Acc1298_Mean_Mem40_MFCC_1"},
    //        {"idx":37,"name":"Std_Acc1298_Mean_Mem40_MFCC_2"},
    //        {"idx":38,"name":"Std_Acc1298_Mean_Mem40_MFCC_3"},
    //        {"idx":39,"name":"Std_Acc1298_Mean_Mem40_MFCC_4"},
    //        {"idx":40,"name":"Std_Acc1298_Mean_Mem40_MFCC_5"},
    //        {"idx":41,"name":"Std_Acc1298_Mean_Mem40_MFCC_6"},
    //        {"idx":42,"name":"Std_Acc1298_Mean_Mem40_MFCC_7"},
    //        {"idx":43,"name":"Std_Acc1298_Mean_Mem40_MFCC_8"},
    //        {"idx":44,"name":"Std_Acc1298_Mean_Mem40_MFCC_9"},
    //        {"idx":45,"name":"Std_Acc1298_Mean_Mem40_MFCC_10"},
    //        {"idx":46,"name":"Std_Acc1298_Mean_Mem40_MFCC_11"},
    //        {"idx":47,"name":"Std_Acc1298_Mean_Mem40_MFCC_12"},
    //        {"idx":48,"name":"Std_Acc1298_Std_Mem40_Centroid"},
    //        {"idx":49,"name":"Std_Acc1298_Std_Mem40_Rolloff"},
    //        {"idx":50,"name":"Std_Acc1298_Std_Mem40_Flux"},
    //        {"idx":51,"name":"Std_Acc1298_Std_Mem40_MFCC_0"},
    //        {"idx":52,"name":"Std_Acc1298_Std_Mem40_MFCC_1"},
    //        {"idx":53,"name":"Std_Acc1298_Std_Mem40_MFCC_2"},
    //        {"idx":54,"name":"Std_Acc1298_Std_Mem40_MFCC_3"},
    //        {"idx":55,"name":"Std_Acc1298_Std_Mem40_MFCC_4"},
    //        {"idx":56,"name":"Std_Acc1298_Std_Mem40_MFCC_5"},
    //        {"idx":57,"name":"Std_Acc1298_Std_Mem40_MFCC_6"},
    //        {"idx":58,"name":"Std_Acc1298_Std_Mem40_MFCC_7"},
    //        {"idx":59,"name":"Std_Acc1298_Std_Mem40_MFCC_8"},
    //        {"idx":60,"name":"Std_Acc1298_Std_Mem40_MFCC_9"},
    //        {"idx":61,"name":"Std_Acc1298_Std_Mem40_MFCC_10"},
    //        {"idx":62,"name":"Std_Acc1298_Std_Mem40_MFCC_11"},
    //        {"idx":63,"name":"Std_Acc1298_Std_Mem40_MFCC_12"},
    //        {"idx":64,"name":"BH_LowPeakAmp"},
    //        {"idx":65,"name":"BH_LowPeakBPM"},
    //        {"idx":66,"name":"BH_HighPeakAmp"},
    //        {"idx":67,"name":"BH_HighPeakBPM"},
    //        {"idx":68,"name":"BH_HighLowRatio"},
    //        {"idx":69,"name":"BHSUM1"},
    //        {"idx":70,"name":"BHSUM2"},
    //        {"idx":71,"name":"BHSUM3"}]},
    //    "global_idx":[0,1,2,...,71],"num_attrs":72}}


    log.info("Show first 20 instances")
    dataset.show()
    //    +--------------------+--------------------+
    //    |          multilabel|            features|
    //    +--------------------+--------------------+
    //    | (6,[1,2],[1.0,1.0])|[0.034741,0.08966...|
    //    | (6,[0,5],[1.0,1.0])|[0.081374,0.27274...|
    //    | (6,[1,5],[1.0,1.0])|[0.110545,0.27356...|
    //    |       (6,[2],[1.0])|[0.042481,0.19928...|
    //    |       (6,[3],[1.0])|[0.07455,0.14088,...|
    //    | (6,[1,2],[1.0,1.0])|[0.052434,0.11065...|
    //    | (6,[0,1],[1.0,1.0])|[0.064067,0.14737...|
    //    |       (6,[5],[1.0])|[0.044949,0.09208...|
    //    | (6,[0,1],[1.0,1.0])|[0.081354,0.30205...|
    //    |(6,[2,3,4],[1.0,1...|[0.039819,0.05698...|
    //    | (6,[1,2],[1.0,1.0])|[0.070779,0.24974...|
    //    |       (6,[2],[1.0])|[0.07661,0.173846...|
    //    |       (6,[0],[1.0])|[0.112665,0.3462,...|
    //    | (6,[2,4],[1.0,1.0])|[0.031987,0.06375...|
    //    |       (6,[4],[1.0])|[0.056384,0.11465...|
    //    | (6,[0,5],[1.0,1.0])|[0.081393,0.49057...|
    //    |(6,[2,3,4],[1.0,1...|[0.036562,0.07414...|
    //    | (6,[1,2],[1.0,1.0])|[0.073788,0.25040...|
    //    | (6,[4,5],[1.0,1.0])|[0.072026,0.19156...|
    //    |(6,[2,3,4],[1.0,1...|[0.030467,0.04433...|
    //    +--------------------+--------------------+
    //    only showing top 20 rows


    val numLabels = dataset.first().getAs[Vector]("multilabel")
      .size

    import org.apache.spark.sql.functions._
    log.info("Number of unique subsets of labels")
    val unique = dataset.groupBy("multilabel")
      .count()
      .sort(desc("count"))

    unique.show()
    //    +--------------------+-----+
    //    |          multilabel|count|
    //    +--------------------+-----+
    //    | (6,[0,5],[1.0,1.0])|   81|
    //    | (6,[1,2],[1.0,1.0])|   74|
    //    |       (6,[5],[1.0])|   72|
    //    |(6,[2,3,4],[1.0,1...|   67|
    //    |       (6,[2],[1.0])|   42|
    //    | (6,[0,1],[1.0,1.0])|   38|
    //    | (6,[3,4],[1.0,1.0])|   37|
    //    | (6,[2,3],[1.0,1.0])|   30|
    //    | (6,[2,4],[1.0,1.0])|   25|
    //    |       (6,[0],[1.0])|   24|
    //    |       (6,[1],[1.0])|   23|
    //    | (6,[4,5],[1.0,1.0])|   12|
    //    |       (6,[4],[1.0])|   12|
    //    |(6,[0,1,2],[1.0,1...|   11|
    //    |(6,[0,1,5],[1.0,1...|    7|
    //    |(6,[1,2,3],[1.0,1...|    6|
    //    | (6,[0,4],[1.0,1.0])|    6|
    //    |       (6,[3],[1.0])|    5|
    //    | (6,[1,5],[1.0,1.0])|    5|
    //    |(6,[0,4,5],[1.0,1...|    4|
    //    +--------------------+-----+
    //    only showing top 20 rows

    dataset.agg(
      new LabelFrequency(numLabels)(col("multilabel")).as("multilabelFreq"))
      .show(truncate = false)
    //    +-------------------------------------+
    //    |multilabelFreq                       |
    //    +-------------------------------------+
    //    |[173.0,166.0,264.0,148.0,168.0,189.0]|
    //    +-------------------------------------+


    val cardinalityUDF = udf { labels: Vector => labels.numActives }
    val densityUDF = udf { labels: Vector => labels.numActives.toDouble / labels.size }
    val labelStat = dataset
      .withColumn("cardinality", cardinalityUDF(col("multilabel")))
      .withColumn("density", densityUDF(col("multilabel")))
      .agg(avg("cardinality"), avg("density"))

    labelStat.show()
    //    +------------------+-------------------+
    //    |  avg(cardinality)|       avg(density)|
    //    +------------------+-------------------+
    //    |1.8684654300168635|0.31141090500281104|
    //    +------------------+-------------------+

    log.info("Characteristics of the dataset")
    val numInstances = dataset.count()
    val numFeatures = dataset.first().getAs[Vector]("features").size
    val cardinality = labelStat.first().getDouble(0)
    val density = labelStat.first().getDouble(1)
    val numUnique = unique.count()

    println(s"Dataset characteristics:\n instances:$numInstances features:$numFeatures " +
      s"labels:$numLabels cardinality:$cardinality density:$density unique:$numUnique")
    //    Dataset characteristics:
    //      instances:593 features:72 labels:6 cardinality:1.8684654300168635 density:0.31141090500281104 unique:27
  }
}

class LabelFrequency(size: Int) extends UserDefinedAggregateFunction {

  def inputSchema: StructType = new StructType().add("multilabel", SQLDataTypes.VectorType)

  def bufferSchema: StructType = new StructType().add("buffer", ArrayType(DoubleType))

  def dataType: DataType = SQLDataTypes.VectorType

  def deterministic: Boolean = true

  def initialize(buffer: MutableAggregationBuffer): Unit = {
    buffer.update(0, Array.fill(size)(0.0))
  }

  def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    if (!input.isNullAt(0)) {
      val buff = buffer.getAs[mutable.WrappedArray[Double]](0)
      val v = input.getAs[Vector](0).toSparse
      for (i <- v.indices) {
        buff(i) += v(i)
      }
      buffer.update(0, buff)
    }
  }

  def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    val buff1 = buffer1.getAs[mutable.WrappedArray[Double]](0)
    val buff2 = buffer2.getAs[mutable.WrappedArray[Double]](0)
    for ((x, i) <- buff2.zipWithIndex) {
      buff1(i) += x
    }
    buffer1.update(0, buff1)
  }

  def evaluate(buffer: Row): Vector = Vectors.dense(
    buffer.getAs[Seq[Double]](0).toArray)
}

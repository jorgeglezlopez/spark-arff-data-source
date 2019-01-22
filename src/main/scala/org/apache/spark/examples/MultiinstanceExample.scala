package org.apache.spark.examples

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.util.MetadataUtils
import org.apache.spark.sql.SparkSession

object MultiinstanceExample {

  def main(args: Array[String]): Unit = {

    val multiInstanceDataset = getClass.getClassLoader.getResource("musk1.arff").getPath

    //    Optionally: turn off logger, so prints are more clear
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val log = Logger.getLogger(getClass.getName)

    val sparkSession = SparkSession.builder
      .appName("MultiinstanceExample")
      .master("local[*]")
      .getOrCreate()

    log.info("Loading the multiInstance ARFF data")
    val dataset = sparkSession.read.format("org.apache.spark.ml.source.arff")
      .option("multiInstance", value = true)
      .load(multiInstanceDataset)

    log.info("Print schema")
    dataset.printSchema()
    //    root
    //    |-- bag-id: double (nullable = true)
    //    |-- label: double (nullable = true)
    //    |-- features: vector (nullable = true)

    log.info("Print schema metadata")
    dataset.schema.foreach(x => println(x.metadata))
    //    {"ml_attr":{"name":"molecule_name","vals":["MUSK-jf78","MUSK-jf67",...],"idx":0,"global_idx":[0],"type":"nominal"}}
    //    {"ml_attr":{"name":"class","vals":["0","1"],"idx":0,"global_idx":[167],"type":"nominal"}}
    //    {"ml_attr":{"attrs":{"numeric":[{"idx":0,"name":"f1"},
    //        {"idx":1,"name":"f2"},
    //        {"idx":2,"name":"f3"},
    //        {"idx":3,"name":"f4"},
    //        {"idx":4,"name":"f5"},
    //        {"idx":5,"name":"f6"},
    //        {"idx":6,"name":"f7"},
    //        {"idx":7,"name":"f8"},
    //        {"idx":8,"name":"f9"},
    //        {"idx":9,"name":"f10"},
    //        {"idx":10,"name":"f11"},
    //        {"idx":11,"name":"f12"},
    //        {"idx":12,"name":"f13"},
    //        {"idx":13,"name":"f14"},
    //        {"idx":14,"name":"f15"},
    //        {"idx":15,"name":"f16"},
    //        {"idx":16,"name":"f17"},
    //        {"idx":17,"name":"f18"},
    //        {"idx":18,"name":"f19"},
    //        {"idx":19,"name":"f20"},
    //        {"idx":20,"name":"f21"},
    //        {"idx":21,"name":"f22"},
    //        {"idx":22,"name":"f23"},
    //        {"idx":23,"name":"f24"},
    //        {"idx":24,"name":"f25"},
    //        {"idx":25,"name":"f26"},
    //        {"idx":26,"name":"f27"},
    //        {"idx":27,"name":"f28"},
    //        {"idx":28,"name":"f29"},
    //        {"idx":29,"name":"f30"},
    //        {"idx":30,"name":"f31"},
    //        {"idx":31,"name":"f32"},
    //        {"idx":32,"name":"f33"},
    //        {"idx":33,"name":"f34"},
    //        {"idx":34,"name":"f35"},
    //        {"idx":35,"name":"f36"},
    //        {"idx":36,"name":"f37"},
    //        {"idx":37,"name":"f38"},
    //        {"idx":38,"name":"f39"},
    //        {"idx":39,"name":"f40"},
    //        {"idx":40,"name":"f41"},
    //        {"idx":41,"name":"f42"},
    //        {"idx":42,"name":"f43"},
    //        {"idx":43,"name":"f44"},
    //        {"idx":44,"name":"f45"},
    //        {"idx":45,"name":"f46"},
    //        {"idx":46,"name":"f47"},
    //        {"idx":47,"name":"f48"},
    //        {"idx":48,"name":"f49"},
    //        {"idx":49,"name":"f50"},
    //        {"idx":50,"name":"f51"},
    //        {"idx":51,"name":"f52"},
    //        {"idx":52,"name":"f53"},
    //        {"idx":53,"name":"f54"},
    //        {"idx":54,"name":"f55"},
    //        {"idx":55,"name":"f56"},
    //        {"idx":56,"name":"f57"},
    //        {"idx":57,"name":"f58"},
    //        {"idx":58,"name":"f59"},
    //        {"idx":59,"name":"f60"},
    //        {"idx":60,"name":"f61"},
    //        {"idx":61,"name":"f62"},
    //        {"idx":62,"name":"f63"},
    //        {"idx":63,"name":"f64"},
    //        {"idx":64,"name":"f65"},
    //        {"idx":65,"name":"f66"},
    //        {"idx":66,"name":"f67"},
    //        {"idx":67,"name":"f68"},
    //        {"idx":68,"name":"f69"},
    //        {"idx":69,"name":"f70"},
    //        {"idx":70,"name":"f71"},
    //        {"idx":71,"name":"f72"},
    //        {"idx":72,"name":"f73"},
    //        {"idx":73,"name":"f74"},
    //        {"idx":74,"name":"f75"},
    //        {"idx":75,"name":"f76"},
    //        {"idx":76,"name":"f77"},
    //        {"idx":77,"name":"f78"},
    //        {"idx":78,"name":"f79"},
    //        {"idx":79,"name":"f80"},
    //        {"idx":80,"name":"f81"},
    //        {"idx":81,"name":"f82"},
    //        {"idx":82,"name":"f83"},
    //        {"idx":83,"name":"f84"},
    //        {"idx":84,"name":"f85"},
    //        {"idx":85,"name":"f86"},
    //        {"idx":86,"name":"f87"},
    //        {"idx":87,"name":"f88"},
    //        {"idx":88,"name":"f89"},
    //        {"idx":89,"name":"f90"},
    //        {"idx":90,"name":"f91"},
    //        {"idx":91,"name":"f92"},
    //        {"idx":92,"name":"f93"},
    //        {"idx":93,"name":"f94"},
    //        {"idx":94,"name":"f95"},
    //        {"idx":95,"name":"f96"},
    //        {"idx":96,"name":"f97"},
    //        {"idx":97,"name":"f98"},
    //        {"idx":98,"name":"f99"},
    //        {"idx":99,"name":"f100"},
    //        {"idx":100,"name":"f101"},
    //        {"idx":101,"name":"f102"},
    //        {"idx":102,"name":"f103"},
    //        {"idx":103,"name":"f104"},
    //        {"idx":104,"name":"f105"},
    //        {"idx":105,"name":"f106"},
    //        {"idx":106,"name":"f107"},
    //        {"idx":107,"name":"f108"},
    //        {"idx":108,"name":"f109"},
    //        {"idx":109,"name":"f110"},
    //        {"idx":110,"name":"f111"},
    //        {"idx":111,"name":"f112"},
    //        {"idx":112,"name":"f113"},
    //        {"idx":113,"name":"f114"},
    //        {"idx":114,"name":"f115"},
    //        {"idx":115,"name":"f116"},
    //        {"idx":116,"name":"f117"},
    //        {"idx":117,"name":"f118"},
    //        {"idx":118,"name":"f119"},
    //        {"idx":119,"name":"f120"},
    //        {"idx":120,"name":"f121"},
    //        {"idx":121,"name":"f122"},
    //        {"idx":122,"name":"f123"},
    //        {"idx":123,"name":"f124"},
    //        {"idx":124,"name":"f125"},
    //        {"idx":125,"name":"f126"},
    //        {"idx":126,"name":"f127"},
    //        {"idx":127,"name":"f128"},
    //        {"idx":128,"name":"f129"},
    //        {"idx":129,"name":"f130"},
    //        {"idx":130,"name":"f131"},
    //        {"idx":131,"name":"f132"},
    //        {"idx":132,"name":"f133"},
    //        {"idx":133,"name":"f134"},
    //        {"idx":134,"name":"f135"},
    //        {"idx":135,"name":"f136"},
    //        {"idx":136,"name":"f137"},
    //        {"idx":137,"name":"f138"},
    //        {"idx":138,"name":"f139"},
    //        {"idx":139,"name":"f140"},
    //        {"idx":140,"name":"f141"},
    //        {"idx":141,"name":"f142"},
    //        {"idx":142,"name":"f143"},
    //        {"idx":143,"name":"f144"},
    //        {"idx":144,"name":"f145"},
    //        {"idx":145,"name":"f146"},
    //        {"idx":146,"name":"f147"},
    //        {"idx":147,"name":"f148"},
    //        {"idx":148,"name":"f149"},
    //        {"idx":149,"name":"f150"},
    //        {"idx":150,"name":"f151"},
    //        {"idx":151,"name":"f152"},
    //        {"idx":152,"name":"f153"},
    //        {"idx":153,"name":"f154"},
    //        {"idx":154,"name":"f155"},
    //        {"idx":155,"name":"f156"},
    //        {"idx":156,"name":"f157"},
    //        {"idx":157,"name":"f158"},
    //        {"idx":158,"name":"f159"},
    //        {"idx":159,"name":"f160"},
    //        {"idx":160,"name":"f161"},
    //        {"idx":161,"name":"f162"},
    //        {"idx":162,"name":"f163"},
    //        {"idx":163,"name":"f164"},
    //        {"idx":164,"name":"f165"},
    //        {"idx":165,"name":"f166"}]},
    //    "global_idx":[1,2,...,166],"num_attrs":166}}

    log.info("Show first 20 instances")
    dataset.show()
    //    +------+-----+--------------------+
    //    |bag-id|label|            features|
    //    +------+-----+--------------------+
    //    |  46.0|  1.0|[42.0,-198.0,-109...|
    //    |  46.0|  1.0|[42.0,-191.0,-142...|
    //    |  46.0|  1.0|[42.0,-191.0,-142...|
    //    |  46.0|  1.0|[42.0,-198.0,-110...|
    //    |  45.0|  1.0|[42.0,-198.0,-102...|
    //    |  45.0|  1.0|[42.0,-191.0,-142...|
    //    |  45.0|  1.0|[42.0,-190.0,-142...|
    //    |  45.0|  1.0|[42.0,-199.0,-102...|
    //    |  44.0|  1.0|[40.0,-173.0,-142...|
    //    |  44.0|  1.0|[44.0,-159.0,-63....|
    //    |  43.0|  1.0|[42.0,-170.0,-63....|
    //    |  43.0|  1.0|[41.0,-95.0,-61.0...|
    //    |  43.0|  1.0|[45.0,-199.0,-108...|
    //    |  42.0|  1.0|[41.0,90.0,-141.0...|
    //    |  42.0|  1.0|[70.0,-30.0,-61.0...|
    //    |  42.0|  1.0|[85.0,-158.0,-63....|
    //    |  42.0|  1.0|[50.0,-192.0,-143...|
    //    |  41.0|  1.0|[46.0,-194.0,-148...|
    //    |  41.0|  1.0|[47.0,-102.0,-60....|
    //    |  40.0|  1.0|[47.0,-197.0,-144...|
    //    +------+-----+--------------------+
    //    only showing top 20 rows

    log.info("Compute statistics about the instances in the bags")
    import org.apache.spark.sql.functions._
    dataset.groupBy("bag-id")
      .count()
      .agg(min("count"), max("count"), avg("count"))
      .show()
    //    +----------+----------+-----------------+
    //    |min(count)|max(count)|       avg(count)|
    //    +----------+----------+-----------------+
    //    |         2|        40|5.173913043478261|
    //    +----------+----------+-----------------+


    log.info("Characteristics of the dataset")
    val numBags = dataset.select("bag-id").distinct().count()
    val numInstances = dataset.count()
    val numFeatures = dataset.first().getAs[Vector]("features").size
    val numClasses = MetadataUtils.getNumClasses(dataset.schema("label")).getOrElse(-1)

    //    If we didn't have the information in the metadata it would be..
    //    val numClasses = dataset.select("label").distinct().count()

    println(s"Dataset characteristics:\n bags:$numBags instances:$numInstances " +
      s"features:$numFeatures classes:$numClasses")
    //    Dataset characteristics:
    //      bags:92 instances:476 features:166 classes:2

  }
}

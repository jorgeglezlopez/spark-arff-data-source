package org.apache.spark.ml.source.arff

/**
  * `ARFF` package implements Spark SQL data source API for loading ARFF data as DataFrame.
  *
  * The loaded DataFrame can have the following columns:
  * * Bag:
  *   * One column of type Double, if "multiinstance" is set to true.
  * * Features:
  *   * One column of type Vector.
  * * Labels:
  *   * One column of type Double if "multilabelFile" is not specified
  *   * One column of type Vector (sparse) if the xml "multilabelFile" with the labels is
  *     specified, or more than one output is indicated.
  *
  *
  * To use ARFF data source, you need to set "org.apache.spark.ml.source.arff" as the format
  * in DataFrameReader and optionally specify options, for example:
  * {{{
  *
  *   // Scala
  *   val df = spark.read.format("org.apache.spark.ml.source.arff")
  *     .load("data/big_iris.arff")
  *
  *   // Java
  *   Dataset<Row> df = spark.read().format("org.apache.spark.ml.source.arff")
  *     .load("data/big_iris.arff");
  *
  *  // Python
  *   df = spark.read.format("org.apache.spark.ml.source.arff")
  *     .load("data/big_iris.arff")
  *
  * }}}
  *
  * ARFF data source supports the following options:
  *  - "comment": symbol to declare comments (? by default).
  *  - "schemaFile": header file that declares the attributes, empty by deault because the header
  *      is included at the begining of the main file.
  *  - "xmlMultilabelFile": xml file that specifies the names of the labels
  *  - "numOutputs": The number of outputs (labels) is set to 1 by default.
  *  - "numFeatures": number of features.
  *  - "multiInstance": Indicates if the file is multiinstance, and has a bag-id field.
  *
  * Note that this class is public for documentation purpose. Please don't use this class directly.
  * Rather, use the data source API as illustrated above.
  *
  * @author Jorge Gonzalez Lopez
  *         gonzalezlopej@vcu.edu
  */
class ARFFDataSource private() {}
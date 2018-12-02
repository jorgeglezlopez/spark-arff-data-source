package org.apache.spark.ml.source.arff

import java.io.IOException

import org.apache.spark.internal.Logging
import org.apache.spark.ml.attribute.ExtendedAttributeGroup
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructField

import scala.collection.mutable.ArrayBuffer

/**
  * This class receives the StructField of the requiered field (columns) in the DataFrame
  * Using these StructField it reconstructs the ARFFAttributeParser of each attribute
  * and uses them in order to parse ARFF records (lines) and transform them into Rows.
  *
  * @param bagID Optional StructField from the bag-id field.
  * @param features StructField extracted from the features field
  * @param labels Structfield extracted from the labels field.
  *
  */
private[arff] class ARFFInstanceParser(bagID: Option[StructField],
                                       labels: StructField,
                                       features: StructField)
  extends Logging with Serializable {

  def this(labels: StructField, features: StructField) = {
    this(None, labels, features)
  }

  /** Group attribute for the Bag in case is specified **/
  val bagAttributeGroup: Option[ExtendedAttributeGroup] = {
    if (bagID.isDefined) Some(ExtendedAttributeGroup.fromStructField(bagID.get))
    else None
  }

  /** Group attribute for the features **/
  val featuresAttributeGroup: ExtendedAttributeGroup = ExtendedAttributeGroup.fromStructField(features)

  /** Group attribute for the labels **/
  val labelsAttributeGroup: ExtendedAttributeGroup = ExtendedAttributeGroup.fromStructField(labels)

  /** Attribute parsers that includes bag, features, and labels **/
  val attributesParsers: Array[ARFFAttributeParser] = {

    var size: Int = 0
    if (bagID.isDefined) size += 1
    size += featuresAttributeGroup.numAttributes.get
    size += labelsAttributeGroup.numAttributes.get

    val p = Array.ofDim[ARFFAttributeParser](size)

    if (bagID.isDefined) {
      val bagAttributeGroup = ExtendedAttributeGroup.fromStructField(bagID.get)
      val pos = bagAttributeGroup.global_idx.get.apply(0)
      if (bagAttributeGroup.isExtended.get.apply(0)) {
        p(pos) = ARFFAttributeParser.attributeMetadata2Parser(bagAttributeGroup
          .getExtendedAttribute(0)
          .get.toMetadata, ARFFAttributeCategory.Bag)
      } else {
        p(pos) = ARFFAttributeParser.attributeMetadata2Parser(bagAttributeGroup.getAttribute(0).get
          .toMetadata(), ARFFAttributeCategory.Bag)
      }
    }

    for (i <- 0 until featuresAttributeGroup.numAttributes.get) {
      val pos = featuresAttributeGroup.global_idx.get.apply(i)
      if (featuresAttributeGroup.isExtended.get.apply(i)) {
        val meta = featuresAttributeGroup.getExtendedAttribute(i).get.toMetadata
        p(pos) = ARFFAttributeParser.attributeMetadata2Parser(meta, ARFFAttributeCategory.Features)
      } else {
        p(pos) = ARFFAttributeParser.attributeMetadata2Parser(featuresAttributeGroup.getAttribute(i).get.toMetadata(), ARFFAttributeCategory.Features)
      }
    }


    for (i <- 0 until labelsAttributeGroup.numAttributes.get) {
      val pos = labelsAttributeGroup.global_idx.get.apply(i)
      if (labelsAttributeGroup.isExtended.get.apply(i)) {
        p(pos) = ARFFAttributeParser.attributeMetadata2Parser(labelsAttributeGroup.getExtendedAttribute(i).get.toMetadata, ARFFAttributeCategory.Label)
      } else {
        p(pos) = ARFFAttributeParser.attributeMetadata2Parser(labelsAttributeGroup.getAttribute(i).get.toMetadata(), ARFFAttributeCategory.Label)
      }
    }

    p
  }

  /** Number of bags, labels, and features **/
  val numBags: Int = if (bagAttributeGroup.isDefined) bagAttributeGroup.get.numAttributes.get else 0
  val numLabels: Int = labelsAttributeGroup.numAttributes.get
  val numFeatures: Int = featuresAttributeGroup.numAttributes.get

  def parseArffRecord(record: String): Row = {

    var output: Row = null

    // If is sparse
    if (record.trim().startsWith("{")) {

      val unformatted = record.substring(1, record.length() - 1).split(",")

      val bag = ArrayBuffer[(Int, Double)]()
      val features = ArrayBuffer[(Int, Double)]()
      val labels = ArrayBuffer[(Int, Double)]()

      for (i <- unformatted) {
        val tuple = i.trim().split(" ").map(_.trim)

        val pos = tuple(0).toInt
        val value = if (tuple(1).equals("?")) {
          log.warn(s"Missing value (?) found in attribute $pos")
          Double.NaN
        } else {
          attributesParsers(pos).getValue(tuple(1))
        }

        if (pos > attributesParsers.length) {
          throw new IOException(s"The position of the sparse attribute is $pos but the number of attributes parsed from header is ${attributesParsers.length}.")
        }
        else { // pos is the global_idx that can be used with the array of parsers, but we also need the local_idx to build the sparse vectors
          if (attributesParsers(pos).arffAttributeCategory == ARFFAttributeCategory.Features) {
            val localPos = featuresAttributeGroup.global_idx.get.indexOf(pos)
            features += ((localPos, value))
          } else if (attributesParsers(pos).arffAttributeCategory == ARFFAttributeCategory.Label) {

            val localPos = labelsAttributeGroup.global_idx.get.indexOf(pos)
            labels += ((localPos, value))
          } else {
            val localPos = bagAttributeGroup.get.global_idx.get.indexOf(pos)
            bag += ((localPos, value))
          }
        }
      }

      if (numBags == 0 && numLabels == 1) {
        output = Row(labels(0)._2, Vectors.sparse(numFeatures, features))
      }
      else if (numBags == 0 && numLabels > 1) {
        output = Row(Vectors.sparse(numLabels, labels), Vectors.sparse(numFeatures, features))
      }
      else if (numBags == 1 && numLabels == 1) {
        output = Row(bag(0)._2, labels(0)._2, Vectors.sparse(numFeatures, features))
      }
      else if (numBags == 1 && numLabels > 1) {
        output = Row(bag(0)._2, Vectors.sparse(numLabels, labels), Vectors.sparse(numFeatures, features))
      } else {
        throw new Exception(s"Malformed record $record has ${bag.size} bags, ${labels.size} labels and ${features.size} features")
      }
    }
    // If dense records
    else {

      val unformatted = record.trim().split(",").map(_.trim)

      if (unformatted.length != attributesParsers.length) {
        throw new IOException(s"Illegal header and/or data. The number of attributes parsed from header is ${attributesParsers.length}. And the number of attributes found in the data is ${unformatted.length}")
      }

      val bag = new Array[Double](numBags)
      val features = new Array[Double](numFeatures)
      val labels = new Array[Double](numLabels)

      for (i <- 0 until numBags) {
        val pos = bagAttributeGroup.get.global_idx.get.apply(i)
        val value = unformatted(pos)
        if (value.equals("?")) {
          log.warn(s"Missing value (?) found in attribute $pos")
          features(i) = Double.NaN
        } else {
          features(i) = attributesParsers(pos).getValue(value)
        }
      }

      for (i <- 0 until numLabels) {
        val pos = labelsAttributeGroup.global_idx.get.apply(i)
        val value = unformatted(pos)
        if (value.equals("?")) {
          log.warn(s"Missing value (?) found in attribute $pos")
          labels(i) = Double.NaN
        } else {
          labels(i) = attributesParsers(pos).getValue(value)
        }
      }

      for (i <- 0 until numFeatures) {
        val pos = featuresAttributeGroup.global_idx.get.apply(i)
        val value = unformatted(pos)
        if (value.equals("?")) {
          log.warn(s"Missing value (?) found in attribute $pos")
          features(i) = Double.NaN
        } else {
          features(i) = attributesParsers(pos).getValue(value)
        }
      }

      if (numBags == 0 && numLabels == 1) {
        output = Row(labels(0), Vectors.dense(features).compressed)
      }
      else if (numBags == 0 && numLabels > 1) {
        output = Row(Vectors.dense(labels).toSparse, Vectors.dense(features).compressed)
      }
      else if (numBags == 1 && numLabels == 1) {
        output = Row(bag(0), labels(0), Vectors.dense(features).compressed)
      }
      else if (numBags == 1 && numLabels > 1) {
        output = Row(bag(0), Vectors.dense(labels).toSparse, Vectors.dense(features).compressed)
      } else {
        throw new Exception(s"Malformed record $record has ${bag.length} bags, ${labels.length} " +
          s"labels and ${features.length} features")
      }

    }

    output
  }


  def parseArffRow(row: Row): String = {

    var str = ""

    // Process bags if any
    if (numBags > 0) {
      val record = row.getAs[Double]("bag-id")
      val pos = bagAttributeGroup.get.global_idx.get.apply(0)
      val bag = attributesParsers(pos).getRecord(record)
      str += bag + ","
    }

    // Adds the features
    val features = row.getAs[Vector]("features")
    for (i <- 0 until numFeatures) {
      val record = features.apply(0)
      val pos = featuresAttributeGroup.global_idx.get.apply(i)
      val f = attributesParsers(pos).getRecord(record)
      str += f + ","
    }

    // At the end of the string adds the label/s
    if (numLabels > 1) {
      val labels = row.getAs[Vector]("multilabel")
      for (i <- 0 until numLabels) {
        val record = labels.apply(0)
        val pos = labelsAttributeGroup.global_idx.get.apply(i)
        val l = attributesParsers(pos).getRecord(record)
        str += l + ","
      }
    } else {
      val record = row.getAs[Double]("label")
      val pos = labelsAttributeGroup.global_idx.get.apply(0)
      val l = attributesParsers(pos).getRecord(record)
      str += l + ","
    }

    // Remove the last colon
    str.dropRight(1)

    str
  }

}





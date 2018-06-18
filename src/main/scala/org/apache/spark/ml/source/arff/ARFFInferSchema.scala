package org.apache.spark.ml.source.arff

import org.apache.spark.ml.attribute._
import org.apache.spark.ml.linalg.VectorUDT
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.xml.XML


/**
  * This object infers an schema from the ARFF header.
  * The schema will depend on the learning paradigm used in the file:
  *
  * // Traditional learning
  * root
  * |-- label: double (nullable = true)
  * |-- features: vector (nullable = true)
  *
  * // Multi-output learning
  * root
  * |-- multilabel: vector (nullable = true)
  * |-- features: vector (nullable = true)
  *
  * // Multi-instance learning
  * root
  * |-- bag-id: double (nullable = true)
  * |-- label: double (nullable = true)
  * |-- features: vector (nullable = true)
  *
  * Once the schema has been infered, the information parsed from the header
  * will be stored in the metadata.
  *
  */
private[arff] object ARFFInferSchema {

  def infer(sparkSession: SparkSession,
            header: RDD[(String, Long)],
            arffOptions: ARFFOptions): StructType = {

    var schema = new StructType()

    var remainingHeader = header

    if (arffOptions.multiInstance) {

      // In a multiInstance arff file the first attribute is the bag-id
      val headerBag = header.first()

      // Update remaining header
      remainingHeader = remainingHeader.filter({ case (_, index) => index > 0 })

      val bagIDParser = ARFFAttributeParser.attributeDefinition2Parser(headerBag._1,
        ARFFAttributeCategory.Bag)
      val att: Attribute = bagIDParser match {
        case parser: NominalParser =>
          NominalAttribute.defaultAttr
            .withName(bagIDParser.name)
            .withIndex(0)
            .withValues(parser.nominalValues)
        case _ =>
          NumericAttribute.defaultAttr
            .withName(bagIDParser.name)
            .withIndex(0)
      }

      val bagGroup = new ExtendedAttributeGroup("bag-id", Array(headerBag._2.toInt), Array(att))

      // Add struct field to the schema with the created metadata
      schema = schema.add(StructField("bag-id", DoubleType, nullable = false, bagGroup.toMetadata))
    }

    // Finds the correct struct field depending on whenever the file is multilabel or singlelabel
    if (arffOptions.xmlMultilabelFile.isDefined) {

      val labelsNames = loadLabels(sparkSession, arffOptions.xmlMultilabelFile.get)

      val bcLabelsNames = sparkSession.sparkContext.broadcast(labelsNames)

      val multilabelHeader = remainingHeader.filter({ case (line, _) =>
        val tmp = ARFFAttributeParser.attributeDefinition2Parser(line, ARFFAttributeCategory.Label)
        var isLabel = false
        for (i <- bcLabelsNames.value.indices; if !isLabel) {
          if (tmp.name.equals(bcLabelsNames.value(i))) {
            isLabel = true
          }
        }

        isLabel
      })

      bcLabelsNames.unpersist(false)

      val numFoundLabels = multilabelHeader.count()
      require(labelsNames.length == multilabelHeader.count(), s"Only $numFoundLabels labels " +
        s"were found out of ${labelsNames.length}. Some labels were not recognized, check the " +
        s"names don't have special characters")

      val multilabelIndices = multilabelHeader.values.collect().sortWith(_ < _)

      val attLabels = new ArrayBuffer[Attribute]()
      val ext_attLabels = new ArrayBuffer[ExtendedAttribute]()

      multilabelHeader
        .zipWithIndex()
        .collect()
        .foreach({ case ((definition, _), localIdx) =>
          val labelParser = ARFFAttributeParser.attributeDefinition2Parser(definition, ARFFAttributeCategory.Label)
          labelParser match {
            case parser: NominalParser =>
              attLabels += NominalAttribute.defaultAttr
                .withName(labelParser.name)
                .withIndex(localIdx.toInt)
                .withValues(parser.nominalValues)
            //          case parser: DateParser =>
            //            ext_attLabels += DateAttribute.defaultAttr
            //              .withName(labelParser.name)
            //              .withIndex(localIdx.toInt)
            //              .withFormat(parser.dateFormat)
            //          case _: StringParser =>
            //            ext_attLabels += StringAttribute.defaultAttr
            //              .withName(labelParser.name)
            //              .withIndex(localIdx.toInt)
            case _ =>
              attLabels += NumericAttribute.defaultAttr
                .withName(labelParser.name)
                .withIndex(localIdx.toInt)
          }
        })

      val multilabelGroup = new ExtendedAttributeGroup("multilabel", multilabelHeader.values.collect().map(_.toInt), attLabels.toArray, ext_attLabels.toArray)

      schema = schema.add(StructField("multilabel", new VectorUDT(), nullable = false, multilabelGroup.toMetadata))

      remainingHeader = remainingHeader.filter({ case (_, index) =>
        !multilabelIndices.contains(index)
      })

    }
    else {

      if (arffOptions.numOutputs == 1) {

        val sizeHeader = header.count()

        val labelHeader = remainingHeader
          .filter({ case (_, index) => index == (sizeHeader - 1) })
          .first()

        val attLabels = new ArrayBuffer[Attribute]()
        val ext_attLabels = new ArrayBuffer[ExtendedAttribute]()

        val labelParser = ARFFAttributeParser.attributeDefinition2Parser(labelHeader._1, ARFFAttributeCategory.Label)
        labelParser match {
          case parser: NominalParser =>
            attLabels += NominalAttribute.defaultAttr
              .withName(labelParser.name)
              .withIndex(0)
              .withValues(parser.nominalValues)
          case parser: DateParser =>
            ext_attLabels += DateAttribute.defaultAttr
              .withName(labelParser.name)
              .withIndex(0)
              .withFormat(parser.dateFormat)
          case _: StringParser =>
            ext_attLabels += StringAttribute.defaultAttr
              .withName(labelParser.name)
              .withIndex(0)
          case _ =>
            attLabels += NumericAttribute.defaultAttr
              .withName(labelParser.name)
              .withIndex(0)
        }

        val labelGroup = new ExtendedAttributeGroup("label", Array(labelHeader._2.toInt), attLabels.toArray, ext_attLabels.toArray)

        // Add struct field to the schema with the created metadata
        schema = schema.add(StructField("label", DoubleType, nullable = false, labelGroup.toMetadata))

        remainingHeader = remainingHeader.filter({ case (_, index) => index != (sizeHeader - 1) })

      } else {

        val sizeHeader = header.count()

        val attLabels = new ArrayBuffer[Attribute]()
        val ext_attLabels = new ArrayBuffer[ExtendedAttribute]()

        val labelHeader = remainingHeader
          .filter({ case (_, index) => index >= (sizeHeader - arffOptions.numOutputs) })
          .collect()


        labelHeader.zipWithIndex.foreach({ case ((definition, _), localIdx) =>

          val labelParser = ARFFAttributeParser.attributeDefinition2Parser(definition, ARFFAttributeCategory.Label)
          labelParser match {
            case parser: NominalParser =>
              attLabels += NominalAttribute.defaultAttr
                .withName(labelParser.name)
                .withIndex(localIdx)
                .withValues(parser.nominalValues)
            case _ =>
              attLabels += NumericAttribute.defaultAttr
                .withName(labelParser.name)
                .withIndex(localIdx)
          }
        })

        val labelIndices = labelHeader.map(_._2.toInt)

        val labelGroup = new ExtendedAttributeGroup("label", labelIndices, attLabels.toArray, ext_attLabels.toArray)

        // Add struct field to the schema with the created metadata
        schema = schema.add(StructField("multilabel", new VectorUDT(), nullable = false, labelGroup
          .toMetadata))

        remainingHeader = remainingHeader.filter({ case (_, index) =>
          !labelIndices.contains(index)
        })

      }

    }

    val attFeatures = new ArrayBuffer[Attribute]()
    val ext_attFeatures = new ArrayBuffer[ExtendedAttribute]()

    remainingHeader.zipWithIndex().collect().foreach({ case ((definition, _), localIdx) =>
      val labelParser = ARFFAttributeParser.attributeDefinition2Parser(definition, ARFFAttributeCategory.Label)
      labelParser match {
        case parser: NominalParser =>
          attFeatures += NominalAttribute.defaultAttr
            .withName(labelParser.name)
            .withIndex(localIdx.toInt)
            .withValues(parser.nominalValues)
        case parser: DateParser =>
          ext_attFeatures += DateAttribute.defaultAttr
            .withName(labelParser.name)
            .withIndex(localIdx.toInt)
            .withFormat(parser.dateFormat)
        case _: StringParser =>
          ext_attFeatures += StringAttribute.defaultAttr
            .withName(labelParser.name)
            .withIndex(localIdx.toInt)
        case _ =>
          attFeatures += NumericAttribute.defaultAttr
            .withName(labelParser.name)
            .withIndex(localIdx.toInt)
      }
    })

    val featuresIndices = remainingHeader.values.collect().map(_.toInt)

    val featuresGroup = new ExtendedAttributeGroup("features", featuresIndices, attFeatures.toArray, ext_attFeatures.toArray)

    schema = schema.add(StructField("features", new VectorUDT(), nullable = false, featuresGroup.toMetadata))

    schema
  }

  private def loadLabels(sparkSession: SparkSession,
                         labelsFile: String): mutable.Buffer[String] = {

    // Load file, clean it, and delete the <?xml version="1.0" encoding="utf-8"?> line
    val xmlLabels = sparkSession.sparkContext.textFile(labelsFile)
      .filter(line => !line.isEmpty || line.contains("<!--"))
      .map(_.trim)
      .collect
      .mkString("").split("\\?>")(1)

    // Transform to elements to find all the tags with label and the names
    val xmlElem = XML.loadString(xmlLabels)
    val allLabels = xmlElem \\ "label"
    val names = allLabels.map { x =>
      (x \ "@name").text
        .replace("\"", "").replaceAll("'", "").replace("\\", "")
    }.toBuffer

    names
  }


}
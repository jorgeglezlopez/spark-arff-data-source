package org.apache.spark.ml.source.arff

import java.text.SimpleDateFormat
import java.util.Date

import org.apache.spark.sql.types.Metadata

import scala.util.matching.Regex

/**
  * Class that parses a attribute from the header in an arff file.
  *
  * @param name                  name of the attribute.
  * @param arffAttributeCategory category of the attribute.
  */
private[arff]
abstract class ARFFAttributeParser(val name: String,
                                   val arffAttributeCategory: ARFFAttributeCategory)
  extends Serializable {

  /**
    * Transforms the raw value from an arff file to Double
    *
    * @param record raw value.
    * @return transformed value.
    */
  def getValue(record: String): Double = record.toDouble

  /**
    * Transforms a double value to a raw value for an arff file
    *
    * @param value transformed value.
    * @return raw value.
    */
  def getRecord(value: Double): String = value.toString
}

/**
  * Class that parses a numeric attribute from the header in an arff file.
  *
  * @param _name                  name of the attribute.
  * @param _arffAttributeCategory category of the attribute.
  */
private[arff] case class NumericParser(_name: String, _arffAttributeCategory: ARFFAttributeCategory)
  extends ARFFAttributeParser(_name, _arffAttributeCategory) {}

/**
  * Class that parses a integer attribute from the header in an arff file.
  *
  * @param _name                  name of the attribute.
  * @param _arffAttributeCategory category of the attribute.
  */
private[arff] case class IntegerParser(_name: String, _arffAttributeCategory: ARFFAttributeCategory)
  extends ARFFAttributeParser(_name, _arffAttributeCategory) {

  /**
    * Transforms a double value to integer and then to raw value for an arff file
    *
    * @param value transformed value.
    * @return raw value
    */
  override def getRecord(value: Double): String = value.toInt.toString
}

/**
  * Class that parses a real number attribute from the header in an arff file.
  *
  * @param _name                  name of the attribute.
  * @param _arffAttributeCategory category of the attribute.
  */
private[arff] case class RealParser(_name: String, _arffAttributeCategory: ARFFAttributeCategory)
  extends ARFFAttributeParser(_name, _arffAttributeCategory) {}

/**
  * Class that parses a nominal attribute from the header in an arff file.
  *
  * @param _name                  name of the attribute.
  * @param _arffAttributeCategory category of the attribute.
  */
private[arff] case class NominalParser(_name: String,
                                       _arffAttributeCategory: ARFFAttributeCategory,
                                       nominalValues: Array[String])
  extends ARFFAttributeParser(_name, _arffAttributeCategory) {

  /** Number of different values */
  lazy val length: Int = nominalValues.length

  /**
    * Returns the position that takes the record within the array of possible values
    *
    * @param record raw value
    * @return pos
    */
  override def getValue(record: String): Double = nominalValues.indexOf(record).toDouble

  /**
    * Transforms the position to the original raw value
    *
    * @param value pos
    * @return raw value
    */
  override def getRecord(value: Double): String = nominalValues(value.toInt)
}

/**
  * Class that parses a string attribute from the header in an arff file.
  *
  * Warning: The parse of the string values is only one way, once is transformed to numeric
  * it cannot recuperate the original string value
  *
  * @param _name              name of the attribute.
  * @param _arffAttributeType category of the attribute.
  */
private[arff] case class StringParser(_name: String, _arffAttributeType: ARFFAttributeCategory)
  extends ARFFAttributeParser(_name, _arffAttributeType) {

  /**
    * Transforms a string to a numeric value computing its hashcode
    *
    * @param record raw value
    * @return hash value
    */
  override def getValue(record: String): Double = record.hashCode()
}

/**
  * Class that parses a date attribute from the header in an arff file.
  *
  * @param _name              name of the attribute.
  * @param _arffAttributeType category of the attribute.
  * @param dateFormat         string representation of format. Accepts the ISO-8601 combined date and time format. By default 'yyyy-MM-dd'T'HH:mm:ss'
  */
private[arff] case class DateParser(_name: String,
                                    _arffAttributeType: ARFFAttributeCategory,
                                    dateFormat: String = "yyyy-MM-dd'T'HH:mm:ss")
  extends ARFFAttributeParser(_name, _arffAttributeType) {

  /** Date formatter */
  val sdfDate = new SimpleDateFormat(dateFormat)

  /**
    * Transforms the date to milliseconds
    *
    * @param record string representation of date
    * @return milliseconds
    */
  override def getValue(record: String): Double = {
    // Some datasets have the date values between double quotes
    if (record(0) == '\"') {
      sdfDate.parse(record.substring(1, record.length() - 1)).getTime
    } else {
      sdfDate.parse(record).getTime
    }
  }

  /**
    * Transforms milliseconds to a date which uses the original format
    *
    * @param value milliseconds
    * @return formated date
    */
  override def getRecord(value: Double): String = {
    val d = new Date(value.toLong)
    sdfDate.format(d)
  }

}

/**
  * Class that represents an unssoported attribute. At the moment is no longer used since all the types of attributes
  * for arff file are supported.
  *
  * @param _name              name of the attribute.
  * @param _arffAttributeType category of the attribute.
  *
  */
private[arff] case class NotSupportedParser(_name: String,
                                            _arffAttributeType: ARFFAttributeCategory)
  extends ARFFAttributeParser(_name, _arffAttributeType) {

  /**
    * Transforms the raw record to NaN
    *
    * @param record value
    * @return NaN
    */
  override def getValue(record: String): Double = Double.NaN
}


object ARFFAttributeParser {

  //   Regular expressions based on:
  //   https://github.com/strelec/QuantumLearn/blob/master/src/main/scala/qlearn/dataset/loaders/ArffLoader.scala
  //   Thanks to QuantumLearn
  private val literal = "'?(.*?)'?"
  private val nominal = raw"\{\s*.*?\s*\}"
  private val dateFormat = raw"\042?\s*(.*?)\s*\042?"
  private val kind = raw"(real|numeric|integer|string|relational|$nominal|date)"
  val attribute: Regex = raw"(?i)@attribute\s+$literal{1}\s+$kind\s*$dateFormat".r
  //		private val attribute = raw"(?i)@attribute\s*+$literal\s*+$kind+\s*$dateFormat".r <- old one used * instead of +
  //    allowing to to don't have spaces between name and values


  def attributeDefinition2Parser(attributeDefinition: String,
                                 arffAttributeCategory: ARFFAttributeCategory): ARFFAttributeParser = {

    // word1 is the name of the attr, word2 is the type of value and word3 is the date format
    var attribute(word1, word2, word3) = attributeDefinition

    word1 = word1.replace("\"", "").replaceAll("'", "").replace("\\", "") // Sometimes the names are between quotes or with scaping characters

    val parser = if (word2.toLowerCase() == "real") {
      RealParser(word1, arffAttributeCategory)
    }
    else if (word2.toLowerCase() == "numeric") {
      NumericParser(word1, arffAttributeCategory)
    }
    else if (word2.toLowerCase() == "integer") {
      IntegerParser(word1, arffAttributeCategory)
    }
    else if (word2.startsWith("{")) {
      NominalParser(word1, arffAttributeCategory, word2.substring(1, word2.length() - 1).split(",").map(_.trim()))
    }
    else if (word2.toLowerCase() == "date") {
      DateParser(word1, arffAttributeCategory, word3)
    }
    else if (word2.toLowerCase() == "string") {
      StringParser(word1, arffAttributeCategory)
    }
    else {
      NotSupportedParser(word1, arffAttributeCategory)
      throw new Exception(s"The attribute $word1 is malformed and/or not supported.")
    }

    parser
  }


  def attributeMetadata2Parser(metadata: Metadata,
                               arffAttributeCategory: ARFFAttributeCategory): ARFFAttributeParser = {

    if (metadata.contains("ml_attr")) {
      val attributeMetadata = metadata.getMetadata("ml_attr")

      if (attributeMetadata.contains("type")) {
        val attType = attributeMetadata.getString("type")
        val name = attributeMetadata.getString("name")
        if (attType.toLowerCase() == "real") {
          RealParser(name, arffAttributeCategory)
        }
        else if (attType.toLowerCase() == "numeric") {
          NumericParser(name, arffAttributeCategory)
        }
        else if (attType.toLowerCase() == "integer") {
          IntegerParser(name, arffAttributeCategory)
        }
        else if (attType.startsWith("nominal")) {
          NominalParser(name, arffAttributeCategory, attributeMetadata.getStringArray("vals"))
        }
        else if (attType.toLowerCase() == "date") {
          if (attributeMetadata.contains("date_format")) {
            DateParser(name, arffAttributeCategory, attributeMetadata.getString("date_format"))
          } else {
            DateParser(name, arffAttributeCategory)
          }
        }
        else if (attType.toLowerCase() == "string") {
          StringParser(name, arffAttributeCategory)
        }
        else {
          throw new Exception(s"The attribute $attType is malformed and/or not supported.")
          // NotSupportedParser(name, arffAttributeCategory)
        }
      } else {
        val name = attributeMetadata.getString("name")
        // The Attribute toMetadata() implementation (original source of Spark) skips the type whenever is Numeric
        NumericParser(name, arffAttributeCategory)
      }
    } else {
      throw new Exception(s"The metadata $metadata is malformed and/or not supported.")
      // NotSupportedParser("unknown", arffAttributeCategory)
    }

  }

}





package org.apache.spark.ml.source.arff

import org.apache.spark.internal.Logging

private[arff] class ARFFOptions(@transient private val parameters: Map[String, String])
  extends Logging with Serializable {

  private def getChar(paramName: String, default: Char): Char = {
    val paramValue = parameters.get(paramName)
    paramValue match {
      case None => default
      case Some(null) => default
      case Some(value) if value.length == 0 => '\u0000'
      case Some(value) if value.length == 1 => value.charAt(0)
      case _ => throw new RuntimeException(s"$paramName cannot be more than one character")
    }
  }

  private def getInt(paramName: String, default: Int): Int = {
    val paramValue = parameters.get(paramName)
    paramValue match {
      case None => default
      case Some(null) => default
      case Some(value) => try {
        value.toInt
      } catch {
        case e: NumberFormatException =>
          throw new RuntimeException(s"$paramName should be an integer. Found $value")
      }
    }
  }

  private def getBool(paramName: String, default: Boolean = false): Boolean = {
    val param = parameters.getOrElse(paramName, default.toString)
    if (param == null) {
      default
    } else if (param.toLowerCase == "true") {
      true
    } else if (param.toLowerCase == "false") {
      false
    } else {
      throw new Exception(s"$paramName flag can be true or false")
    }
  }

  val comment = getChar("comment", '%')

  val schemaFile = parameters.get("schemaFile")

  val xmlMultilabelFile = parameters.get("xmlMultilabelFile")

  val numOutputs = getInt("numOutputs", 1)

  val multiInstance = getBool("multiInstance", false)

}
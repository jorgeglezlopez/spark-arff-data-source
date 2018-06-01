package org.apache.spark.ml.attribute

sealed abstract class ExtendedAttributeType(val name: String)

object ExtendedAttributeType {

  /** String type. */
  val String: ExtendedAttributeType = {
    case object AlphaNumerical extends ExtendedAttributeType("string")
    AlphaNumerical
  }

  /** Time type. */
  val Date: ExtendedAttributeType = {
    case object Time extends ExtendedAttributeType("date")
    Time
  }

  /** Unresolved type. */
  val Unresolved: ExtendedAttributeType = {
    case object Unresolved extends ExtendedAttributeType("unresolved")
    Unresolved
  }

  /**
    * Gets the [[AttributeType]] object from its name.
    *
    * @param name attribute type name: "numeric", "nominal", or "binary"
    */
  def fromName(name: String): ExtendedAttributeType = {
    if (name == String.name) {
      String
    } else if (name == Date.name) {
      Date
    } else if (name == Unresolved.name) {
      Unresolved
    } else {
      throw new IllegalArgumentException(s"Cannot recognize type $name.")
    }
  }
}

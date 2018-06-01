package org.apache.spark.ml.source.arff

/**
  * :: DeveloperApi ::
  * An enum-like type for arff attribute types Bag, Features, and Label
  */
sealed abstract class ARFFAttributeCategory(val name: String)

/**
  * :: DeveloperApi ::
  */
object ARFFAttributeCategory {

  /** Features type. */
  case object Features extends ARFFAttributeCategory("features")

  /** Bag type. */
  case object Bag extends ARFFAttributeCategory("bag-id")

  /** Label type. */
  case object Label extends ARFFAttributeCategory("label")

  /** Unresolved type. */
  case object Unresolved extends ARFFAttributeCategory("unresolved")

  /**
    * Gets the AttributeType object from its name.
    *
    * @param name attribute type name: "numeric", "nominal", or "binary"
    */
  def fromName(name: String): ARFFAttributeCategory = {
    name match {
      case Features.name =>
        Features
      case Bag.name =>
        Bag
      case Label.name =>
        Label
      case Unresolved.name =>
        Unresolved
      case _ =>
        throw new IllegalArgumentException(s"Cannot recognize type $name.")
    }
  }
}
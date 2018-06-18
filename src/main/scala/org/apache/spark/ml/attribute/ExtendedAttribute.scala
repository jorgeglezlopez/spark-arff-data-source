package org.apache.spark.ml.attribute

import org.apache.spark.sql.types.{DoubleType, Metadata, MetadataBuilder, StructField}

/**
  * :: DeveloperApi ::
  * Abstract class that represents an attribute type added in this data source.
  * StringAttribute and DateAttribute extend this class.
  */
sealed abstract class ExtendedAttribute extends Serializable {

  name.foreach { n =>
    require(n.nonEmpty, "Cannot have an empty string for name.")
  }
  index.foreach { i =>
    require(i >= 0, s"Index cannot be negative but got $i")
  }

  /** Attribute type. */
  def attrType: ExtendedAttributeType

  /** Name of the attribute. None if it is not set. */
  def name: Option[String]

  /** Copy with a new name. */
  def withName(name: String): ExtendedAttribute

  /** Copy without the name. */
  def withoutName: ExtendedAttribute

  /** Index of the attribute. None if it is not set. */
  def index: Option[Int]

  /** Copy with a new index. */
  def withIndex(index: Int): ExtendedAttribute

  /** Copy without the index. */
  def withoutIndex: ExtendedAttribute

  /**
    * Tests whether this attribute is alpha-numeric, true for [[String]].
    */
  def isString: Boolean

  /**
    * Tests whether this attribute is a date, true for Date.
    */
  def isDate: Boolean

  /**
    * Converts this attribute to [[Metadata]].
    *
    * @param withType whether to include the type info
    */
  private[attribute] def toMetadataImpl(withType: Boolean): Metadata

  /**
    * Converts this attribute to [[Metadata]]. For numeric attributes, the type info is excluded to
    * save space, because numeric type is the default attribute type. For nominal and binary
    * attributes, the type info is included.
    */
  private[attribute] def toMetadataImpl: Metadata = {

    // In the original implementation they avoid the type for Numeric, we will always save the type
    toMetadataImpl(withType = true)
  }

  /** Converts to ML metadata with some existing metadata. */
  def toMetadata(existingMetadata: Metadata): Metadata = {
    new MetadataBuilder()
      .withMetadata(existingMetadata)
      .putMetadata(AttributeKeys.ML_ATTR, toMetadataImpl)
      .build()
  }

  /** Converts to ML metadata */
  def toMetadata: Metadata = toMetadata(Metadata.empty)

  /**
    * Converts to a [[StructField]] with some existing metadata.
    *
    * @param existingMetadata existing metadata to carry over
    */
  def toStructField(existingMetadata: Metadata): StructField = {
    val newMetadata = new MetadataBuilder()
      .withMetadata(existingMetadata)
      .putMetadata(ExtendedAttributeKeys.ML_EXT_ATTR, withoutName.withoutIndex.toMetadataImpl)
      .build()
    StructField(name.get, DoubleType, nullable = false, newMetadata)
  }

  /** Converts to a [[StructField]]. */
  def toStructField: StructField = toStructField(Metadata.empty)

  override def toString: String = toMetadataImpl(withType = true).toString
}

/** Trait for extended ML attribute factories. */
private[attribute] trait ExtendedAttributeFactory {

  /**
    * Creates an [[ExtendedAttribute]] from a [[Metadata]] instance.
    */
  private[attribute] def fromMetadata(metadata: Metadata): ExtendedAttribute

  /**
    * Creates an [[ExtendedAttribute]] from a [[StructField]] instance, optionally preserving name.
    */
  private[ml] def decodeStructField(field: StructField, preserveName: Boolean): ExtendedAttribute = {
    //    require(field.dataType.isInstanceOf[NumericType])
    val metadata = field.metadata
    val mlAttr = ExtendedAttributeKeys.ML_EXT_ATTR
    if (metadata.contains(mlAttr)) {
      val attr = fromMetadata(metadata.getMetadata(mlAttr))
      if (preserveName) {
        attr
      } else {
        attr.withName(field.name)
      }
    } else {
      ExtendedUnresolvedAttribute
    }
  }

  /**
    * Creates an [[Attribute]] from a [[StructField]] instance.
    */
  def fromStructField(field: StructField): ExtendedAttribute = decodeStructField(field, preserveName = false)
}


object ExtendedAttribute extends ExtendedAttributeFactory {

  private[attribute] override def fromMetadata(metadata: Metadata): ExtendedAttribute = {
    import org.apache.spark.ml.attribute.ExtendedAttributeKeys._
    val attrType = if (metadata.contains(TYPE)) {
      metadata.getString(TYPE)
    } else {
      //      ExtendedAttributeType.Unresolved.name
      throw new IllegalArgumentException(s"Cannot recognize type.")
    }
    getFactory(attrType).fromMetadata(metadata)
  }

  /** Gets the attribute factory given the attribute type name. */
  private def getFactory(attrType: String): ExtendedAttributeFactory = {
    if (attrType == ExtendedAttributeType.String.name) {
      StringAttribute
    } else if (attrType == ExtendedAttributeType.Date.name) {
      DateAttribute
    } else {
      throw new IllegalArgumentException(s"Cannot recognize type $attrType.")
    }
  }
}

/**
  * :: DeveloperApi ::
  * A string attribute.
  *
  * @param name  optional name
  * @param index optional index
  */
class StringAttribute private[ml](
                                   override val name: Option[String] = None,
                                   override val index: Option[Int] = None) extends ExtendedAttribute {

  override def attrType: ExtendedAttributeType = ExtendedAttributeType.String

  override def withName(name: String): StringAttribute = copy(name = Some(name))

  override def withoutName: StringAttribute = copy(name = None)

  override def withIndex(index: Int): StringAttribute = copy(index = Some(index))

  override def withoutIndex: StringAttribute = copy(index = None)

  override def isString: Boolean = true

  override def isDate: Boolean = false

  /** Convert this attribute to metadata. */
  override private[attribute] def toMetadataImpl(withType: Boolean): Metadata = {
    import org.apache.spark.ml.attribute.ExtendedAttributeKeys._
    val bldr = new MetadataBuilder()
    if (withType) bldr.putString(TYPE, attrType.name)
    name.foreach(bldr.putString(NAME, _))
    index.foreach(bldr.putLong(INDEX, _))
    bldr.build()
  }

  /** Creates a copy of this attribute with optional changes. */
  private def copy(
                    name: Option[String] = name,
                    index: Option[Int] = index): StringAttribute = {
    new StringAttribute(name, index)
  }

  override def equals(other: Any): Boolean = {
    other match {
      case o: StringAttribute =>
        (name == o.name) &&
          (index == o.index)
      case _ =>
        false
    }
  }

  override def hashCode: Int = {
    var sum = 17
    sum = 37 * sum + name.hashCode
    sum = 37 * sum + index.hashCode
    sum
  }
}

/**
  * :: DeveloperApi ::
  * Factory methods for string attributes.
  */
object StringAttribute extends ExtendedAttributeFactory {

  /** The default string attribute. */
  val defaultAttr: StringAttribute = new StringAttribute

  private[attribute] override def fromMetadata(metadata: Metadata): StringAttribute = {
    import org.apache.spark.ml.attribute.ExtendedAttributeKeys._
    val name = if (metadata.contains(NAME)) Some(metadata.getString(NAME)) else None
    val index = if (metadata.contains(INDEX)) Some(metadata.getLong(INDEX).toInt) else None
    new StringAttribute(name, index)
  }
}


/**
  * :: DeveloperApi ::
  * A date attribute.
  *
  * @param name   optional name
  * @param index  optional index
  * @param format string representation of the format of the date (optional)
  */
class DateAttribute private[ml](
                                 override val name: Option[String] = None,
                                 override val index: Option[Int] = None,
                                 val format: Option[String] = None) extends ExtendedAttribute {

  override def attrType: ExtendedAttributeType = ExtendedAttributeType.Date

  override def isString: Boolean = false

  override def isDate: Boolean = true

  override def withName(name: String): DateAttribute = copy(name = Some(name))

  override def withoutName: DateAttribute = copy(name = None)

  override def withIndex(index: Int): DateAttribute = copy(index = Some(index))

  override def withoutIndex: DateAttribute = copy(index = None)

  def withFormat(format: String): DateAttribute = copy(format = Some(format))

  def withoutFormat: DateAttribute = copy(format = None)

  /** Creates a copy of this attribute with optional changes. */
  private def copy(
                    name: Option[String] = name,
                    index: Option[Int] = index,
                    format: Option[String] = format): DateAttribute = {
    new DateAttribute(name, index, format)
  }

  override private[attribute] def toMetadataImpl(withType: Boolean): Metadata = {
    import org.apache.spark.ml.attribute.ExtendedAttributeKeys._
    val bldr = new MetadataBuilder()
    if (withType) bldr.putString(TYPE, attrType.name)
    name.foreach(bldr.putString(NAME, _))
    index.foreach(bldr.putLong(INDEX, _))
    format.foreach(bldr.putString(FORMAT, _))
    bldr.build()
  }

  override def equals(other: Any): Boolean = {
    other match {
      case o: DateAttribute =>
        (name == o.name) &&
          (index == o.index) &&
          (format == o.format)
      case _ =>
        false
    }
  }

  override def hashCode: Int = {
    var sum = 17
    sum = 37 * sum + name.hashCode
    sum = 37 * sum + index.hashCode
    sum = 37 * sum + format.hashCode
    sum
  }
}

/**
  * :: DeveloperApi ::
  * Factory methods for date attributes.
  */
object DateAttribute extends ExtendedAttributeFactory {

  /** The default nominal attribute. */
  final val defaultAttr: DateAttribute = new DateAttribute

  private[attribute] override def fromMetadata(metadata: Metadata): DateAttribute = {
    import org.apache.spark.ml.attribute.ExtendedAttributeKeys._
    val name = if (metadata.contains(NAME)) Some(metadata.getString(NAME)) else None
    val index = if (metadata.contains(INDEX)) Some(metadata.getLong(INDEX).toInt) else None
    val format = if (metadata.contains(FORMAT)) Some(metadata.getString(FORMAT)) else None
    new DateAttribute(name, index, format)
  }
}


/**
  * :: DeveloperApi ::
  * An unresolved attribute.
  */
object ExtendedUnresolvedAttribute extends ExtendedAttribute {

  override def attrType: ExtendedAttributeType = ExtendedAttributeType.Unresolved

  override def withIndex(index: Int): ExtendedAttribute = this

  override def isString: Boolean = false

  override def withoutIndex: ExtendedAttribute = this

  override def isDate: Boolean = false

  override def name: Option[String] = None

  override private[attribute] def toMetadataImpl(withType: Boolean): Metadata = {
    Metadata.empty
  }

  override def withoutName: ExtendedAttribute = this

  override def index: Option[Int] = None

  override def withName(name: String): ExtendedAttribute = this

}
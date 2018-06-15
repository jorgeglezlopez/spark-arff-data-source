package org.apache.spark.ml.attribute

import org.apache.spark.ml.attribute.Attribute.getFactory
import org.apache.spark.ml.linalg.VectorUDT
import org.apache.spark.sql.types.{Metadata, MetadataBuilder, StructField}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
  * :: DeveloperApi ::
  * ExtendedAttributes that describe a vector ML column.
  *
  * @param name          name of the attribute group (the ML column name)
  * @param numAttributes optional number of attributes. At most one of `numAttributes` and `attrs`
  *                      can be defined.
  * @param global_idx    optional array of indices respect all the attributes.
  * @param ml_attrs      optional array of attributes. Attribute will be copied with their corresponding
  *                      indices in the array.
  * @param ml_ext_attrs  optional array of extended-attributes. Attribute will be copied with their
  *                      corresponding indices in the array.
  */
class ExtendedAttributeGroup private(val name: String,
                                     var numAttributes: Option[Int],
                                     val global_idx: Option[Array[Int]],
                                     ml_attrs: Option[Array[Attribute]],
                                     ml_ext_attrs: Option[Array[ExtendedAttribute]]) extends Serializable {

  require(!name.isEmpty, "Cannot have an empty string for name.")
  require(!(numAttributes.isDefined && ml_attrs.isDefined),
    "Cannot have both numAttributes and ml_attrs defined.")
  require(!(numAttributes.isDefined && ml_ext_attrs.isDefined),
    "Cannot have both numAttributes and ml_ext_attrs defined.")

  if (numAttributes.isEmpty) {
    var aux = 0
    if (ml_attrs.isDefined) aux += ml_attrs.get.length
    if (ml_ext_attrs.isDefined) aux += ml_ext_attrs.get.length

    if (aux == 0) aux = numAttributes.get
    numAttributes = Some(aux)
  }


  if (global_idx.isDefined) {
    require(numAttributes.get == global_idx.get.length,
      s"Global_idx has size ${global_idx.get.length}, while there are ${numAttributes.get} attributes.")
  }

  def this(name: String, numAttributes: Int) =
    this(name, Some(numAttributes), None, None, None)

  def this(name: String, ml_attrs: Array[Attribute]) =
    this(name, None, None, Some(ml_attrs), None)

  def this(name: String, ml_ext_attrs: Array[ExtendedAttribute]) =
    this(name, None, None, None, Some(ml_ext_attrs))

  def this(name: String, ml_attrs: Array[Attribute], ml_ext_attrs: Array[ExtendedAttribute]) =
    this(name, None, None, Some(ml_attrs), Some(ml_ext_attrs))

  def this(name: String, global_idx: Array[Int], numAttributes: Int) =
    this(name, Some(numAttributes), Some(global_idx), None, None)

  def this(name: String, global_idx: Array[Int], ml_attrs: Array[Attribute]) =
    this(name, None, Some(global_idx), Some(ml_attrs), None)

  def this(name: String, global_idx: Array[Int], ml_ext_attrs: Array[ExtendedAttribute]) =
    this(name, None, Some(global_idx), None, Some(ml_ext_attrs))

  def this(name: String, global_idx: Array[Int], ml_attrs: Array[Attribute], ml_ext_attrs: Array[ExtendedAttribute]) =
    this(name, None, Some(global_idx), Some(ml_attrs), Some(ml_ext_attrs))

  def this(name: String, global_idx: Option[Array[Int]], ml_attrs: Option[Array[Attribute]], ml_ext_attrs: Option[Array[ExtendedAttribute]]) =
    this(name, None, global_idx, ml_attrs, ml_ext_attrs)

  val attributes: Option[Array[Attribute]] = ml_attrs.map(_.view.zipWithIndex.map { case (attr, i) =>
    if (attr.index.isEmpty) attr.withIndex(i)
    else attr
  }.toArray.sortWith({ case (a, b) => a.index.get < b.index.get }))

  val ext_attributes: Option[Array[ExtendedAttribute]] = ml_ext_attrs.map(_.view.zipWithIndex.map { case (attr, i) =>
    if (attr.index.isEmpty) attr.withIndex(i)
    else attr
  }.toArray.sortWith({ case (a, b) => a.index.get < b.index.get }))

  val isExtended: Option[Array[Boolean]] = {
    val check = Array.fill(numAttributes.get)(false)
    if (ext_attributes.isDefined) {
      for (att <- ext_attributes.get) {
        check(att.index.get) = true
      }
    }
    Some(check)
  }

  val idx2Pos: Option[mutable.Map[Int, Int]] = {
    val map = mutable.Map[Int, Int]()

    if (attributes.isDefined) {
      for (i <- attributes.get.indices) {
        map += ((attributes.get.apply(i).index.get, i))
      }
    }
    if (ext_attributes.isDefined) {
      for (i <- ext_attributes.get.indices) {
        map += ((ext_attributes.get.apply(i).index.get, i))
      }
    }

    Some(map)
  }

  def getAttribute(idx: Int): Option[Attribute] = {
    if (!isExtended.get.apply(idx)) {
      val pos = idx2Pos.get(idx)
      Some(attributes.get.apply(pos))
    } else {
      None
    }
  }

  def getExtendedAttribute(idx: Int): Option[ExtendedAttribute] = {
    if (isExtended.get.apply(idx)) {
      val pos = idx2Pos.get(idx)
      Some(ext_attributes.get.apply(pos))
    } else {
      None
    }
  }

  /** Converts to metadata without name. */
  private[attribute] def toMetadataImpl: Metadata = {

    val bldr = new MetadataBuilder()
    val attrBldr = new MetadataBuilder()

    if (ext_attributes.isDefined) {
      val stringMetadata = ArrayBuffer.empty[Metadata]
      val dateMetadata = ArrayBuffer.empty[Metadata]
      ext_attributes.get.foreach {
        case string: StringAttribute =>
          stringMetadata += string.toMetadataImpl(withType = false)
        case date: DateAttribute =>
          dateMetadata += date.toMetadataImpl(withType = false)
        case ExtendedUnresolvedAttribute =>
      }
      if (stringMetadata.nonEmpty) {
        attrBldr.putMetadataArray(ExtendedAttributeType.String.name, stringMetadata.toArray)
      }
      if (dateMetadata.nonEmpty) {
        attrBldr.putMetadataArray(ExtendedAttributeType.Date.name, dateMetadata.toArray)
      }
    }
    if (attributes.isDefined) {
      val numericMetadata = ArrayBuffer.empty[Metadata]
      val nominalMetadata = ArrayBuffer.empty[Metadata]
      val binaryMetadata = ArrayBuffer.empty[Metadata]
      attributes.get.foreach {
        case numeric: NumericAttribute =>
          numericMetadata += numeric.toMetadataImpl(withType = false)
        case nominal: NominalAttribute =>
          nominalMetadata += nominal.toMetadataImpl(withType = false)
        case binary: BinaryAttribute =>
          binaryMetadata += binary.toMetadataImpl(withType = false)
        case UnresolvedAttribute =>
      }
      if (numericMetadata.nonEmpty) {
        attrBldr.putMetadataArray(AttributeType.Numeric.name, numericMetadata.toArray)
      }
      if (nominalMetadata.nonEmpty) {
        attrBldr.putMetadataArray(AttributeType.Nominal.name, nominalMetadata.toArray)
      }
      if (binaryMetadata.nonEmpty) {
        attrBldr.putMetadataArray(AttributeType.Binary.name, binaryMetadata.toArray)
      }
    }

    if (ext_attributes.isDefined || attributes.isDefined) {
      bldr.putMetadata(AttributeKeys.ATTRIBUTES, attrBldr.build())
    }

    bldr.putLong(AttributeKeys.NUM_ATTRIBUTES, numAttributes.get)

    if (global_idx.isDefined) {
      bldr.putLongArray(ExtendedAttributeKeys.GLOBAL_INDEX, global_idx.get.map(_.toLong))
    }

    bldr.build()
  }

  /** Converts to ML metadata with some existing metadata. */
  def toMetadata(existingMetadata: Metadata): Metadata = {


    if (numAttributes.get == 1) {

      var bldr = new MetadataBuilder()

      val meta = if (attributes.isDefined && attributes.get.size > 0) {
        bldr.putLongArray(ExtendedAttributeKeys.GLOBAL_INDEX, global_idx.get.map(_.toLong))
          .withMetadata(attributes.get.head.toMetadataImpl())
          .build()
      } else {
        bldr.putLongArray(ExtendedAttributeKeys.GLOBAL_INDEX, global_idx.get.map(_.toLong))
          .withMetadata(ext_attributes.get.head.toMetadataImpl)
          .build()
      }

      bldr = new MetadataBuilder()

      bldr
        .withMetadata(existingMetadata)
        .putMetadata(AttributeKeys.ML_ATTR, meta)
        .build()

    } else {
      val bldr = new MetadataBuilder()

      bldr.withMetadata(existingMetadata)
        .putMetadata(AttributeKeys.ML_ATTR, toMetadataImpl)
        .build()
    }

  }

  /** Converts to ML metadata */
  def toMetadata: Metadata = toMetadata(Metadata.empty)

  /** Converts to a StructField with some existing metadata. */
  def toStructField(existingMetadata: Metadata): StructField = {
    StructField(name, new VectorUDT, nullable = false, toMetadata(existingMetadata))
  }

  /** Converts to a StructField. */
  def toStructField: StructField = toStructField(Metadata.empty)

  override def equals(other: Any): Boolean = {
    other match {
      case o: ExtendedAttributeGroup =>
        (name == o.name) &&
          (numAttributes == o.numAttributes) &&
          (attributes.map(_.toSeq) == o.attributes.map(_.toSeq)) &&
          (ext_attributes.map(_.toSeq) == o.ext_attributes.map(_.toSeq))
      case _ =>
        false
    }
  }

  override def hashCode: Int = {
    var sum = 17
    sum = 37 * sum + name.hashCode
    sum = 37 * sum + global_idx.hashCode
    sum = 37 * sum + numAttributes.hashCode
    sum = 37 * sum + attributes.map(_.toSeq).hashCode
    sum = 37 * sum + ext_attributes.map(_.toSeq).hashCode
    sum
  }

  override def toString: String = toMetadata.toString
}

/**
  * :: DeveloperApi ::
  * Factory methods to create attribute groups.
  */
object ExtendedAttributeGroup {

  //  import ExtendedAttributeKeys._

  /** Creates an attribute group from a [[Metadata]] instance with name. */
  private[attribute] def fromMetadata(metadata: Metadata, name: String): ExtendedAttributeGroup = {

    import org.apache.spark.ml.attribute.AttributeType._
    import org.apache.spark.ml.attribute.ExtendedAttributeType._

    if (metadata.contains(AttributeKeys.ATTRIBUTES)) {
      val numAttrs: Int = metadata.getLong(AttributeKeys.NUM_ATTRIBUTES).toInt
      val attributes = new ArrayBuffer[Attribute]
      val ext_attributes = new ArrayBuffer[ExtendedAttribute]
      val attrMetadata = metadata.getMetadata(AttributeKeys.ATTRIBUTES)

      if (attrMetadata.contains(Numeric.name)) {
        attrMetadata.getMetadataArray(Numeric.name)
          .map(NumericAttribute.fromMetadata)
          .foreach { attr =>
            attributes += attr
            //          attributes(attr.index.get) = attr
          }
      }
      if (attrMetadata.contains(Nominal.name)) {
        attrMetadata.getMetadataArray(Nominal.name)
          .map(NominalAttribute.fromMetadata)
          .foreach { attr =>
            attributes += attr
            //          attributes(attr.index.get) = attr
          }
      }
      if (attrMetadata.contains(Binary.name)) {
        attrMetadata.getMetadataArray(Binary.name)
          .map(BinaryAttribute.fromMetadata)
          .foreach { attr =>
            attributes += attr
            //          attributes(attr.index.get) = attr
          }
      }
      if (attrMetadata.contains(String.name)) {
        attrMetadata.getMetadataArray(String.name)
          .map(StringAttribute.fromMetadata)
          .foreach { attr =>
            ext_attributes += attr
            //          attributes(attr.index.get) = attr
          }
      }
      if (attrMetadata.contains(Date.name)) {
        attrMetadata.getMetadataArray(Date.name)
          .map(DateAttribute.fromMetadata)
          .foreach { attr =>
            ext_attributes += attr
            //          attributes(attr.index.get) = attr
          }
      }

      require(numAttrs == (attributes.length + ext_attributes.length),
        s"The number of attributes specified in metadata is $numAttrs, but only " +
          s"${attributes.length + ext_attributes.length} attributes were defined.")

      val optionAtt = if (attributes.nonEmpty) Some(attributes.toArray)
      else None

      val optionExtAtt = if (ext_attributes.nonEmpty) Some(ext_attributes.toArray)
      else None

      val globalIdx = if (metadata.contains(ExtendedAttributeKeys.GLOBAL_INDEX))
        Some(metadata.getLongArray(ExtendedAttributeKeys.GLOBAL_INDEX).map(_.toInt))
      else {
        throw new IllegalArgumentException(s"Tag ${ExtendedAttributeKeys.GLOBAL_INDEX} and " +
          s"is missing from metadata, impossible to create ExtendedAttributeGroup.")
      }

      new ExtendedAttributeGroup(name, globalIdx, optionAtt, optionExtAtt)
    } else {

      import org.apache.spark.ml.attribute.AttributeKeys._
      val attType = if (metadata.contains(TYPE)) {
        metadata.getString(TYPE)
      } else {
        AttributeType.Numeric.name
      }

      val globalIdx = if (metadata.contains(ExtendedAttributeKeys.GLOBAL_INDEX))
        Some(metadata.getLongArray(ExtendedAttributeKeys.GLOBAL_INDEX).map(_.toInt))
      else {
        throw new IllegalArgumentException(s"Tag ${ExtendedAttributeKeys.GLOBAL_INDEX} and " +
          s"is missing from metadata, impossible to create ExtendedAttributeGroup.")
      }

      if (attType == ExtendedAttributeType.Date.name ||
        attType == ExtendedAttributeType.String.name) {
        val att = ExtendedAttribute.fromMetadata(metadata)
        new ExtendedAttributeGroup(name, globalIdx, None, Some(Array(att)))
      } else {
        val att = Attribute.fromMetadata(metadata)
        new ExtendedAttributeGroup(name, globalIdx, Some(Array(att)), None)
      }

    }


    //    else if (metadata.contains(AttributeKeys.NUM_ATTRIBUTES)) {
    //      new ExtendedAttributeGroup(name, metadata.getLong(AttributeKeys.NUM_ATTRIBUTES).toInt)
    //    } else {
    //      throw new IllegalArgumentException(s"Tag ${AttributeKeys.ML_ATTR} and " +
    //        s"${AttributeKeys.NUM_ATTRIBUTES} missing from metadata, impossible to " +
    //        s"create ExtendedAttributeGroup.")
    //      null
    //    }
  }

  /** Creates an attribute group from a [[StructField]] instance. */
  def fromStructField(field: StructField): ExtendedAttributeGroup = {

    if (field.metadata.contains(AttributeKeys.ML_ATTR)) {
      fromMetadata(field.metadata.getMetadata(AttributeKeys.ML_ATTR), field.name)
    } else {
      throw new IllegalArgumentException(s"Tag ${AttributeKeys.ML_ATTR} missing from metadata, " +
        s"impossible to create ExtendedAttributeGroup.")
    }

  }
}

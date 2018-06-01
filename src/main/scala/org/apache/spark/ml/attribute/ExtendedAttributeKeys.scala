package org.apache.spark.ml.attribute

/**
  * Keys used to store attributes.
  */
private[attribute] object ExtendedAttributeKeys {
  val GLOBAL_INDEX: String = "global_idx"
  val ML_EXT_ATTR: String = "ext_ml_attr"
  val TYPE: String = "type"
  val NAME: String = "name"
  val INDEX: String = "idx"
  val FORMAT: String = "date_format"
  val EXT_ATTRIBUTES: String = "attrs"
  val NUM_ATTRIBUTES: String = "num_attrs"
}
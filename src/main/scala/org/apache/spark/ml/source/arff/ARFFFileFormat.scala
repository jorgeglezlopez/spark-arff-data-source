package org.apache.spark.ml.source.arff

import java.io.IOException

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.FileStatus
import org.apache.hadoop.mapreduce.Job
import org.apache.spark.SparkException
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.SQLDataTypes
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.catalyst.expressions.codegen.GenerateUnsafeProjection
import org.apache.spark.sql.catalyst.expressions.{AttributeReference, UnsafeProjection}
import org.apache.spark.sql.execution.datasources.{HadoopFileLinesReader, OutputWriterFactory, PartitionedFile, TextBasedFileFormat}
import org.apache.spark.sql.sources.{DataSourceRegister, Filter}
import org.apache.spark.sql.types.{DataTypes, StructType}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.SerializableConfiguration


class DefaultSource extends TextBasedFileFormat with DataSourceRegister with Logging {

  override def shortName(): String = "arff"

  override def toString: String = "ARFF"

  private def verifySchema(dataSchema: StructType, arffOptions: ARFFOptions): Unit = {

    if ((arffOptions.multiInstance && !dataSchema.apply("bag-id").dataType
      .sameType(DataTypes.DoubleType)) ||
      (arffOptions.xmlMultilabelFile.isEmpty && arffOptions.numOutputs == 1 &&
        !dataSchema.apply("label").dataType.sameType(DataTypes.DoubleType)) ||
      (arffOptions.xmlMultilabelFile.isDefined && !dataSchema.apply("multilabel").dataType
        .sameType(SQLDataTypes.VectorType)) ||
      (arffOptions.xmlMultilabelFile.isEmpty && arffOptions.numOutputs > 1 &&
        !dataSchema.apply("multilabel").dataType.sameType(SQLDataTypes.VectorType)) ||
      !dataSchema.apply("features").dataType.sameType(SQLDataTypes.VectorType)
    ) {
      throw new IOException(s"Illegal DataTypes in schema for arff data, schema=$dataSchema")
    }
  }

  override def inferSchema(sparkSession: SparkSession,
                           options: Map[String, String],
                           files: Seq[FileStatus]): Option[StructType] = {

    if (files.isEmpty) {
      log.error("Arff files not found")
      None
    } else {
      val arffOptions: ARFFOptions = new ARFFOptions(options)

      val schemaFile = arffOptions.schemaFile match {

        // If the schema file was specified in the options
        case Some(path) =>
          log.info(s"Infering schema from $path")
          path

        // Otherwise we filter the invalid files and use the first one for the schema
        case _ =>
          val dataFiles = files.filterNot { status =>
            val name = status.getPath.getName
            (name.startsWith("_") && !name.contains("=")) || name.startsWith(".")
          }.toArray

          // Fixed: Not using Uri because spaces in path are transformed to '%20'
          val path: String = if (dataFiles.length == 1) {
            dataFiles.head.getPath.toString
            //      				dataFiles.head.getPath.toUri.toString
          } else if (dataFiles.isEmpty) {
            throw new IOException("No input path specified for arff data")
          } else {
            log.warn(s"Multiple schema infering is not supported. From all the input files the schema will be infered from ${dataFiles.head.getPath.toString}")
            dataFiles.head.getPath.toString
            //      				dataFiles.head.getPath.toUri.toString
          }
          log.info(s"Using the $path file to build the schema")
          path
      }

      // Loads the file that contains the schema and persists it.
      val file = sparkSession.sparkContext.textFile(schemaFile)
        .zipWithIndex()
        .persist(StorageLevel.MEMORY_AND_DISK)

      // Find where the data starts
      val dataLineIndex = {

        val dataLine = file
          .filter({
            case (line, _) => line.trim().toLowerCase().contains("@data")
          }).collect()

        if (dataLine.length == 0) {
          log.warn(s"Line '@data' not found in schema file, will try to use the whole file as header")
          file.count() - 1
        }
        else if (dataLine.length == 1) {
          dataLine(0)._2
        }
        else {
          throw new IOException("Invalid arff file, found '@data' multiple times")
        }
      }

      // Filter comments, empty lines and retrieve lines only from header
      val header = file
        .filter { case (line, index) =>
          (index < dataLineIndex) &&
            line.trim.nonEmpty &&
            !line.trim().startsWith(arffOptions.comment.toString) &&
            line.trim().toLowerCase().startsWith("@attribute") &&
            !line.trim().toLowerCase().endsWith("relational")
        }
        .keys.zipWithIndex() // Rezip, since we have filter some lines and the line numbers will be wrong

      val numAttributes = header.count()
      if (numAttributes == 0) {
        throw new IOException("No attributes found in header file")
      }

      // Unpersist the whole schema file, since from now on we only need the filtered header
      file.unpersist(false)

      val schema = ARFFInferSchema.infer(sparkSession, header, arffOptions)

      Some(schema)
    }
  }

  override def prepareWrite(
                             sparkSession: SparkSession,
                             job: Job,
                             options: Map[String, String],
                             dataSchema: StructType): OutputWriterFactory = {

    throw new SparkException("ARFF writer is not supported, there is not way to write the " +
      "header since OutputWriterFactory writes a line per row")

    null
  }

  override def buildReader(
                            sparkSession: SparkSession,
                            dataSchema: StructType,
                            partitionSchema: StructType,
                            requiredSchema: StructType,
                            filters: Seq[Filter],
                            options: Map[String, String],
                            hadoopConf: Configuration): (PartitionedFile) => Iterator[InternalRow] = {

    val arffOptions: ARFFOptions = new ARFFOptions(options)

    verifySchema(dataSchema, arffOptions)

    val broadcastedHadoopConf =
      sparkSession.sparkContext.broadcast(new SerializableConfiguration(hadoopConf))

    (file: PartitionedFile) => {

      val linesReader = new HadoopFileLinesReader(file, broadcastedHadoopConf.value.value)

      val bagStructField = if (dataSchema.fieldNames.contains("bag-id")) Some(dataSchema.apply("bag-id"))
      else None

      val labelStructField = if (dataSchema.fieldNames.contains("label")) dataSchema.apply("label")
      else dataSchema.apply("multilabel")

      val featuresStructField = dataSchema.apply("features")

      val instanceParser = new ARFFInstanceParser(bagStructField, labelStructField,
        featuresStructField)

      val converter = RowEncoder(dataSchema)

      val fullOutput = dataSchema.map { f =>
        AttributeReference(f.name, f.dataType, f.nullable, f.metadata)()
      }
      val requiredOutput = fullOutput.filter { a =>
        requiredSchema.fieldNames.contains(a.name)
      }

      val requiredColumns: UnsafeProjection = GenerateUnsafeProjection.generate(requiredOutput, fullOutput)

      //				  val iteratorOfRows: Iterator[UnsafeRow] = if(!arffOptions.multiInstance){
      linesReader
        .map(_.toString.trim)
        .filterNot(line => line.isEmpty ||
          line.startsWith("@") ||
          line.startsWith(arffOptions.comment.toString) ||
          line.equals("{}"))
        .map({ line =>
          requiredColumns(converter.toRow(instanceParser.parseArffRecord(line)))
        })
    }
  }
}
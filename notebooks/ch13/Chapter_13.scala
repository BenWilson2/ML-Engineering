// Databricks notebook source
// MAGIC %md 
// MAGIC # Chapter 13 ML Engineering in Action
// MAGIC Author: Ben Wilson

// COMMAND ----------

// MAGIC %md 
// MAGIC ## ML Development Hubris
// MAGIC In this notebook, we'll be looking at some patterns of development that can cause problems in projects. Specifically, issues that arise from unintentionally obfuscated (overly complex) code, prematurely / incorrectly optimized code, and the complexities introduced from early generalization of a code base.

// COMMAND ----------

// MAGIC %md Before we get into the examples that are in the book, we need to generate some data for the use cases here. Keeping with the common theme of this part of the book, we're talking about dogs again, as well as the one thing that they care about more than anything: food.

// COMMAND ----------

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassifier, DecisionTreeClassificationModel}
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel, VectorAssembler, IndexToString}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.evaluation.{
  BinaryClassificationEvaluator,
  MulticlassClassificationEvaluator}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import scala.util.Random
import scala.reflect.ClassTag
import scala.collection.mutable.ArrayBuffer

// COMMAND ----------

case class Dogs(age: Int,
                weight: Double,
                favorite_food: String,
                breed: String,
                good_boy_or_girl: String,
                hungry: Boolean)

case object CoreData {
  def dogBreeds: Seq[String] =
    Seq(
      "Husky",
      "GermanShepherd",
      "Dalmation",
      "Pug",
      "Malamute",
      "Akita",
      "BelgianMalinois",
      "Chinook",
      "Estrela",
      "Doberman",
      "Mastiff"
    )
  def foods: Seq[String] =
    Seq(
      "Kibble",
      "Spaghetti",
      "Labneh",
      "Steak",
      "Hummus",
      "Fajitas",
      "BeoufBourgignon",
      "Bolognese"
    )
  def goodness: Seq[String] =
    Seq("yes", "no", "sometimes", "yesWhenFoodAvailable")
  def hungry: Seq[Boolean] = Seq(true, false)
  def ageSigma = 3
  def ageMean = 2
  def weightSigma = 12
  def weightMean = 60
}

trait DogUtility {

  def getDoggoData[T: ClassTag](a: Seq[T], dogs: Int, seed: Long): Seq[T] = {
    val rnd = new Random(seed)
    Seq.fill(dogs)(a(rnd.nextInt(a.size)))
  }

  def getDistributedIntData(sigma: Double,
                            mean: Double,
                            dogs: Int,
                            seed: Long): Seq[Int] = {
    val rnd = new Random(seed)
    (0 until dogs).map(
      _ => math.ceil(math.abs(rnd.nextGaussian() * sigma + mean)).toInt
    )
  }

  def getDistributedDoubleData(sigma: Double,
                               mean: Double,
                               dogs: Int,
                               seed: Long): Seq[Double] = {
    val rnd = new Random(seed)
    (0 until dogs).map(
      _ =>
        math
          .round(math.abs(rnd.nextGaussian() * sigma * 100 + mean))
          .toDouble / 100
    )
  }

}

object DogDataGeneration extends DogUtility {

  def generateData(rows: Int, seed: Long): DataFrame = {

    val ageData = getDistributedIntData(CoreData.ageSigma, CoreData.ageMean, rows, seed)
    val weightData = getDistributedDoubleData(CoreData.weightSigma, CoreData.weightMean, rows, seed)
    val foodData = getDoggoData(CoreData.foods, rows, seed)
    val breedData = getDoggoData(CoreData.dogBreeds, rows, seed)
    val goodData = getDoggoData(CoreData.goodness, rows, seed)
    val hungryData = getDoggoData(CoreData.hungry, rows, seed)
    val collection = (0 until rows).toArray.map(x => {
      Dogs(ageData(x), weightData(x), foodData(x), breedData(x), goodData(x), hungryData(x))
    }).toSeq
    collection.toDF()
  }
  
  def generateColl(rows: Int, seed: Long) = {

    val ageData = getDistributedIntData(CoreData.ageSigma, CoreData.ageMean, rows, seed)
    val weightData = getDistributedDoubleData(CoreData.weightSigma, CoreData.weightMean, rows, seed)
    val foodData = getDoggoData(CoreData.foods, rows, seed)
    val breedData = getDoggoData(CoreData.dogBreeds, rows, seed)
    val goodData = getDoggoData(CoreData.goodness, rows, seed)
    val hungryData = getDoggoData(CoreData.hungry, rows, seed)
    val collection = Seq(ageData, weightData, foodData, breedData, goodData, hungryData)
    collection
  }

}

// COMMAND ----------

// MAGIC %md ```Note: the object DogDataGeneration is not a very optimal way of generating data. We will be covering a more optimal methodology in listing 13.10```

// COMMAND ----------

// MAGIC %md To get the initial data set for listing 13.1, let's use the sub-optimal generator above to build it.

// COMMAND ----------

val dataLarger = DogDataGeneration.generateData(100000, 42L)
      .withColumn("hungry", when(col("hungry"), "true").otherwise("false"))
      .withColumn("hungry", when(col("breed") === "Husky", "true").otherwise(col("hungry")))
      .withColumn("good_boy_or_girl",  when(col("breed") === "Husky", "yesWhenFoodAvailable")
          .otherwise(col("good_boy_or_girl")))

// COMMAND ----------

// MAGIC %md
// MAGIC ###Listing 13.1 Imperative model prototype
// MAGIC In this code listing, we see an example of building a fairly standard SparkML DecisionTreeClassifier model. The code style, highly imperative in nature, is indicative of how a lot of demos are written, reference implementations, and how a large amount of experimentation work is conducted. 

// COMMAND ----------

val DATA_SOURCE = dataLarger

val indexerFood = new StringIndexer()
  .setInputCol("favorite_food")
  .setOutputCol("favorite_food_si")
  .setHandleInvalid("keep")
  .fit(DATA_SOURCE)

val indexerBreed = new StringIndexer()
  .setInputCol("breed")
  .setOutputCol("breed_si")
  .setHandleInvalid("keep")
  .fit(DATA_SOURCE)

val indexerGood = new StringIndexer()
  .setInputCol("good_boy_or_girl")
  .setOutputCol("good_boy_or_girl_si")
  .setHandleInvalid("keep")
  .fit(DATA_SOURCE)

val indexerHungry = new StringIndexer()
  .setInputCol("hungry")
  .setOutputCol("hungry_si")
  .setHandleInvalid("error")
  .fit(DATA_SOURCE)

val Array(train, test) = DATA_SOURCE.randomSplit(Array(0.75, 0.25))

val indexerLabelConversion = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictionLabel")
  .setLabels(indexerHungry.labelsArray(0))

val assembler = new VectorAssembler()
  .setInputCols(Array("age", "weight", "favorite_food_si", "breed_si", "good_boy_or_girl_si"))
  .setOutputCol("features")

val decisionTreeModel = new DecisionTreeClassifier()
  .setLabelCol("hungry_si")
  .setFeaturesCol("features")
  .setImpurity("gini")
  .setMinInfoGain(1e-4)
  .setMaxDepth(6)
  .setMinInstancesPerNode(5)
  .setMinWeightFractionPerNode(0.05)

val pipeline = new Pipeline()
  .setStages(Array(indexerFood, indexerBreed, indexerGood, indexerHungry, assembler, decisionTreeModel, indexerLabelConversion))

val model = pipeline.fit(train)

val predictions = model.transform(test)

val lossMetric = new BinaryClassificationEvaluator()
  .setLabelCol("hungry_si")
  .setRawPredictionCol("prediction")
  .setMetricName("areaUnderROC")
  .evaluate(predictions)



// COMMAND ----------

// MAGIC %md ### Listing 13.2 Overly complex model prototype
// MAGIC What happens if someone who is trying to 'get fancy' tries their hand at building a prototype? What if they just want to show off their coding skills (flexing) to the rest of the team? Just how convoluted could this potentially become?

// COMMAND ----------

case class ModelReturn(
                        pipeline: PipelineModel,
                        metric: Double
                      )

class BuildDecisionTree(data: DataFrame,
                        trainPercent: Double,
                        labelCol: String) {

  final val LABEL_COL = "label"
  final val FEATURES_COL = "features"
  final val PREDICTION_COL = "prediction"
  final val SCORING_METRIC = "areaUnderROC"

  private def constructIndexers(): Array[StringIndexerModel] = {
    data.schema
      .collect {
        case x if (x.dataType == StringType) & (x.name != labelCol) => x.name
      }
      .map { x =>
        new StringIndexer()
          .setInputCol(x)
          .setOutputCol(s"${x}_si")
          .setHandleInvalid("keep")
          .fit(data)
      }
      .toArray

  }

  private def indexLabel(): StringIndexerModel = {
    data.schema.collect {
      case x if (x.name == labelCol) & (x.dataType == StringType) =>
        new StringIndexer()
          .setInputCol(x.name)
          .setOutputCol(LABEL_COL)
          .setHandleInvalid("error")
          .fit(data)
    }.head
  }

  private def labelInversion(
    labelIndexer: StringIndexerModel
  ): IndexToString = {
    new IndexToString()
      .setInputCol(PREDICTION_COL)
      .setOutputCol(s"${LABEL_COL}_${PREDICTION_COL}")
      .setLabels(labelIndexer.labelsArray(0))
  }

  private def buildVector(
    featureIndexers: Array[StringIndexerModel]
  ): VectorAssembler = {

    val featureSchema = data.schema.names.filterNot(_.contains(labelCol))
    val updatedSchema = featureIndexers.map(_.getInputCol)
    val features = featureSchema.filterNot(updatedSchema.contains) ++ featureIndexers
      .map(_.getOutputCol)
    new VectorAssembler()
      .setInputCols(features)
      .setOutputCol(FEATURES_COL)
  }

  private def buildDecisionTree(): DecisionTreeClassifier = {
    new DecisionTreeClassifier()
      .setLabelCol(LABEL_COL)
      .setFeaturesCol(FEATURES_COL)
      .setImpurity("entropy")
      .setMinInfoGain(1e-7)
      .setMaxDepth(6)
      .setMinInstancesPerNode(5)
  }
  
  private def scorePipeline(testData: DataFrame, pipeline: PipelineModel): Double = {
    new BinaryClassificationEvaluator()
      .setLabelCol(LABEL_COL)
      .setRawPredictionCol(PREDICTION_COL)
      .setMetricName(SCORING_METRIC)
      .evaluate(pipeline.transform(testData))
  }

  def buildPipeline(): ModelReturn = {

    val featureIndexers = constructIndexers()
    val labelIndexer = indexLabel()
    val vectorAssembler = buildVector(featureIndexers)
    val Array(train, test) = data.randomSplit(Array(trainPercent, 1.0-trainPercent))
    val pipeline = new Pipeline()
      .setStages(
        featureIndexers ++ 
        Array(
          labelIndexer,
          vectorAssembler,
          buildDecisionTree(),
          labelInversion(labelIndexer)
        )
      )
      .fit(train)
    
    ModelReturn(pipeline, scorePipeline(test, pipeline))
    
  }

}

object BuildDecisionTree {
  def apply(data: DataFrame,
            trainPercent: Double,
            labelCol: String): BuildDecisionTree =
    new BuildDecisionTree(data, trainPercent, labelCol)
}

// COMMAND ----------

val build = BuildDecisionTree(DATA_SOURCE, 0.75, "hungry").buildPipeline()
display(build.pipeline.transform(dataLarger))

// COMMAND ----------

// MAGIC %md I don't know about you, but I wouldn't want to have to modify this code during the process of testing different features, adapting to model architecture changes, and adding in additional feature engineering steps. It's complicated, tightly-coupled, and looks more like generic framework code than job code. <br>
// MAGIC Were this a final-stage build of the code after the features and all steps involving the modeling completed, this code could potentially make some sense (and even then, it's very difficult to test) from a final production version standpoint. However, in an early phase (as this is clearly at with respect to the level of customization involved), this introduces more chaos than it prevents. It's just far too over-engineered to be useful.

// COMMAND ----------

// MAGIC %md As a bonus (this isn't covered in the text of the book), here's an example of a truly over-engineered analysis of the extracted information from a decision tree model. As you peruse this code, try to imagine having to update a part of it during a development cycle to support additional functionality.<br>
// MAGIC Code like this belongs firmly embedded within a utility framework and should be fully external to any project work. It's a useful bit of code for analyzing the results of a tree-based algorithm, but it's far too complex and singularly focused / generic to be put into the solution repo for a project.

// COMMAND ----------

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.tree.{ContinuousSplit, InternalNode, Node, Split}
import org.apache.spark.ml.classification.{
  DecisionTreeClassificationModel,
  RandomForestClassificationModel
}
import org.apache.spark.ml.tree.{
  CategoricalSplit,
  ContinuousSplit,
  InternalNode,
  Node
}
import org.json4s.jackson.Serialization
import org.json4s.jackson.Serialization.writePretty
import org.json4s.{Formats, FullTypeHints}

case class NodeData(
  featureIndex: Option[Int],
  informationGain: Option[Double],
  continuousSplitThreshold: Option[Double],
  treeNodeType: String,
  splitType: Option[String],
  leftNodeCategories: Option[Array[Double]],
  rightNodeCategories: Option[Array[Double]],
  leftChild: Option[NodeData],
  rightChild: Option[NodeData],
  prediction: Double
) 

object PayloadType extends Enumeration {
  type PayloadType = Value
  val MODEL, PIPELINE = Value
}

object NodeType extends Enumeration {
  type NodeType = Value
  val NODE, LEAF = Value
}

object SplitType extends Enumeration {
  type SplitType = Value
  val CONTINUOUS, CATEGORICAL = Value
}

object PayloadDetermination {

  import PayloadType._

  def payloadType[T](value: T): PayloadType = {
    value match {
      case _: PipelineModel => PIPELINE
      case _                => MODEL
    }
  }

}

case class TreesReport(tree: Int, data: NodeData)

case class FeatureIndexRenamingStructure(featureName: String,
                                    replacementText: String)

def refactorDebugTree(debugStatement: String,
                        features: Array[String]): String = {

    val featureMapping = features.zipWithIndex.map(
      x => FeatureIndexRenamingStructure(x._1, s"""featureIndex" : ${x._2}""")
    )

    featureMapping.foldLeft(debugStatement) {
      case (debugString, field) =>
        debugString.replaceAll(field.replacementText, s"""feature Index" : "${field.featureName}"""")
    }

  }
  
object NodeDetermination {

  import NodeType._
  import SplitType._

  def nodeType(node: Node): NodeType = node match {
    case _: InternalNode => NODE
    case _               => LEAF
  }

  def splitType(split: Split): SplitType = split match {
    case x: ContinuousSplit => CONTINUOUS
    case _                  => CATEGORICAL
  }

}


class TreeModelExtractor[T]
object TreeModelExtractor {
  implicit object DecisionTreeClassifierExtractor
      extends TreeModelExtractor[DecisionTreeClassificationModel]
  implicit object RandomForestClassificationExtractor
      extends TreeModelExtractor[RandomForestClassificationModel]
}


object ModelDecisionExtractor {

  import NodeType._
  import SplitType._

  private def getSplitRuleSet(treeNode: Node): NodeData = {

    val nodeType = NodeDetermination.nodeType(treeNode)
    val internalNodeData = nodeType match {
      case NODE => Some(treeNode.asInstanceOf[InternalNode])
      case LEAF => None
    }

    val splitType = nodeType match {
      case NODE => Some(NodeDetermination.splitType(internalNodeData.get.split))
      case _    => None
    }

    NodeData(
      featureIndex = nodeType match {
        case NODE => Some(internalNodeData.get.split.featureIndex)
        case _    => None
      },
      informationGain = nodeType match {
        case NODE => Some(internalNodeData.get.gain)
        case _    => None
      },
      continuousSplitThreshold = splitType.getOrElse(None) match {
        case CONTINUOUS =>
          Some(
            internalNodeData.get.split.asInstanceOf[ContinuousSplit].threshold
          )
        case _ => None
      },
      treeNodeType = nodeType match {
        case NODE => "node"
        case _    => "leaf"
      },
      splitType = nodeType match {
        case NODE =>
          splitType.get match {
            case CONTINUOUS  => Some("continuous")
            case CATEGORICAL => Some("categorical")
            case _           => None
          }
        case _ => None
      },
      leftNodeCategories = splitType.getOrElse(None) match {
        case CATEGORICAL =>
          Some(
            internalNodeData.get.split
              .asInstanceOf[CategoricalSplit]
              .leftCategories
          )
        case _ => None
      },
      rightNodeCategories = splitType.getOrElse(None) match {
        case CATEGORICAL =>
          Some(
            internalNodeData.get.split
              .asInstanceOf[CategoricalSplit]
              .rightCategories
          )
        case _ => None
      },
      leftChild = nodeType match {
        case NODE => Some(getSplitRuleSet(internalNodeData.get.leftChild))
        case _    => None
      },
      rightChild = nodeType match {
        case NODE => Some(getSplitRuleSet(internalNodeData.get.rightChild))
        case _    => None
      },
      prediction = treeNode.prediction
    )

  }

  private def rulesToString(rules: NodeData): String = {
    implicit val jsonFormat: Formats =
      Serialization.formats(hints = FullTypeHints(List(NodeData.getClass)))
    writePretty(rules)
  }

  private def replaceFeatureIndicesWithText(
    rules: NodeData,
    featureVectorNames: Array[String]
  ): String = {

    featureVectorNames.zipWithIndex
      .map(
        x =>
          FeatureIndexRenamingStructure(x._1, s""""featureIndex" : ${x._2}""")
      )
      .foldLeft(rulesToString(rules)) {
        case (treeText, field) =>
          treeText.replaceAll(
            field.replacementText,
            s"""""featureIndex" : "${field.featureName}""""
          )
      }
  }

  def extractModel[T: TreeModelExtractor](
    model: T,
    featureVectorNames: Array[String]
  ): Array[String] =
    model match {
      case _: DecisionTreeClassificationModel =>
        Array(
          replaceFeatureIndicesWithText(
            getSplitRuleSet(
              model.asInstanceOf[DecisionTreeClassificationModel].rootNode
            ),
            featureVectorNames
          )
        )
      case _: RandomForestClassificationModel =>
        model
          .asInstanceOf[RandomForestClassificationModel]
          .trees
          .map(
            x =>
              replaceFeatureIndicesWithText(
                getSplitRuleSet(x.rootNode),
                featureVectorNames
            )
          )
    }

  private def rulesToStringAlt(rules: TreesReport): String = {
    implicit val jsonFormat: Formats =
      Serialization.formats(hints = FullTypeHints(List(NodeData.getClass)))
    writePretty(rules)
  }

  private def replaceFeatureIndicesWithTextAlt(
    rules: TreesReport,
    featureVectorNames: Array[String]
  ): String = {

    featureVectorNames.zipWithIndex
      .map(
        x =>
          FeatureIndexRenamingStructure(x._1, s""""featureIndex" : ${x._2}""")
      )
      .foldLeft(rulesToStringAlt(rules)) {
        case (treeText, field) =>
          treeText.replaceAll(
            field.replacementText,
            s""""featureIndex" : "${field.featureName}""""
          )
      }
  }

  def extractModelAlt[T: TreeModelExtractor](
    model: T,
    featureVectorNames: Array[String]
  ): Array[String] =
    model match {
      case _: DecisionTreeClassificationModel =>
        Array(
          getSplitRuleSet(
            model.asInstanceOf[DecisionTreeClassificationModel].rootNode
          )
        ).zipWithIndex
          .map(x => TreesReport(x._2, x._1))
          .map(replaceFeatureIndicesWithTextAlt(_, featureVectorNames))
      case _: RandomForestClassificationModel =>
        model
          .asInstanceOf[RandomForestClassificationModel]
          .trees
          .map(x => getSplitRuleSet(x.rootNode))
          .zipWithIndex
          .map(x => TreesReport(x._2, x._1))
          .map(replaceFeatureIndicesWithTextAlt(_, featureVectorNames))
    }


  def extractTreesData[T: TreeModelExtractor](
    model: T,
    featureVectorNames: Array[String]
  ): Array[TreesReport] = {

    model match {
      case _: DecisionTreeClassificationModel =>
        Array(
          getSplitRuleSet(
            model.asInstanceOf[DecisionTreeClassificationModel].rootNode
          )
        ).zipWithIndex.map(x => TreesReport(x._2, x._1))
      case _: RandomForestClassificationModel =>
        model
          .asInstanceOf[RandomForestClassificationModel]
          .trees
          .map(x => getSplitRuleSet(x.rootNode))
          .zipWithIndex
          .map(x => TreesReport(x._2, x._1))
    }

  }

  def blockConvertToJSON(rules: Array[TreesReport]): String = {
    implicit val jsonFormat: Formats =
      Serialization.formats(hints = FullTypeHints(List(TreesReport.getClass)))
    writePretty(rules)
  }

  def blockReplaceJSONText(rules: Array[TreesReport],
                           featureVectorNames: Array[String]): String = {

    featureVectorNames.zipWithIndex
      .map(
        x =>
          FeatureIndexRenamingStructure(x._1, s""""featureIndex" : ${x._2}""")
      )
      .foldLeft(blockConvertToJSON(rules)) {
        case (treeText, field) =>
          treeText.replaceAll(
            field.replacementText,
            s""""featureIndex" : "${field.featureName}""""
          )
      }
  }

}

// COMMAND ----------

// MAGIC %md And using the above monstrosity would look like this:

// COMMAND ----------

val builtModel = build.pipeline.stages.takeRight(2).head.asInstanceOf[DecisionTreeClassificationModel]
val features = build.pipeline.stages.takeRight(3).head.asInstanceOf[VectorAssembler].getInputCols
val treesReport = ModelDecisionExtractor.extractTreesData(builtModel, features)
ModelDecisionExtractor.blockReplaceJSONText(treesReport, features)

// COMMAND ----------

// MAGIC %md Useful? Sure, I guess. Should something like this be in your ML solution code base? That would be a resounding 'no'.<br>
// MAGIC ```[NOTE]```: the model is garbage. It's supposed to be. We generated silly data that utilizes random sampling with a common seed value. If you see a modeling solution that is using real world data that can achieve 100% classification accuracy, you're either leaking your label or the problem is so simple that you shouldn't be using supervised ML to solve it

// COMMAND ----------

// MAGIC %md ### Listing 13.3 Imperative casting
// MAGIC In the following few listings, we're going to look at some relatively innocuous operations (the casting of a data type within a Spark DataFrame) through the lens of different approaches. <br>
// MAGIC In this listing, we'll see the imperative design that is generally used for demonstration in API Docs. It's clear, concise, easy to follow, and as simple of an approach that is possible. The only downside to it is that the constants are hard-coded in the code, which can prove complex if these references are spread throughout a large code base. 

// COMMAND ----------

def simple(df: DataFrame): DataFrame = {
  df.withColumn("age", col("age").cast("double"))
    .withColumn("weight", col("weight").cast("double"))
    .withColumn("hungry", col("hungry").cast("string"))
}

// COMMAND ----------

dataLarger.printSchema

// COMMAND ----------

simple(dataLarger).printSchema

// COMMAND ----------

// MAGIC %md ###Listing 13.4 A hacker's attempt at casting columns
// MAGIC Here we see some pretty unintelligble code. It's overly complex not due to what it's doing, but rather in how it's doing it. It's also incredibly non-performant by using a mutable object definition around the DataFrame. The usage of the function is also merely migrating the constant values of column and type references to another place in the code. This does nothing to solve any problems and is just indicative of someone trying to look smart and failing in an epic manner. <br>
// MAGIC Be on the lookout for implementations like this. If the code is introducing complexity without reducing the computational or space complexity, then it's just useless code that's hard to read, hard to debug, and really challenging to expand upon.

// COMMAND ----------

def hacker(df: DataFrame, castChanges: List[(String, String)]): DataFrame = {
  var mutated = df
  castChanges.foreach { x =>
    mutated = mutated.withColumn(x._1, mutated(x._1).cast(x._2))
  }
  mutated
}

// COMMAND ----------

hacker(dataLarger, List(("age", "double"), ("weight", "double"), ("hungry", "string"))).printSchema

// COMMAND ----------

// MAGIC %md Yikes! Look at that function signature! Who wants to pass in a list of string tuples to a function?! That's just insanity.

// COMMAND ----------

// MAGIC %md ###Listing 13.5 A pure functional programming approach
// MAGIC Here we have 'the mystic'. Someone who is attempting to adhere to a set of design principles (in this case, FP, but in other cases it might be a strict adherence to a particular design theory (e.g. factory patterns) in OO). The code is cleaner than the hacker's code, certainly, but it's still really challenging to call the function due to the weird nature of using a collection of tuples.

// COMMAND ----------

def mystic(df: DataFrame,
              castChanges: List[(String, DataType)]): DataFrame = {
  castChanges.foldLeft(df) {
    case (data, (c, t)) => data.withColumn(c, df(c).cast(t))
  }
}

// COMMAND ----------

mystic(dataLarger, List(("age", DoubleType), ("weight", DoubleType), ("hungry", StringType))).printSchema

// COMMAND ----------

// MAGIC %md The implementation is clearly better than the previous 2 listings, but the signature is still just relocating complexity to a different place. There are also a LOT of things that could go wrong here if the end-user isn't aware of how to perform casting conversion to complex types (Arrays, Maps, etc.). 

// COMMAND ----------

// MAGIC %md ###Listing 13.6 The show-off's casting implementation
// MAGIC Here we have someone who really, desperately wants for people to think that they're skilled. They're going to approach the problem in a way that they think is going to cover a lot of potential issues that might arise from converting types, but in the process of building their solution just ends up going down a rabbit hole of complexity that generates a confusing mess of temporary references and bloated redundant code.

// COMMAND ----------

final val numTypes = List(IntegerType, FloatType, DoubleType, LongType, DecimalType, ShortType)

def showOff(df: DataFrame): DataFrame = {
  df.schema
    .map(
      s =>
        s.dataType match {
          case x if numTypes.contains(x) => s.name -> "n"
          case _                         => s.name -> "s"
      }
    )
    .foldLeft(df) {
      case (df, x) =>
        df.withColumn(x._1, df(x._1).cast(x._2 match {
          case "n" => "double"
          case _   => "string"
        }))
    }
}

// COMMAND ----------

showOff(dataLarger).printSchema

// COMMAND ----------

// MAGIC %md While the approach here is better from an instantiation standpoint (namely that it doesn't require configuring a list of tuples and tries to automate conversions to support the goal of casting numeric types to Doubles and the Boolean type of the target column to a string), the internals of the showOff function are convoluted and unnecessarily complex. The first map over the schema is completely useless (creating a mapping of the name to a temporary string representation of the data type) and the foldLeft utilizes positional notation for referring to the values within the map. Not to mention that this implementation will absolutely detonate if any complex types are present in the DataFrame (Array, Map, Vector) and create confusing exceptions for an end-user of the function. It's sloppy, amateurish, and incredibly complicated to read. F- for effort.

// COMMAND ----------

// MAGIC %md ###Listing 13.7 A slightly more sophisticated casting implementation
// MAGIC This code block is approaching something more akin to what a production-grade implementation for automating casting should look like. As I mentioned in the chapter, though, this in and of itself can be highly obfuscated if no one else on the team is familiar with what is going on here. Because it's so dense, efficient, and involves matching directly on the iterated collection of schema values, it can be very confusing for people who are more accustomed to imperative programming styles. 
// MAGIC As I mentioned in the chapter, it's important to verify that the team is familiar with FP programming styles such as this before just blindly submitting PR's that contain code like this. If this is the development standard that the team chooses to employ, make sure that the entire team is taught and mentored in this paradigm. Provide examples of simple use cases that are relevant to the type of operations that project work will require. Hold hackathons. Just make sure that you're not smugly expecting people to figure something like this out with no guidance.

// COMMAND ----------

def madScientist(df: DataFrame): DataFrame = {
  df.schema.foldLeft(df) {
    case (accum, s) =>
      accum.withColumn(s.name, accum(s.name).cast(s.dataType match {
        case x: IntegerType => x
        case x if numTypes.contains(x) => DoubleType
        case ArrayType(_,_) | MapType(_,_,_) => s.dataType
        case _                         => StringType
      }))
  }
}

// COMMAND ----------

madScientist(dataLarger).printSchema

// COMMAND ----------

// MAGIC %md ###Listing 13.8 Configuration and common structures for data generator
// MAGIC This is the same code as above (before listing 13.1) repeated here for reference.

// COMMAND ----------

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, SparkSession}
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.util.Random

case class Dogs(age: Int,
                weight: Double,
                favorite_food: String,
                breed: String,
                good_boy_or_girl: String,
                hungry: Boolean)

case object CoreData {
  def dogBreeds: Seq[String] = Seq(
      "Husky", "GermanShepherd", "Dalmation", "Pug", "Malamute", "Akita", 
      "BelgianMalinois", "Chinook", "Estrela", "Doberman", "Mastiff")
  def foods: Seq[String] = Seq("Kibble", "Spaghetti", "Labneh", "Steak",
      "Hummus", "Fajitas", "BeoufBourgignon", "Bolognese")
  def goodness: Seq[String] = Seq("yes", "no", "sometimes", "yesWhenFoodAvailable")
  def hungry: Seq[Boolean] = Seq(true, false)
  def ageSigma = 3
  def ageMean = 2
  def weightSigma = 12
  def weightMean = 60
}

trait DogUtility {
  lazy val spark: SparkSession = SparkSession.builder().getOrCreate()
  def getDoggoData[T: ClassTag](a: Seq[T], dogs: Int, seed: Long): Seq[T] = {
    val rnd = new Random(seed)
    Seq.fill(dogs)(a(rnd.nextInt(a.size)))
  }
  def getDistributedIntData(sigma: Double,
                            mean: Double,
                            dogs: Int,
                            seed: Long): Seq[Int] = {
    val rnd = new Random(seed)
    (0 until dogs).map(
      _ => math.ceil(math.abs(rnd.nextGaussian() * sigma + mean)).toInt
    )
  }
  def getDistributedDoubleData(sigma: Double, mean: Double, dogs: Int,
                               seed: Long): Seq[Double] = {
    val rnd = new Random(seed)
    (0 until dogs).map( _ => math.round(math.abs(rnd.nextGaussian() * sigma * 100 + mean))
          .toDouble / 100)
  }
}

// COMMAND ----------

// MAGIC %md ###Listing 13.9 An overly complex and incorrectly optimized data generator
// MAGIC Here we're looking at an eager early-optimization effort. Individual portions of it are optimized for performance (the generators for individual univariate series), but the key part of the final section (the creation of the structures to be cast into a Spark DataFrame) are incredibly non-performant. Each iteration through the collection forces a scan of the Sequence from the beginning, making the total runtime complexity of this approach somehting close to O(n * log(n)).

// COMMAND ----------


object PrematureOptimization extends DogUtility {

  import spark.implicits._

  case class DogInfo(columnName: String,
                     stringData: Either[Seq[String], Seq[Boolean]],
                     sigmaData: Option[Double],
                     meanData: Option[Double],
                     valueType: String)

  def dogDataConstruct: Seq[DogInfo] = {
    Seq(
      DogInfo(
        "age",
        Left(Seq("")),
        Some(CoreData.ageSigma),
        Some(CoreData.ageMean),
        "Int"
      ),
      DogInfo(
        "weight",
        Left(Seq("")),
        Some(CoreData.weightSigma),
        Some(CoreData.weightMean),
        "Double"
      ),
      DogInfo("food", Left(CoreData.foods), Some(0.0), Some(0.0), "String"),
      DogInfo(
        "breed",
        Left(CoreData.dogBreeds),
        Some(0.0),
        Some(0.0),
        "String"
      ),
      DogInfo("good", Left(CoreData.goodness), Some(0.0), Some(0.0), "String"),
      DogInfo(
        "hungry",
        Right(CoreData.hungry),
        Some(CoreData.ageSigma),
        Some(CoreData.ageMean),
        "Boolean"
      )
    )
  }

  def generateOptimizedData(rows: Int, seed: Long): DataFrame = {

    val data = dogDataConstruct
      .map(
        x =>
          x.columnName -> {
            x.valueType match {
              case "Int" =>
                getDistributedIntData(
                  x.sigmaData.get,
                  x.meanData.get,
                  rows,
                  seed
                )
              case "Double" =>
                getDistributedDoubleData(
                  x.sigmaData.get,
                  x.meanData.get,
                  rows,
                  seed
                )
              case "String" => getDoggoData(x.stringData.left.get, rows, seed)
              case _        => getDoggoData(x.stringData.right.get, rows, seed)
            }
          } // This should be cast to an IndexedSeq in order to speed things up
      )
      .toMap
    val collection = (0 until rows).toArray
      .map(x => {
        Dogs(
          data("age")(x).asInstanceOf[Int],
          data("weight")(x).asInstanceOf[Double],
          data("food")(x).asInstanceOf[String],
          data("breed")(x).asInstanceOf[String],
          data("good")(x).asInstanceOf[String],
          data("hungry")(x).asInstanceOf[Boolean]
        )
      })
      .toSeq
    collection.toDF()
  }
  
}

// COMMAND ----------

// MAGIC %md Let's demonstrate how terrible this is for performance.

// COMMAND ----------

PrematureOptimization.generateOptimizedData(5000, 42L)

// COMMAND ----------

PrematureOptimization.generateOptimizedData(50000, 42L)

// COMMAND ----------

// MAGIC %md Setting aside the failure in proper optimization, the fact that this data generator is built early on in a project for testing of performance means that, throughout the experimentation phases of feature generation and testing, this code is going to need to be constantly updated and refactored. What happens when different data types are added? What happens when interactions between features would like to be explored? In order to generate the data within this code base, extensive refactoring to the point of needing a full rewrite will be occurring often and taking a long time to execute. 

// COMMAND ----------

// MAGIC %md ###Listing 13.10 A far more performant data generator
// MAGIC This is a bit different from the generator defined at the top of this notebook. It is both more performant than that one and certainly faster to execute than the preceding one in listing 13.9. The main difference in this one is the significantly simpler generation of the individual series (which could actually use a bit more of a refactor to make it truly useable for rapidly changing data format change testing) and only a single location to update for the generated schema definition. 

// COMMAND ----------

object ConfusingButOptimizedDogData extends DogUtility {
  
  import spark.implicits._
  
  private def generateCollections(rows: Int, seed: Long): ArrayBuffer[Seq[Any]] = {
    
    var collections = new ArrayBuffer[Seq[Any]]()
    
    collections += getDistributedIntData(CoreData.ageSigma, CoreData.ageMean, rows, seed)
    
    collections += getDistributedDoubleData(CoreData.weightSigma, CoreData.weightMean, rows, seed)
    
    Seq(CoreData.foods, CoreData.dogBreeds, CoreData.goodness, CoreData.hungry)
      .foreach(x => { collections += getDoggoData(x, rows, seed)})
    
    collections
  }
  def buildDogDF(rows: Int, seed: Long): DataFrame = {
    
    val data = generateCollections(rows, seed)
    
    data.flatMap(_.zipWithIndex)
        .groupBy(_._2).values.map( x =>
          Dogs(
            x(0)._1.asInstanceOf[Int],
            x(1)._1.asInstanceOf[Double],
            x(2)._1.asInstanceOf[String],
            x(3)._1.asInstanceOf[String],
            x(4)._1.asInstanceOf[String],
            x(5)._1.asInstanceOf[Boolean])).toSeq.toDF()
      .withColumn("hungry", when(col("hungry"), "true").otherwise("false"))
      .withColumn("hungry", when(col("breed") === "Husky", "true").otherwise(col("hungry")))
      .withColumn("good_boy_or_girl",  when(col("breed") === "Husky", "yesWhenFoodAvailable")
          .otherwise(col("good_boy_or_girl")))
  }
}

// COMMAND ----------

ConfusingButOptimizedDogData.buildDogDF(5000, 42L)

// COMMAND ----------

ConfusingButOptimizedDogData.buildDogDF(50000, 42L)

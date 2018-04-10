package com.example.bayes_classifier;

import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.Arrays;
import java.util.List;

public class BayesPipeline {


    public static void main(String... args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("BayesPipeline")
                .master("local[*]")
                .getOrCreate();

        List<Row> dataTraining = Arrays.asList(
                RowFactory.create("not to eat too much is not enough to lose weight", "health"),
                RowFactory.create("Russia is trying to invade the Ukraine", "politics"),
                RowFactory.create("do not neglect exercise", "health"),
                RowFactory.create("Syria is the main issue Obama says", "politics"),
                RowFactory.create("eat to lose weight", "health"),
                RowFactory.create("you should not eat too much", "health")
        );

        StructType schema = new StructType(new StructField[]{
                new StructField("text", DataTypes.StringType, false, Metadata.empty()),
                new StructField("label", DataTypes.StringType, false, Metadata.empty())
        });


        Dataset<Row> df = spark.createDataFrame(dataTraining, schema);

        StringIndexer indexer = new StringIndexer()
                .setInputCol("label")
                .setOutputCol("labelIndex");

        Dataset<Row> indexed = indexer.fit(df).transform(df);

        indexed.show();

        Tokenizer tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words");

        Dataset<Row> wordDf = tokenizer.transform(indexed);


        // fit a CountVectorizerModel from the corpus
        CountVectorizerModel cvModel = new CountVectorizer()
                .setInputCol("words")
                .setOutputCol("features")
//                .setVocabSize(3)
//                .setMinDF(2)
                .fit(wordDf);

//        // alternatively, define CountVectorizerModel with a-priori vocabulary
//        CountVectorizerModel cvm = new CountVectorizerModel(new String[]{"a", "b", "c"})
//                .setInputCol("text")
//                .setOutputCol("features");

        Dataset<Row> trainDf = cvModel.transform(wordDf);

        NaiveBayes nb = new NaiveBayes();
        nb.setLabelCol("labelIndex");

        NaiveBayesModel model = nb.fit(trainDf);


        List<Row> dataTest = Arrays.asList(
                RowFactory.create("Even if I eat too much, is not it possible to lose some weight", "health"));


        model.transform(cvModel.transform(tokenizer.transform(spark.createDataFrame(dataTest, schema)))).show();

    }
}

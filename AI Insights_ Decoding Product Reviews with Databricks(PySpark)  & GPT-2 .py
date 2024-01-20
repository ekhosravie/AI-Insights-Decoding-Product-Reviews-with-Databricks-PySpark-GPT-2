# Databricks notebook source
# MAGIC %sh
# MAGIC pip install transformers
# MAGIC pip install torch

# COMMAND ----------

from transformers import GPT2LMHeadModel , GPT2Tokenizer
from pyspark.sql import SparkSession 
from pyspark.sql.functions import udf , expr , col , lit , when , greatest , count 
from pyspark.sql.types import FloatType , ArrayType

# COMMAND ----------

spark = SparkSession.builder.appName('ProductReviewAnalysis').getOrCreate()

# COMMAND ----------

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# COMMAND ----------

data = [
    ("The quality of this product is excellent. I am very satisfied.",),
    ("The price of this product is very reasonable.",),
    ("The delivery of this product was done very quickly.",),
    ("Comparison of this product with similar models from other companies shows that it is the best choice.",),
    ("The quality of the product is good, but its price is higher compared to other companies.",),
    ("The delay in the delivery of this product has caused severe dissatisfaction.",),
    ("Comparison with similar products in other companies shows that it is not worth its value.",),
    ("This product exceeded my expectations. It's amazing!",),
    ("I love this product. It's fantastic.",),
    ("This product is terrible. I regret buying it.",),
]

# COMMAND ----------

columns = ['review']
df = spark.createDataFrame(data,columns)

# COMMAND ----------

@udf(ArrayType(FloatType()))
def analyze_sentiment_percentage(review: str):

    tokens = tokenizer.tokenize(review)
    review_for_gpt = " ".join(tokens)

    input_ids = tokenizer.encode(review_for_gpt, return_tensors="pt")
    logits = model(input_ids).logits
    probabilities = logits.softmax(dim=-1)[0]

    positive_score = probabilities[:, tokenizer.encode("positive")[0]].sum().item()
    negative_score = probabilities[:, tokenizer.encode("negative")[0]].sum().item()

    total = positive_score + negative_score
    positive_percentage = (positive_score / total) * 100 if total > 0 else 0.0
    negative_percentage = 100 - positive_percentage

    return [positive_percentage, negative_percentage]


# COMMAND ----------

df_with_sentiments = df.withColumn("sentiment_percentages", analyze_sentiment_percentage(df["review"])).cache()

# COMMAND ----------

df_with_sentiments = df_with_sentiments.withColumn("positive_percentage", expr("sentiment_percentages[0]"))
df_with_sentiments = df_with_sentiments.withColumn("negative_percentage", expr("sentiment_percentages[1]"))

# COMMAND ----------

df_with_sentiments = df_with_sentiments.withColumn("neutral_percentage", lit(0.0))

# COMMAND ----------

neutral_expr = (
    (col("positive_percentage") >= 40.0) &
    (col("positive_percentage") <= 60.0) &
    (col("negative_percentage") >= 40.0) &
    (col("negative_percentage") <= 60.0)
)

# COMMAND ----------

df_with_sentiments = df_with_sentiments.withColumn(
    "neutral_percentage",
    when(neutral_expr, 100.0).otherwise(0.0)
)

# COMMAND ----------

df_with_sentiments = df_with_sentiments.withColumn(
    "max_sentiment_percentage",
    greatest(col("positive_percentage"), col("negative_percentage"), col("neutral_percentage"))
)

# COMMAND ----------

df_with_sentiments = df_with_sentiments.withColumn(
    "Sentiment_Category",
    when(col("max_sentiment_percentage") == col("positive_percentage"), "Positive")
    .when(col("max_sentiment_percentage") == col("negative_percentage"), "Negative")
    .when(col("max_sentiment_percentage") == col("neutral_percentage"), "Neutral")
    .otherwise("Unknown")
)

# COMMAND ----------

df_with_sentiments.select("review", "positive_percentage", "negative_percentage", "neutral_percentage", "Sentiment_Category").show(truncate=False)

# COMMAND ----------

total_reviews = df_with_sentiments.count()
positive_reviews_count = df_with_sentiments.filter(col("Sentiment_Category") == "Positive").count()
negative_reviews_count = df_with_sentiments.filter(col("Sentiment_Category") == "Negative").count()
neutral_reviews_count = df_with_sentiments.filter(col("Sentiment_Category") == "Neutral").count()

# COMMAND ----------

positive_review_percentage = (positive_reviews_count / total_reviews) * 100
negative_review_percentage = (negative_reviews_count / total_reviews) * 100
neutral_review_percentage = (neutral_reviews_count / total_reviews) * 100

# COMMAND ----------

result_data = [
    ('Positive' , positive_review_percentage),
    ('Negative' , negative_review_percentage),
    ('Neutral' , neutral_review_percentage)
]
result_columns = ['Sentiment_Type' , 'Percentage']
result_df = spark.createDataFrame(result_data, result_columns )
 

# COMMAND ----------

result_df.show()

# COMMAND ----------



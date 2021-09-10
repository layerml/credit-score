/*
    Empty Project Example
    `featureset1_feature1` Feature
*/
-- Note: It is a good practice to use the same names for your features in the code below and dataset.yaml file
-- Ensure that you have the same ID column across all of your sql features. Layer joins your singular features using that ID column.

SELECT
ID_column,

--Your SQL query goes here

AS featureset1_feature1
FROM your_table_name
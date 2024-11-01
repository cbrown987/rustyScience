use serde::Deserialize;
use std::error::Error;
use csv::ReaderBuilder;

#[derive(Debug, Deserialize)]
pub struct IrisRecord {
    pub sepal_length: f64,
    pub sepal_width: f64,
    pub petal_length: f64,
    pub petal_width: f64,
    pub species: String,
}

pub fn load_iris_data<F, L>( select_features: F, map_label: L, ) -> Result<(Vec<Vec<f64>>, Vec<i64>), Box<dyn Error>>
where
    F: Fn(&IrisRecord) -> Vec<f64>,
    L: Fn(&str) -> i64,
{
    let csv_data = include_str!("datasets/iris.csv");
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(csv_data.as_bytes());

    let mut dataset = Vec::new();
    let mut labels = Vec::new();

    for result in rdr.deserialize() {
        let record: IrisRecord = result?;
        let features = select_features(&record);
        dataset.push(features);
        let label = map_label(&record.species);
        labels.push(label.into());
    }
    Ok((dataset, labels))
}


pub(crate) fn load_iris_data_labeled() -> (Vec<Vec<f64>>, Vec<i64>) {
    let select_features = |record: &IrisRecord| vec![record.sepal_length, record.sepal_width];
    let map_label = |species: &str| match species {
        "Setosa" => 1i64,
        "Versicolor" => 2i64,
        "Virginica" => 3i64,
        _ => panic!("Unknown species: {}", species),
    };

    load_iris_data(select_features, map_label).expect("Failed to load data")
}


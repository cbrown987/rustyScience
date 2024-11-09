use crate::data::datasets::data_structs::{BreastCancerData, HousingData, IrisData};
use crate::data::datasets::load_csv::{read_breast_cancer_csv, read_housing_csv, read_iris_csv};

mod data_structs;
mod load_csv;

/// Loads the Iris dataset.
///
/// # Returns
///
/// * `IrisData` - A structure containing data from the Iris dataset.
///
/// If the CSV file cannot be read, an empty `IrisData` is returned and an error message is printed to stderr.
///
/// # Errors
/// Prints an error message to stderr if the CSV file cannot be read.
///
/// # Example
///
/// ```
/// use rusty_science::data::load_iris;
/// let iris_data = load_iris();
/// ```
pub fn load_iris() -> IrisData {
    let data_path = "src/data/datasets/data_files/iris.csv";
    match read_iris_csv(data_path) {
        Ok(data) => {
            data
        },
        Err(e) => {
            eprintln!("Error reading CSV file: {}", e);
            IrisData::new()
        },
    }}

/// Loads the Breast Cancer dataset.
///
/// # Returns
///
/// * `BreastCancerData` - A structure containing data from the Breast Cancer dataset.
///
/// If the CSV file cannot be read, an empty `BreastCancerData` is returned and an error message is printed to stderr.
///
/// # Errors
/// Prints an error message to stderr if the CSV file cannot be read.
///
/// # Example
///
/// ```
/// use rusty_science::data::load_brest_cancer;
/// let breast_cancer_data = load_brest_cancer();
/// ```
pub fn load_brest_cancer() -> BreastCancerData {
    let data_path = "src/data/datasets/data_files/brest-cancer.csv";
    match read_breast_cancer_csv(data_path) {
        Ok(data) => {
            data 
        },
        Err(e) => {
            eprintln!("Error reading CSV file: {}", e);
            BreastCancerData::new()
        }
    }
}

/// Loads the Housing dataset.
///
/// # Returns
///
/// * `HousingData` - A structure containing data from the Housing dataset.
///
/// If the CSV file cannot be read, an empty `HousingData` is returned and an error message is printed to stderr.
///
/// # Errors
/// Prints an error message to stderr if the CSV file cannot be read.
///
/// # Example
///
/// ```
/// use rusty_science::data::load_housing;
/// let housing_data = load_housing();
/// ```
pub fn load_housing() -> HousingData {
    let data_path = "src/data/datasets/data_files/housing-data.csv";
    match read_housing_csv(data_path) {
        Ok(data) => {
            data
        },
        Err(e) => {
            eprintln!("Error reading CSV file: {}", e);
            HousingData::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_iris_data_exists() {
        let iris_data = load_iris();    
        assert!(iris_data.sepal_length.len() > 0, "No data was loaded for Iris dataset");
        assert_eq!(iris_data.sepal_width.len(), iris_data.sepal_length.len(), "Mismatch in sepal_width and sepal_length record counts");
        assert_eq!(iris_data.species.len(), iris_data.sepal_length.len(), "Mismatch in species and sepal_length record counts");

        if iris_data.sepal_length.len() > 0 {
            assert_eq!(iris_data.species[0], "setosa", "The species of the first Iris record is incorrect");
        }
    }

    #[test]
    fn test_load_breast_cancer_data_exists() {
        let breast_cancer_data = load_brest_cancer();

        assert!(breast_cancer_data.ids.len() > 0, "No data was loaded for Breast Cancer dataset");
        assert_eq!(breast_cancer_data.diagnoses.len(), breast_cancer_data.ids.len(), "Mismatch in diagnoses and ids record counts");
        assert_eq!(breast_cancer_data.radius_mean.len(), breast_cancer_data.ids.len(), "Mismatch in radius_mean and ids record counts");

        if breast_cancer_data.ids.len() > 0 {
            assert_eq!(breast_cancer_data.diagnoses[0], "M", "The diagnosis of the first Breast Cancer record is incorrect");
        }
    }
    
    #[test]
    fn test_housing_data_exists() {
        let housing_data = load_housing();
        println!("{:?}", housing_data)
    }
}


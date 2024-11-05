use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use crate::data::datasets::data_structs::{BreastCancerData, HousingData, IrisData};

fn parse_and_push_f32(value: &str, idx: usize, field_name: &str, target_vec: &mut Vec<f32>) {
    match value.trim().parse::<f32>() {
        Ok(val) => target_vec.push(val),
        Err(e) => {
            if value == "NA"{
                target_vec.push(-1f32);
                return;
            }
            eprintln!(
                "Warning: Invalid value '{}' for field '{}' on line {}: {:?}",
                value, field_name, idx + 1, e
            );
        }
    }
}

fn parse_and_push_f64(value: &str, idx: usize, field_name: &str, target_vec: &mut Vec<f64>) {
    match value.trim().parse::<f64>() {
        Ok(val) => target_vec.push(val),
        
        Err(e) => {
            if value == "NA"{
                target_vec.push(-1f64);
                return;
            }
            eprintln!(
                "Warning: Invalid value '{}' for field '{}' on line {}: {:?}",
                value, field_name, idx + 1, e
            );
        }
    }
}


pub(crate) fn read_iris_csv(filename: &str) -> io::Result<IrisData> {
    let mut data = IrisData::new();

    let path = Path::new(filename);
    let file = File::open(&path).expect("Could not open or find data file");
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    lines.next(); 
    
    for (idx, line) in lines.enumerate() {
        let line = line?;

        let fields: Vec<&str> = line.split(',').map(|f| f.trim_matches('\"')).collect();

        parse_and_push_f32(&fields[0], idx, "sepal_length", &mut data.sepal_length);
        parse_and_push_f32(&fields[1], idx, "sepal_width", &mut data.sepal_width);
        parse_and_push_f32(&fields[2], idx, "petal_length", &mut data.petal_length);
        parse_and_push_f32(&fields[3], idx, "petal_width", &mut data.petal_width);

        data.species.push(fields[4].trim().to_lowercase());
    }

    Ok(data)
}



pub(crate) fn read_breast_cancer_csv(filename: &str) -> io::Result<BreastCancerData> {

    let path = Path::new(filename);
    let file = File::open(&path).expect("Could not open or find data file");
    let reader = BufReader::new(file);
    let lines = reader.lines();
    
    let mut data = BreastCancerData::new();

    for (idx, line) in lines.enumerate() {
        let line = line?;

        if idx == 0 && line.starts_with("\"id\"") {
            continue;
        }

        let fields: Vec<&str> = line.split(',').collect();

        match fields[0].trim().parse::<u32>() {
            Ok(id) => data.ids.push(id),
            Err(e) => {
                eprintln!("Warning: Invalid ID value '{}' on line {}: {:?}", fields[0], idx + 1, e);
                continue; // Skip this row if parsing fails
            }
        }

        data.diagnoses.push(fields[1].trim().to_string());

        parse_and_push_f32(&fields[2], idx, "radius_mean", &mut data.radius_mean);
        parse_and_push_f32(&fields[3], idx, "texture_mean", &mut data.texture_mean);
        parse_and_push_f32(&fields[4], idx, "perimeter_mean", &mut data.perimeter_mean);
        parse_and_push_f32(&fields[5], idx, "area_mean", &mut data.area_mean);
        parse_and_push_f32(&fields[6], idx, "smoothness_mean", &mut data.smoothness_mean);
        parse_and_push_f32(&fields[7], idx, "compactness_mean", &mut data.compactness_mean);
        parse_and_push_f32(&fields[8], idx, "concavity_mean", &mut data.concavity_mean);
        parse_and_push_f32(&fields[9], idx, "concave_points_mean", &mut data.concave_points_mean);
        parse_and_push_f32(&fields[10], idx, "symmetry_mean", &mut data.symmetry_mean);
        parse_and_push_f32(&fields[11], idx, "fractal_dimension_mean", &mut data.fractal_dimension_mean);
        parse_and_push_f32(&fields[12], idx, "radius_se", &mut data.radius_se);
        parse_and_push_f32(&fields[13], idx, "texture_se", &mut data.texture_se);
        parse_and_push_f32(&fields[14], idx, "perimeter_se", &mut data.perimeter_se);
        parse_and_push_f32(&fields[15], idx, "area_se", &mut data.area_se);
        parse_and_push_f32(&fields[16], idx, "smoothness_se", &mut data.smoothness_se);
        parse_and_push_f32(&fields[17], idx, "compactness_se", &mut data.compactness_se);
        parse_and_push_f32(&fields[18], idx, "concavity_se", &mut data.concavity_se);
        parse_and_push_f32(&fields[19], idx, "concave_points_se", &mut data.concave_points_se);
        parse_and_push_f32(&fields[20], idx, "symmetry_se", &mut data.symmetry_se);
        parse_and_push_f32(&fields[21], idx, "fractal_dimension_se", &mut data.fractal_dimension_se);
        parse_and_push_f32(&fields[22], idx, "radius_worst", &mut data.radius_worst);
        parse_and_push_f32(&fields[23], idx, "texture_worst", &mut data.texture_worst);
        parse_and_push_f32(&fields[24], idx, "perimeter_worst", &mut data.perimeter_worst);
        parse_and_push_f32(&fields[25], idx, "area_worst", &mut data.area_worst);
        parse_and_push_f32(&fields[26], idx, "smoothness_worst", &mut data.smoothness_worst);
        parse_and_push_f32(&fields[27], idx, "compactness_worst", &mut data.compactness_worst);
        parse_and_push_f32(&fields[28], idx, "concavity_worst", &mut data.concavity_worst);
        parse_and_push_f32(&fields[29], idx, "concave_points_worst", &mut data.concave_points_worst);
        parse_and_push_f32(&fields[30], idx, "symmetry_worst", &mut data.symmetry_worst);
        parse_and_push_f32(&fields[31], idx, "fractal_dimension_worst", &mut data.fractal_dimension_worst);
    }

    Ok(data)
}

pub(crate) fn read_housing_csv(filename: &str) -> io::Result<HousingData> {
    let path = Path::new(filename);
    let file = File::open(&path).expect("Could not open or find data file");
    let reader = BufReader::new(file);
    let lines = reader.lines();
    
    let mut data = HousingData::new();

    for (idx, line) in lines.enumerate() {
        let line = line?;

        if idx == 0 {
            continue;
        }

        let fields: Vec<&str> = line.split(',').collect();

        if fields.len() < 14 {
            eprintln!("Warning: Insufficient fields on line {}: {:?}", idx + 1, line);
            continue; // Skip this row if it doesn't have enough fields
        }

        // Use the provided parse_and_push_f32 function for parsing
        parse_and_push_f64(fields[0], idx, "CRIM", &mut data.crim);
        parse_and_push_f64(fields[1], idx, "ZN", &mut data.zn);
        parse_and_push_f64(fields[2], idx, "INDUS", &mut data.indus);
        parse_and_push_f64(fields[3], idx, "CHAS", &mut data.chas);
        parse_and_push_f64(fields[4], idx, "NOX", &mut data.nox);
        parse_and_push_f64(fields[5], idx, "RM", &mut data.rm);
        parse_and_push_f64(fields[6], idx, "AGE", &mut data.age);
        parse_and_push_f64(fields[7], idx, "DIS", &mut data.dis);
        parse_and_push_f64(fields[8], idx, "RAD", &mut data.rad);
        parse_and_push_f64(fields[9], idx, "TAX", &mut data.tax);
        parse_and_push_f64(fields[10], idx, "PTRATIO", &mut data.ptratio);
        parse_and_push_f64(fields[11], idx, "B", &mut data.b);
        parse_and_push_f64(fields[12], idx, "LSTAT", &mut data.lstat);
        parse_and_push_f64(fields[13], idx, "MEDV", &mut data.medv);
    }

    Ok(data)
}



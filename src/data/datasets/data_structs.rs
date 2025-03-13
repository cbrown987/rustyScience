#[derive(Debug)]
pub struct IrisData {
    pub sepal_length: Vec<f32>,
    pub sepal_width: Vec<f32>,
    pub petal_length: Vec<f32>,
    pub petal_width: Vec<f32>,
    pub species: Vec<String>,
}

impl IrisData {
    pub fn new() -> Self {
        Self {
            sepal_length: Vec::new(),
            sepal_width: Vec::new(),
            petal_length: Vec::new(),
            petal_width: Vec::new(),
            species: Vec::new(),
        }
    }
    
    pub fn to_numerical_labels(&self) -> (Vec<Vec<f64>>, Vec<i64>) {
        let mut data = Vec::new();
        let mut labels = Vec::new();

        for i in 0..self.sepal_length.len() {
            let features = vec![
                self.sepal_length[i] as f64,
                self.sepal_width[i] as f64,
                self.petal_length[i] as f64,
                self.petal_width[i] as f64,
            ];
            data.push(features);

            // Convert species name to a label
            let label = match self.species[i].as_str() {
                "setosa" => 0,
                "versicolor" => 1,
                "virginica" => 2,
                _ => panic!("Unknown species label"),
            };
            labels.push(label);
        }
        (data, labels)
    }
}


#[derive(Debug)]
pub struct BreastCancerData {
    pub ids: Vec<u32>,
    pub diagnoses: Vec<String>,
    pub radius_mean: Vec<f32>,
    pub texture_mean: Vec<f32>,
    pub perimeter_mean: Vec<f32>,
    pub area_mean: Vec<f32>,
    pub smoothness_mean: Vec<f32>,
    pub compactness_mean: Vec<f32>,
    pub concavity_mean: Vec<f32>,
    pub concave_points_mean: Vec<f32>,
    pub symmetry_mean: Vec<f32>,
    pub fractal_dimension_mean: Vec<f32>,
    pub radius_se: Vec<f32>,
    pub texture_se: Vec<f32>,
    pub perimeter_se: Vec<f32>,
    pub area_se: Vec<f32>,
    pub smoothness_se: Vec<f32>,
    pub compactness_se: Vec<f32>,
    pub concavity_se: Vec<f32>,
    pub concave_points_se: Vec<f32>,
    pub symmetry_se: Vec<f32>,
    pub fractal_dimension_se: Vec<f32>,
    pub radius_worst: Vec<f32>,
    pub texture_worst: Vec<f32>,
    pub perimeter_worst: Vec<f32>,
    pub area_worst: Vec<f32>,
    pub smoothness_worst: Vec<f32>,
    pub compactness_worst: Vec<f32>,
    pub concavity_worst: Vec<f32>,
    pub concave_points_worst: Vec<f32>,
    pub symmetry_worst: Vec<f32>,
    pub fractal_dimension_worst: Vec<f32>,
}

impl BreastCancerData {
    pub fn new() -> Self {
        Self {
            ids: Vec::new(),
            diagnoses: Vec::new(),
            radius_mean: Vec::new(),
            texture_mean: Vec::new(),
            perimeter_mean: Vec::new(),
            area_mean: Vec::new(),
            smoothness_mean: Vec::new(),
            compactness_mean: Vec::new(),
            concavity_mean: Vec::new(),
            concave_points_mean: Vec::new(),
            symmetry_mean: Vec::new(),
            fractal_dimension_mean: Vec::new(),
            radius_se: Vec::new(),
            texture_se: Vec::new(),
            perimeter_se: Vec::new(),
            area_se: Vec::new(),
            smoothness_se: Vec::new(),
            compactness_se: Vec::new(),
            concavity_se: Vec::new(),
            concave_points_se: Vec::new(),
            symmetry_se: Vec::new(),
            fractal_dimension_se: Vec::new(),
            radius_worst: Vec::new(),
            texture_worst: Vec::new(),
            perimeter_worst: Vec::new(),
            area_worst: Vec::new(),
            smoothness_worst: Vec::new(),
            compactness_worst: Vec::new(),
            concavity_worst: Vec::new(),
            concave_points_worst: Vec::new(),
            symmetry_worst: Vec::new(),
            fractal_dimension_worst: Vec::new(),
        }
    }
    pub fn to_numerical_labels(&self) -> (Vec<Vec<f32>>, Vec<i32>) {
        let mut data = Vec::new();
        let mut labels = Vec::new();

        for i in 0..self.diagnoses.len() {
            let features = vec![
                self.radius_mean[i],
                self.texture_mean[i],
                self.perimeter_mean[i],
                self.area_mean[i],
                self.smoothness_mean[i],
                self.compactness_mean[i],
                self.concavity_mean[i],
                self.concave_points_mean[i],
                self.symmetry_mean[i],
                self.fractal_dimension_mean[i],
                self.radius_se[i],
                self.texture_se[i],
                self.perimeter_se[i],
                self.area_se[i],
                self.smoothness_se[i],
                self.compactness_se[i],
                self.concavity_se[i],
                self.concave_points_se[i],
                self.symmetry_se[i],
                self.fractal_dimension_se[i],
                self.radius_worst[i],
                self.texture_worst[i],
                self.perimeter_worst[i],
                self.area_worst[i],
                self.smoothness_worst[i],
                self.compactness_worst[i],
                self.concavity_worst[i],
                self.concave_points_worst[i],
                self.symmetry_worst[i],
                self.fractal_dimension_worst[i],
            ];
            data.push(features);

            let label = match self.diagnoses[i].as_str() {
                "M" => 1,  // Malignant
                "B" => 0,  // Benign
                _ => panic!("Unknown diagnosis label"),
            };
            labels.push(label);
        }
        (data, labels)
    }
}

#[derive(Debug)]
pub struct HousingData {
    pub crim: Vec<f64>,
    pub zn: Vec<f64>,
    pub indus: Vec<f64>,
    pub chas: Vec<f64>,
    pub nox: Vec<f64>,
    pub rm: Vec<f64>,
    pub age: Vec<f64>,
    pub dis: Vec<f64>,
    pub rad: Vec<f64>,
    pub tax: Vec<f64>,
    pub ptratio: Vec<f64>,
    pub b: Vec<f64>,
    pub lstat: Vec<f64>,
    pub medv: Vec<f64>,
}

impl HousingData {
    pub fn new() -> Self {
        HousingData {
            crim: Vec::new(),
            zn: Vec::new(),
            indus: Vec::new(),
            chas: Vec::new(),
            nox: Vec::new(),
            rm: Vec::new(),
            age: Vec::new(),
            dis: Vec::new(),
            rad: Vec::new(),
            tax: Vec::new(),
            ptratio: Vec::new(),
            b: Vec::new(),
            lstat: Vec::new(),
            medv: Vec::new(),
        }
    }

    pub fn to_numerical_values(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut data = Vec::new();
        let mut labels = Vec::new();

        for i in 0..self.crim.len() {
            let row = vec![
                self.crim[i], self.zn[i], self.indus[i], self.chas[i],
                self.nox[i], self.rm[i], self.age[i], self.dis[i],
                self.rad[i], self.tax[i], self.ptratio[i], self.b[i],
                self.lstat[i]
            ];

            let has_nan = row.iter().any(|&x| x.is_nan());
            let label_is_nan = self.medv[i].is_nan();

            if !has_nan && !label_is_nan {
                data.push(row);
                labels.push(self.medv[i]);
            }
        }

        (data, labels)
    }
}
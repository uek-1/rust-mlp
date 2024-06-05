use super::{Tensor, ValidNumber};

#[derive(Clone, Debug, PartialEq)]
pub enum Activation {
    None,
    Sigmoid,
    Relu,
    Softmax,
}

impl Activation {
    pub fn activate_vec<T: ValidNumber<T>>(&self, preactivations: Vec<T>) -> Vec<T> {
        preactivations
            .iter()
            .map(|num| match self {
                Activation::None => *num,
                Activation::Relu => Self::relu(*num),
                Activation::Sigmoid => Self::sigmoid(*num),
                Activation::Softmax => Self::softmax(*num, preactivations.clone()),
            })
            .collect()
    }

    pub fn activate_tensor<T: ValidNumber<T>>(
        &self,
        preactivations: &Tensor<T>,
    ) -> Result<Tensor<T>, ()> {
        match preactivations.rank() {
            1 => self.activate_tensor1d(preactivations),
            2 => self.activate_tensor2d(preactivations),
            _ => self.activate_tensor_naive(preactivations),
        }
    }

    pub fn activate_tensor1d<T: ValidNumber<T>>(
        &self,
        preactivations: &Tensor<T>,
    ) -> Result<Tensor<T>, ()> {
        if preactivations.rank() != 1 {
            return Err(());
        }

        let new_data: Vec<T> = self.activate_vec(preactivations.data.clone());
        Ok(Tensor::from(new_data))
    }

    pub fn activate_tensor2d<T: ValidNumber<T>>(
        &self,
        preactivations: &Tensor<T>,
    ) -> Result<Tensor<T>, ()> {
        if preactivations.rank() != 2 || preactivations.shape()[1] != 1 {
            return Err(());
        }

        let new_data: Vec<Vec<T>> = preactivations
            .as_columns()
            .iter()
            .map(|col| self.activate_tensor1d(&col))
            .map(|res| match res {
                Ok(col) => Ok(col.data),
                Err(_) => Err(()),
            })
            .collect::<Result<Vec<Vec<T>>, ()>>()?;

        Ok(Tensor::<T>::from(new_data).transposed())
    }

    pub fn activate_tensor_naive<T: ValidNumber<T>>(
        &self,
        preactivations: &Tensor<T>,
    ) -> Result<Tensor<T>, ()> {
        let new_data = preactivations
            .data
            .iter()
            .map(|x| match self {
                Activation::None => Ok(*x),
                Activation::Sigmoid => Ok(Self::sigmoid(*x)),
                Activation::Relu => Ok(Self::relu(*x)),
                Activation::Softmax => Err(()),
            })
            .collect::<Result<Vec<T>, ()>>()?;

        Ok(Tensor {
            shape: preactivations.shape().to_vec(),
            data: new_data,
        })
    }

    //TODO : check for errors here!
    pub fn derivative<T: ValidNumber<T>>(&self, preactivations: &Tensor<T>) -> Tensor<T> {
        match preactivations.rank() {
            2 => self.derivative2d(preactivations),
            _ => self.derivative_naive(preactivations),
        }
    }

    fn derivative2d<T: ValidNumber<T>>(&self, preactivations: &Tensor<T>) -> Tensor<T> {
        let data = preactivations
            .as_columns()
            .iter()
            .map(|x| {
                x.data
                    .iter()
                    .enumerate()
                    .map(|(num, y)| match self {
                        Activation::Sigmoid => Self::sigmoid_derivative(*y),
                        Activation::Relu => Self::relu_derivative(*y),
                        Activation::None => vec![T::from(1.0)],
                        Activation::Softmax => Self::softmax_derivative(num, x.data.clone()),
                    })
                    .collect::<Vec<Vec<T>>>()
            })
            .collect::<Vec<Vec<Vec<T>>>>();

        // println!("{:?}", Tensor::from(data[0].clone()));

        Tensor::from(data[0].clone())
    }

    fn derivative_naive<T: ValidNumber<T>>(&self, preactivations: &Tensor<T>) -> Tensor<T> {
        let data = preactivations
            .data
            .iter()
            .map(|x| match self {
                Activation::None => T::from(1.0),
                Activation::Sigmoid => Self::sigmoid_derivative(*x)[0],
                Activation::Relu => Self::relu_derivative(*x)[0],
                Activation::Softmax => panic!("softmax undefined for tensors not rank 2"),
            })
            .collect();

        Tensor {
            shape: preactivations.shape().clone(),
            data,
        }
    }

    pub fn activate_num<T: ValidNumber<T>>(&self, num: T) -> T {
        match self {
            Activation::Sigmoid => Self::sigmoid(num),
            Activation::Relu => Self::relu(num),
            Activation::Softmax => todo!("SOFTMAX ACTIVATION HASN'T BEEN IMPLMENETED"),
            Activation::None => num,
        }
    }

    fn sigmoid<T: ValidNumber<T>>(num: T) -> T {
        T::from(1.0 / (1.0 + (-1.0 * num.into()).exp()))
    }

    fn sigmoid_derivative<T: ValidNumber<T>>(num: T) -> Vec<T> {
        let delta = Self::sigmoid(num);
        vec![delta * (T::from(1.0) - delta)]
    }

    fn relu<T: ValidNumber<T>>(num: T) -> T {
        if num < T::from(0.0) {
            return T::from(0.0);
        }
        num
    }

    fn relu_derivative<T: ValidNumber<T>>(num: T) -> Vec<T> {
        if num < T::from(0.0) {
            return vec![T::from(0.0)];
        }
        vec![T::from(1.0)]
    }

    pub fn softmax<T: ValidNumber<T>>(num: T, classes: Vec<T>) -> T {
        // To saturate to zero instead of infinity, subtract max(classes) from top and bottom.
        let max_classes: f64 = classes
            .iter()
            .map(|x| (*x).into())
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();

        // let max_classes = 0.0;

        let top: f64 = (num.into() - max_classes).exp();
        let bottom: f64 = classes.iter().fold(0.0, |sum: f64, x: &T| {
            sum + ((*x).into() - max_classes).exp()
        });
        let out = T::from(top / bottom);

        //println!("softmax ({:?}, {:?}) = {:?}", top, bottom, out);
        //println!("max_classes - {max_classes}");

        // println!("{out:?}");
        // assert!(out <= T::from(1.0));
        out
    }

    fn softmax_derivative<T: ValidNumber<T>>(neuron: usize, classes: Vec<T>) -> Vec<T> {
        let smax_input = Self::softmax(classes[neuron], classes.clone());

        (0..classes.len())
            .map(|idx| {
                let smax_output = Self::softmax(classes[idx], classes.clone());
                let delta = match idx == neuron {
                    true => T::from(1.0),
                    false => T::from(0.0),
                };

                // println!("si {smax_input:?}, so {smax_output:?}, delta {delta:?}, idx {idx:?} neuron {neuron}");

                smax_output * (delta - smax_input)
            })
            .collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn softmax_test() {
        let items = vec![1.0f64, 2.0, 3.0];
        let res = Activation::softmax(items[0], items);
        assert!((res - 0.09003057).abs() < 0.001)
        // close eneough.
    }

    #[test]
    fn softmax_derivative_test() {
        let classes = vec![3.0f64, 4.0, 5.0];
        let res = Activation::softmax_derivative(0, classes);
        assert!((res[0] - 0.08192506).abs() < 0.001);
    }

    #[test]
    fn softmax_derivative_vector_test() {
        let classes = vec![1.0f64, 2.0, 3.0];
        let res = Activation::softmax_derivative(0, classes);
        println!("{res:?}");

        let error = res
            .into_iter()
            .zip([0.08192, -0.02203, -0.059892])
            .fold(0.0, |err, (res, expec)| err + (expec - res).abs());

        assert!(error < 0.001)
    }

    //#[test]
    // fn total_softmax_derivative_test() {
    //     let preactivations = vec![1.0, 2.0, 3.0];
    //     let res = Activation::Softmax.derivative(vec![preactivations].transposed());
    //     println!("{:?}", res);

    //     let truth = vec![
    //         vec![0.0819, -0.0220, -0.0598],
    //         vec![-0.0220, 0.1848, -0.1628],
    //         vec![-0.0598, -0.1628, 0.2226],
    //     ];

    //     for i in 0..truth.len() {
    //         for j in 0..truth[i].len() {
    //             assert!((res[i][j] - truth[i][j]).abs() < 0.01);
    //         }
    //     }
    // }
}

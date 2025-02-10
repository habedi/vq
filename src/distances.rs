use crate::vector::Real;

pub enum Distance {
    SquaredEuclidean,
    Euclidean,
    CosineDistance,
    Manhattan,
    Chebyshev,
    Minkowski(f64),
    Hamming,
}

impl Distance {
    pub fn compute<T: Real>(&self, a: &[T], b: &[T]) -> T {
        assert_eq!(a.len(), b.len());
        match self {
            Distance::SquaredEuclidean => a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| {
                    let diff = x - y;
                    diff * diff
                })
                .fold(T::zero(), |acc, val| acc + val),
            Distance::Euclidean => {
                let sum = a
                    .iter()
                    .zip(b.iter())
                    .map(|(&x, &y)| {
                        let diff = x - y;
                        diff * diff
                    })
                    .fold(T::zero(), |acc, val| acc + val);
                sum.sqrt()
            }
            Distance::CosineDistance => {
                let dot = a
                    .iter()
                    .zip(b.iter())
                    .map(|(&x, &y)| x * y)
                    .fold(T::zero(), |acc, val| acc + val);
                let norm_a = a
                    .iter()
                    .map(|&x| x * x)
                    .fold(T::zero(), |acc, val| acc + val)
                    .sqrt();
                let norm_b = b
                    .iter()
                    .map(|&x| x * x)
                    .fold(T::zero(), |acc, val| acc + val)
                    .sqrt();
                if norm_a == T::zero() || norm_b == T::zero() {
                    T::one()
                } else {
                    T::one() - dot / (norm_a * norm_b)
                }
            }
            Distance::Manhattan => a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| (x - y).abs())
                .fold(T::zero(), |acc, val| acc + val),
            Distance::Chebyshev => a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| (x - y).abs())
                .fold(T::zero(), |acc, val| if val > acc { val } else { acc }),
            Distance::Minkowski(p) => {
                let p_val = T::from_f64(*p);
                let sum = a
                    .iter()
                    .zip(b.iter())
                    .map(|(&x, &y)| (x - y).abs().powf(p_val))
                    .fold(T::zero(), |acc, val| acc + val);
                sum.powf(T::one() / p_val)
            }
            Distance::Hamming => a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| if x == y { T::zero() } else { T::one() })
                .fold(T::zero(), |acc, val| acc + val),
        }
    }
}

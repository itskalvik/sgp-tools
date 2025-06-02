import tensorflow as tf


def jitter_fn(cov, jitter=1e-6):
    """
    Add jitter to the diagonal of the covariance matrix.
    This is useful for numerical stability when computing the
    determinant of the covariance matrix.
    """
    cov = tf.linalg.set_diag(cov, tf.linalg.diag_part(cov) + jitter)
    return cov

class MI:
    """
    Mutual information between X_train and X.

    Jitter is added to the diagonal of the covariance matrices to ensure
    numerical stability.
    """
    def __init__(self, kernel, X_train,
                 jitter=1e-6):
        self.kernel = kernel
        self.X_train = X_train
        self.jitter = lambda x: jitter_fn(x, jitter=jitter)

    def __call__(self, X):
        A = self.kernel(self.X_train)
        D = self.kernel(X)
        M = self.kernel(tf.concat([self.X_train, X], axis=0))

        A_det = tf.math.log(tf.linalg.det(self.jitter(A)))
        D_det = tf.math.log(tf.linalg.det(self.jitter(D)))
        M_det = tf.math.log(tf.linalg.det(self.jitter(M)))

        mi = A_det + D_det - M_det

        return mi
    
class SLogMI(MI):
    """
    Mutual information between X_train and X.

    The log-determinant of the covariance matrices is computed using
    the slogdet method. This is more numerically stable than
    computing the determinant directly and then taking the log.

    Jitter is also added to the diagonal of the covariance matrices to 
    to further ensure numerical stability.
    """
    def __call__(self, X):
        A = self.kernel(self.X_train)
        D = self.kernel(X)
        M = self.kernel(tf.concat([self.X_train, X], axis=0))

        _, A_det = tf.linalg.slogdet(self.jitter(A))
        _, D_det = tf.linalg.slogdet(self.jitter(D))
        _, M_det = tf.linalg.slogdet(self.jitter(M))

        mi = A_det + D_det - M_det

        return mi
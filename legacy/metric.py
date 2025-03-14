import tensorflow as tf

class CSImetric(tf.keras.metrics.Metric):
    def __init__(self, name="csi", **kwargs):
        super(CSImetric,self).__init__(name=name, **kwargs)
        self.H = self.add_weight(name="H", initializer="zeros")
        self.F = self.add_weight(name="F", initializer="zeros")

    def update_state(self,y_true,y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        # y_pred.shpae : (n, 1)
        # y_true.shape : (n,1)
        t_idx = (y_pred == y_true)
        self.H.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(t_idx, y_true!= 0),tf.float32)))
        self.F.assign_add(tf.reduce_sum(tf.cast(tf.logical_not(t_idx) , tf.float32)))
        
    def result(self):
        return self.H/(self.H+self.F+tf.keras.backend.epsilon())
    def reset_states(self):
        self.H.assign(0.0)
        self.F.assign(0.0)
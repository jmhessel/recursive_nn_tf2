'''
Child sum LSTM from

Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks
Kai Sheng Tai, Richard Socher, Christopher D. Manning
https://arxiv.org/pdf/1503.00075.pdf

'''
import argparse
import recursive_nn
import numpy as np
import tensorflow as tf


class ChildSumLSTMTreeLayer(recursive_nn.SimpleTreeLayer):
    ''' Note that self.dim is actually 3*the inside dimension,
    because this model outputs both h and o.'''
    def build(self, input_shape):
        assert self.dim % 3 == 0, 'Dim must be a multiple of 3!'

        self.inner_dim = int(self.dim / 3)
        if len(input_shape) == 2:
            self.feature_dim = input_shape[-1][-1]
            #stacked W * x transformation
            self.input_trans_W = self.add_weight(
                'Wx',
                shape=(4 * self.inner_dim, self.feature_dim),
                initializer='glorot_uniform')

        #stacked U * htilde transformation
        self.trans_U = self.add_weight(
            'Uhtilde',
            shape=(3 * self.inner_dim, self.inner_dim),
            initializer='orthogonal')

        #Uf * h
        self.Uf = self.add_weight(
            'Uf',
            shape=(self.inner_dim, self.inner_dim),
            initializer='orthogonal')

        #stacked biases
        self.b = self.add_weight(
            'b',
            shape = (self.inner_dim * 4),
            initializer='zeros')


    def _combine_inner(self, reprs, features):
        '''Combination function. Outputs three things, concatenated:

        [h_j, c_j, o_j]

        according to the equations in S 3.1.

        - reprs a list of dim lengthed vectors from child nodes
        - features is a dim-lengthed input feature vector for this node.

        a few conventions:
          - if reprs is an empty list, you're at a leaf node, and
            should proceed appropriately

          - if features contains any nans for this node, it will be
            ignored. this can be useful if some, but not all, nodes
            have inputs.

          - if features is None, then no features were handed to the
            layer, and it should be ignored.

        '''

        if not (features is None):
            valid_features = tf.reduce_all(tf.logical_not(tf.math.is_nan(features)))

        n_children = len(reprs)
        if n_children > 0:
            reprs = tf.stack(reprs, axis=0)
            all_h = reprs[:, :self.inner_dim]
            all_c = reprs[:, self.inner_dim:2*self.inner_dim]
            h_tilde = tf.reduce_sum(all_h, axis=0)
        else:
            h_tilde = tf.zeros(self.inner_dim)

        if not (features is None) and valid_features:
            trans_x = tf.tensordot(self.input_trans_W, features, axes=1)
        else:
            trans_x = tf.zeros(4 * self.inner_dim)

        trans_htilde = tf.tensordot(self.trans_U, h_tilde, axes=1)

        iou_gates = trans_x[self.inner_dim:] + trans_htilde + self.b[self.inner_dim:]

        i_gate = tf.math.sigmoid(iou_gates[:self.inner_dim])
        o_gate = tf.math.sigmoid(iou_gates[self.inner_dim:2*self.inner_dim])
        u_gate = tf.math.tanh(iou_gates[2*self.inner_dim:])

        c_gate = i_gate * u_gate
        if n_children > 0:
            fjs = []
            for idx in range(reprs.shape[0]):
                tmp = trans_x[:self.inner_dim]
                tmp += tf.tensordot(self.Uf, all_h[idx], axes=1)
                tmp += self.b[:self.inner_dim]
                fjs.append(tf.math.sigmoid(tmp))
            fjs = tf.stack(fjs, axis=0)
            c_gate += tf.reduce_sum(fjs * all_c, axis=0)

        h_gate = o_gate * tf.math.tanh(c_gate)
        return tf.concat([h_gate, c_gate, o_gate], axis=0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dim',
        type=int,
        default=12)
    parser.add_argument(
        '--seed',
        type=int,
        default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    test_tree_structure = [[1, 4], # root node has 3 children
                           [-1,-1], # first child has no children
                           [-1,-1], # second child has no children
                           [4, 6], # third child has 2 children
                           [-1, -1], # no children for first child of third
                           [-1, -1], # no children for second child of third
                           [-1, -1]] # just a dummy padding node
    test_tree_structure = np.expand_dims(test_tree_structure, 0)
    test_tree_zero_features = np.zeros((1, 7, args.dim)).astype(np.float32)

    tree_lstm = ChildSumLSTMTreeLayer(args.dim, dynamic=True)
    res = tree_lstm([test_tree_structure, test_tree_zero_features])
    print(res)
    res = tree_lstm(test_tree_structure)
    print(res)
    test_tree_zero_features[0,0,:] = np.nan
    res = tree_lstm([test_tree_structure, test_tree_zero_features])
    print(res)

    tree_lstm_full = ChildSumLSTMTreeLayer(args.dim, dynamic=True, just_root_output=False)
    res = tree_lstm_full(test_tree_structure)
    print(res)


if __name__ == '__main__':
    main()

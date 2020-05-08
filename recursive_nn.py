'''
Attempt at a tree recursive neural network
'''
import argparse
import tensorflow as tf
import numpy as np


class SimpleTreeLayer(tf.keras.layers.Layer):
    def __init__(self, dim, just_root_output=True, *args, **kwargs):
        super(SimpleTreeLayer, self).__init__(*args, **kwargs)
        self.dim = dim
        self.just_root_output = just_root_output
        self.supports_masking = False

    def build(self, input_shape):
        self.inner_transform = tf.keras.layers.Dense(self.dim, activation='sigmoid')
        self.input_transform = tf.keras.layers.Dense(self.dim, activation='sigmoid')

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 2:
            if self.just_root_output:
                return (input_shape[0][0], self.dim)
            else:
                return (input_shape[0][0], input_shape[0][1], self.dim)
        else:
            if self.just_root_output:
                return (input_shape[0], self.dim)
            else:
                return (input_shape[0], input_shape[1], self.dim)

    def _combine_inner(self, reprs, features):
        '''Combination function:

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

        this returns the output state of the node, and should include output info,
        and info required for computation from higher nodes
        '''
        if not (features is None):
            valid_features = tf.reduce_all(tf.logical_not(tf.math.is_nan(features)))

        if len(reprs) == 0: # base case
            if features is None: # leaf node and no features
                features = tf.zeros(self.dim)
                valid_features = True
            if valid_features:
                return self.input_transform(tf.expand_dims(features, 0))[0]
            else:
                raise NotImplementedError(
                    'Leaf nodes should either have no features or valid features')

        if not (features is None):
            if valid_features:
                reprs += [features]

        reprs = tf.stack(reprs, axis=0)
        c_mean = tf.reduce_mean(reprs, axis=0, keepdims=True)
        c_max = tf.reduce_mean(reprs, axis=0, keepdims=True)
        c_min = tf.reduce_min(reprs, axis=0, keepdims=True)
        trans = self.inner_transform(tf.concat([c_mean, c_max, c_min], axis=1))
        return trans[0]

    def _encode_tree(self, tree_enc, node_features):
        state = [None for _ in range(tree_enc.shape[0])]
        def _encode_tree_rec(cur_idx):
            ch_start, ch_end = tree_enc[cur_idx][0], tree_enc[cur_idx][1]
            if ch_start == -1:
                if node_features is None:
                    state[cur_idx] = self._combine_inner([], node_features)
                else:
                    state[cur_idx] = self._combine_inner([], node_features[cur_idx])

            else:
                for child_idx in range(ch_start, ch_end):
                    _encode_tree_rec(child_idx)
                if node_features is None:
                    state[cur_idx] = self._combine_inner(
                        state[ch_start: ch_end], node_features)
                else:
                    state[cur_idx] = self._combine_inner(
                        state[ch_start: ch_end], node_features[cur_idx])

        _encode_tree_rec(0)
        if self.just_root_output:
            return state[0]
        else:
            # turn the Nones in state, i.e., the padding nodes, into zeros
            state = [s if not (s is None) else tf.zeros(self.dim) for s in state]
            return tf.stack(state, axis=0)

    def call(self, inputs):
        if isinstance(inputs, list):
            assert len(inputs) == 2, 'Inputs must be either [tree, features] or just tree'
            structure, features = inputs
        else:
            structure, features = inputs, [None for _ in range(inputs.shape[0])]
        outputs = []
        for b_idx in range(structure.shape[0]):
            outputs.append(self._encode_tree(structure[b_idx], features[b_idx]))
        return tf.stack(outputs, axis=0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dim',
        type=int,
        default=10)
    parser.add_argument(
        '--seed',
        type=int,
        default=1)
    return parser.parse_args()


def main():
    '''
    Main to show off some features...
    '''

    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # trees are represented by tree = (batch, max_nodes, 2) shaped tensor
    # where tree[i, j] gives the start (inclusive) and end (exclusive) indices
    # of the children.

    # this implementation also supports arbitrary features passed for each internal
    # node.
    test_tree_structure = [[1, 4], # root node has 3 children
                           [-1,-1], # first child has no children
                           [-1,-1], # second child has no children
                           [4, 6], # third child has 2 children
                           [-1, -1], # no children for first child of third
                           [-1, -1], # no children for second child of third
                           [-1, -1]] # just a dummy padding node

    # order of children doesnt matter to simple tree layer
    idx_equiv = np.array([0, 3, 2, 1, 4, 5, 6])
    test_tree_structure_equiv = np.array(test_tree_structure)[idx_equiv]

    # add a batch dim
    test_tree_structure = np.expand_dims(test_tree_structure, 0)
    test_tree_structure_equiv = np.expand_dims(test_tree_structure_equiv, 0)

    simple = SimpleTreeLayer(args.dim, dynamic=True)

    # example without node features. The leaf nodes are assumed to have zero
    # features, and combination happens from there.
    res_no_features = simple(test_tree_structure)
    res_no_features_equiv = simple(test_tree_structure_equiv)
    np.testing.assert_allclose(res_no_features, res_no_features_equiv)

    # equivalent example --- internal node features are ignored by setting
    # to nan, leaf nodes are zero
    test_tree_zero_features = np.zeros((7, args.dim)).astype(np.float32)
    inner_idxs = np.array([0, 3])
    test_tree_zero_features[inner_idxs,:] = np.nan

    test_tree_zero_features = np.expand_dims(test_tree_zero_features, 0)
    test_tree_zero_features_equiv = test_tree_zero_features[:, idx_equiv, :]

    res_no_features_2 = simple([test_tree_structure, test_tree_zero_features])
    res_no_features_2_equiv = simple([test_tree_structure_equiv, test_tree_zero_features_equiv])

    np.testing.assert_allclose(res_no_features, res_no_features_2)
    np.testing.assert_allclose(res_no_features, res_no_features_2_equiv)

    # of course, you can also hand arbitrary input features for each node
    features = np.random.random((1, 7, args.dim)).astype(np.float32)
    features_equiv = features[:, idx_equiv, :]
    res_features = simple([test_tree_structure, features])
    res_features_equiv = simple([test_tree_structure_equiv, features_equiv])

    np.testing.assert_allclose(res_features, res_features_equiv)

    # and you can ignore any input features, if your tree layer supports it
    # ignore root node
    features[0, 0] = np.nan
    res_features_ignore_root = simple([test_tree_structure, features])

    # and you can return the state of all of the nodes, too...
    simple_all_nodes = SimpleTreeLayer(args.dim, just_root_output=False, dynamic=True)
    res_features_ignore_root_all = simple_all_nodes([test_tree_structure, features])

    # padding nodes will be assigned zero
    print(res_features_ignore_root_all)

    # you can use layers in models too

    tree_input = tf.keras.layers.Input((None, 2), dtype='int32')
    tree_features_input = tf.keras.layers.Input((None, 100), dtype='float32')
    model_layer = SimpleTreeLayer(args.dim, dynamic=True, just_root_output=False)
    res = model_layer([tree_input, tree_features_input])
    model = tf.keras.models.Model(inputs=[tree_input, tree_features_input],
                                  outputs=res)
    model.summary()



if __name__ == '__main__':
    main()

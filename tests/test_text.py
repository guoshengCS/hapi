# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# when test, you should add hapi root path to the PYTHONPATH,
# export PYTHONPATH=PATH_TO_HAPI:$PYTHONPATH
import unittest
import time
import random

import numpy as np

import paddle.fluid as fluid
from paddle.fluid.dygraph import Embedding, Linear, Layer
from paddle.fluid.layers import BeamSearchDecoder
import hapi.text as text
from hapi.model import Model, Input, set_device
from hapi.text import BasicLSTMCell, BasicGRUCell, RNN, DynamicDecode, MultiHeadAttention


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def tanh(x):
    return 2. * sigmoid(2. * x) - 1.


def lstm_step(step_in, pre_hidden, pre_cell, gate_w, gate_b, forget_bias=1.0):
    concat_1 = np.concatenate([step_in, pre_hidden], 1)

    gate_input = np.matmul(concat_1, gate_w)
    gate_input += gate_b
    i, j, f, o = np.split(gate_input, indices_or_sections=4, axis=1)

    new_cell = pre_cell * sigmoid(f + forget_bias) + sigmoid(i) * tanh(j)
    new_hidden = tanh(new_cell) * sigmoid(o)

    return new_hidden, new_cell


def gru_step(step_in, pre_hidden, gate_w, gate_b, candidate_w, candidate_b):
    concat_1 = np.concatenate([step_in, pre_hidden], 1)

    gate_input = np.matmul(concat_1, gate_w)
    gate_input += gate_b
    gate_input = sigmoid(gate_input)
    r, u = np.split(gate_input, indices_or_sections=2, axis=1)

    r_hidden = r * pre_hidden

    candidate = np.matmul(np.concatenate([step_in, r_hidden], 1), candidate_w)

    candidate += candidate_b
    c = tanh(candidate)

    new_hidden = u * pre_hidden + (1 - u) * c

    return new_hidden


class ModuleApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._np_rand_state = np.random.get_state()
        cls._py_rand_state = random.getstate()
        cls._random_seed = 123
        np.random.seed(cls._random_seed)
        random.seed(cls._random_seed)

        cls.model_cls = type(cls.__name__ + "Model", (Model, ), {
            "__init__": cls.model_init_wrapper(cls.model_init),
            "forward": cls.model_forward
        })

    @classmethod
    def tearDownClass(cls):
        np.random.set_state(cls._np_rand_state)
        random.setstate(cls._py_rand_state)

    @staticmethod
    def model_init_wrapper(func):
        def __impl__(self, *args, **kwargs):
            Model.__init__(self)
            func(self, *args, **kwargs)

        return __impl__

    @staticmethod
    def model_init(self, *args, **kwargs):
        raise NotImplementedError(
            "model_init acts as `Model.__init__`, thus must implement it")

    @staticmethod
    def model_forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def make_inputs(self):
        # TODO(guosheng): add default from `self.inputs`
        raise NotImplementedError(
            "model_inputs makes inputs for model, thus must implement it")

    def setUp(self):
        """
        For the model which wraps the module to be tested:
            Set input data by `self.inputs` list
            Set init argument values by `self.attrs` dict
            Set model parameter values by `self.param_states` dict
            Set expected output data by `self.outputs` list
        We can create a model instance and run once with these.
        """
        self.inputs = []
        self.attrs = {}
        self.param_states = {}
        self.outputs = {}

    def _calc_output(self, place, mode="test", dygraph=True):
        if dygraph:
            fluid.enable_dygraph(place)
        else:
            fluid.disable_dygraph()
        fluid.default_main_program().random_seed = self._random_seed
        fluid.default_startup_program().random_seed = self._random_seed
        model = self.model_cls(**self.attrs)
        model.prepare(inputs=self.make_inputs(), device=place)
        if self.param_states:
            model.load(self.param_states, optim_state=None)
        return model.test_batch(self.inputs)

    def check_output_with_place(self, place, mode="test"):
        dygraph_output = self._calc_output(place, mode, dygraph=True)
        stgraph_output = self._calc_output(place, mode, dygraph=False)
        expect_output = getattr(self, "outputs", None)
        for actual_t, expect_t in zip(dygraph_output, stgraph_output):
            self.assertTrue(np.allclose(actual_t, expect_t, rtol=1e-5, atol=0))
        if expect_output:
            for actual_t, expect_t in zip(dygraph_output, expect_output):
                self.assertTrue(
                    np.allclose(
                        actual_t, expect_t, rtol=1e-5, atol=0))

    def check_output(self):
        devices = ["GPU"]
        for device in devices:
            place = set_device(device)
            self.check_output_with_place(place)


class TestBasicLSTM(ModuleApiTest):
    def setUp(self):
        shape = (2, 4, 128)
        self.inputs = [np.random.random(shape).astype("float32")]
        self.outputs = None
        self.attrs = {"input_size": 128, "hidden_size": 128}
        self.param_states = {}

    @staticmethod
    def model_init(self, input_size, hidden_size):
        self.lstm = RNN(
            BasicLSTMCell(
                input_size,
                hidden_size,
                param_attr=fluid.ParamAttr(name="lstm_weight"),
                bias_attr=fluid.ParamAttr(name="lstm_bias")))

    @staticmethod
    def model_forward(self, inputs):
        return self.lstm(inputs)[0]

    def make_inputs(self):
        inputs = [
            Input(
                [None, None, self.inputs[-1].shape[-1]],
                "float32",
                name="input")
        ]
        return inputs

    def test_check_output(self):
        self.check_output()


# class TestBasicGRU(ModuleApiTest):
#     def setUp(self):
#         shape = (2, 4, 128)
#         self.inputs = [np.random.random(shape).astype("float32")]
#         self.outputs = None
#         self.attrs = {"input_size": 128, "hidden_size": 128}
#         self.param_states = {}

#     @staticmethod
#     def model_init(self, input_size, hidden_size):
#         self.gru = RNN(BasicGRUCell(input_size, hidden_size))

#     @staticmethod
#     def model_forward(self, inputs):
#         return self.gru(inputs)[0]

#     def make_inputs(self):
#         inputs = [
#             Input([None, None, self.inputs[-1].shape[-1]],
#                   "float32",
#                   name="input")
#         ]
#         return inputs

#     def test_check_output(self):
#         self.check_output()


class TestBeamSearch(ModuleApiTest):
    def setUp(self):
        shape = (8, 32)
        self.inputs = [
            np.random.random(shape).astype("float32"),
            np.random.random(shape).astype("float32")
        ]
        self.outputs = None
        self.attrs = {
            "vocab_size": 100,
            "embed_dim": 32,
            "hidden_size": 32,
        }
        self.param_states = {}

    @staticmethod
    def model_init(self,
                   vocab_size,
                   embed_dim,
                   hidden_size,
                   bos_id=0,
                   eos_id=1,
                   beam_size=4,
                   max_step_num=20):
        embedder = Embedding(size=[vocab_size, embed_dim])
        output_layer = Linear(hidden_size, vocab_size)
        cell = BasicLSTMCell(embed_dim, hidden_size)
        decoder = BeamSearchDecoder(
            cell,
            start_token=bos_id,
            end_token=eos_id,
            beam_size=beam_size,
            embedding_fn=embedder,
            output_fn=output_layer)
        self.beam_search_decoder = DynamicDecode(
            decoder, max_step_num=max_step_num, is_test=True)

    @staticmethod
    def model_forward(self, init_hidden, init_cell):
        return self.beam_search_decoder([init_hidden, init_cell])[0]

    def make_inputs(self):
        inputs = [
            Input(
                [None, self.inputs[0].shape[-1]],
                "float32",
                name="init_hidden"), Input(
                    [None, self.inputs[1].shape[-1]],
                    "float32",
                    name="init_cell")
        ]
        return inputs

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
